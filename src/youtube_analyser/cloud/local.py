"""The Store and Queue on one machine: a folder, and an in-process or folder-backed queue."""
import json
import os
import queue as _queue
import threading
import time
import uuid
from pathlib import Path

from .base import Message, Queue, Store


class FileStore(Store):
    """Keys are paths under a folder ("jobs/abc/sections.csv" -> <root>/jobs/abc/sections.csv)."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key):
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents:
            raise KeyError(f"{key!r} is outside the store")
        return p

    def put(self, key, data):
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(data)
        tmp.replace(p)            # readers never see half a file

    def put_file(self, key, path):
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp")
        with open(path, "rb") as src, open(tmp, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        tmp.replace(p)

    def put_stream(self, key, chunks):
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp, size = p.with_name(f".{p.name}.{uuid.uuid4().hex}.tmp"), 0
        try:
            with open(tmp, "wb") as f:
                for chunk in chunks:
                    size += len(chunk)
                    f.write(chunk)
            tmp.replace(p)
        finally:
            tmp.unlink(missing_ok=True)
        return size

    def get(self, key):
        try:
            return self.path(key).read_bytes()
        except (FileNotFoundError, IsADirectoryError):
            raise KeyError(key) from None

    def get_file(self, key, path):
        src = self.path(key)
        if not src.is_file():
            raise KeyError(key)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(src, "rb") as a, open(path, "wb") as b:
            while chunk := a.read(1 << 20):
                b.write(chunk)

    def exists(self, key):
        return self.path(key).is_file()

    def delete(self, key):
        p = self.path(key)
        p.unlink(missing_ok=True)
        root = self.root.resolve()
        for parent in p.parents:  # tidy empty folders, up to the root
            if parent == root:
                break
            try:
                parent.rmdir()
            except FileNotFoundError:
                continue
            except OSError:   # not empty
                break

    def list(self, prefix=""):
        root = self.root.resolve()
        start = root / prefix.rsplit("/", 1)[0] if "/" in prefix else root
        if not start.is_dir():
            return []
        keys = (p.relative_to(root).as_posix() for p in start.rglob("*") if p.is_file() and not p.name.endswith(".tmp"))
        return sorted(k for k in keys if k.startswith(prefix))


class MemoryStore(Store):
    """Files in a dict, inside one process (for tests, and as the smallest example of a Store:
    only the four required methods; the others come from Store)."""

    def __init__(self):
        self._files = {}
        self._lock = threading.Lock()

    def put(self, key, data):
        with self._lock:
            self._files[key] = bytes(data)

    def get(self, key):
        with self._lock:
            return self._files[key]

    def delete(self, key):
        with self._lock:
            self._files.pop(key, None)

    def list(self, prefix=""):
        with self._lock:
            return sorted(k for k in self._files if k.startswith(prefix))


class MemoryQueue(Queue):
    """A queue inside one process (the service with its own workers on one machine)."""

    def __init__(self):
        self._q = _queue.Queue()
        self._out = {}
        self._lock = threading.Lock()

    def send(self, body):
        self._q.put(Message(uuid.uuid4().hex, json.loads(json.dumps(body)), dequeue_count=0))

    def receive(self, visibility_s=600, wait_s=0):
        self._requeue_expired()
        try:
            m = self._q.get(timeout=wait_s) if wait_s else self._q.get_nowait()
        except _queue.Empty:
            return None
        m.dequeue_count += 1
        with self._lock:
            self._out[m.id] = (m, time.monotonic() + visibility_s)
        return m

    def _requeue_expired(self):
        now = time.monotonic()
        with self._lock:
            late = [m for m, until in self._out.values() if until <= now]
            for m in late:
                del self._out[m.id]
        for m in late:
            self._q.put(m)

    def done(self, message):
        with self._lock:
            self._out.pop(message.id, None)

    def retry(self, message):
        with self._lock:
            if self._out.pop(message.id, None) is not None:
                self._q.put(message)

    def count(self):
        return self._q.qsize()


class FileQueue(Queue):
    """A queue in a folder, shared by the processes of one machine (or a network folder):
    <root>/ready/<message>.json, taken by renaming it into <root>/taken/ (only one rename wins)."""

    def __init__(self, root):
        self.root = Path(root)
        for sub in ("ready", "taken"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def send(self, body):
        name = f"{time.time_ns():020d}-{uuid.uuid4().hex}.json"
        tmp = self.root / f".{name}.tmp"
        tmp.write_text(json.dumps({"body": body, "count": 0}))
        tmp.replace(self.root / "ready" / name)

    def receive(self, visibility_s=600):
        self._requeue_expired()
        for p in sorted((self.root / "ready").glob("*.json")):
            taken = self.root / "taken" / p.name
            try:
                p.replace(taken)          # atomic: if another process took it first, this fails
            except FileNotFoundError:
                continue
            data = json.loads(taken.read_text())
            data["count"] += 1
            data["until"] = time.time() + visibility_s
            taken.write_text(json.dumps(data))
            return Message(p.stem, data["body"], receipt=taken, dequeue_count=data["count"])
        return None

    def _requeue_expired(self):
        now = time.time()
        for p in (self.root / "taken").glob("*.json"):
            try:
                if json.loads(p.read_text()).get("until", now + 1) <= now:
                    p.replace(self.root / "ready" / p.name)
            except (FileNotFoundError, ValueError):
                continue

    def done(self, message):
        Path(message.receipt).unlink(missing_ok=True)

    def retry(self, message):
        try:
            Path(message.receipt).replace(self.root / "ready" / Path(message.receipt).name)
        except FileNotFoundError:
            pass

    def count(self):
        return len(list((self.root / "ready").glob("*.json")))


def file_url_path(url):
    """file:///a/b -> /a/b; a plain path stays as it is."""
    if url.startswith("file://"):
        return url[len("file://"):] or "/"
    return os.path.expanduser(url)

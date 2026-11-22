"""What the service, the workers and distributed channel searches need from a cloud: somewhere
to keep files (a Store) and a queue of work (a Queue). Everything else is the same code whichever
provider is behind them, so providers are swappable: local folders for one machine, Azure Blob
Storage and Queue Storage for Azure, or anything else registered with cloud.register_store /
cloud.register_queue (or a "youtube_analyser.stores" / "youtube_analyser.queues" entry point).
"""
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


class Store(ABC):
    """Files by key ("jobs/abc/sections.csv"): a folder, a blob container, a bucket."""

    @abstractmethod
    def put(self, key, data):
        """Save bytes under key (replacing what was there)."""

    @abstractmethod
    def get(self, key):
        """The bytes under key; KeyError if there are none."""

    @abstractmethod
    def delete(self, key):
        """Remove key (nothing happens if it is not there)."""

    @abstractmethod
    def list(self, prefix=""):
        """The keys starting with prefix, sorted."""

    def exists(self, key):
        try:
            self.get(key)
            return True
        except KeyError:
            return False

    def put_file(self, key, path):
        """Upload a local file (providers stream it where they can)."""
        self.put(key, Path(path).read_bytes())

    def get_file(self, key, path):
        """Download key to a local file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(self.get(key))

    def put_stream(self, key, chunks):
        """Save what an iterator of byte chunks gives (an upload), without holding it all in memory
        where the provider allows. Returns the number of bytes."""
        data = b"".join(chunks)
        self.put(key, data)
        return len(data)

    def put_json(self, key, obj):
        self.put(key, json.dumps(obj, default=_plain).encode())

    def get_json(self, key):
        return json.loads(self.get(key))


def _plain(o):
    """numpy numbers and sets as JSON."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


@dataclass
class Message:
    """A piece of work taken from a queue; give it back to the queue with done() or retry()."""
    id: str
    body: dict
    receipt: object = None
    dequeue_count: int = 1
    extra: dict = field(default_factory=dict)


class Queue(ABC):
    """Work to do, each message handled once: taken by one worker and, if that worker dies
    before saying it is done, offered again after visibility_s."""

    @abstractmethod
    def send(self, body):
        """Add a message (a JSON-able dict)."""

    @abstractmethod
    def receive(self, visibility_s=600):
        """The next message, hidden from other workers for visibility_s, or None if there is none."""

    @abstractmethod
    def done(self, message):
        """The message has been handled: remove it."""

    def retry(self, message):  # noqa: B027 - optional: by default the message comes back when its visibility runs out
        """Offer the message again at once (the default: when its visibility runs out)."""

    def extend(self, message, visibility_s):  # noqa: B027 - optional: providers that cannot do it lose nothing but time
        """Keep the message hidden for another visibility_s (a worker still busy with it)."""

    def count(self):
        """Messages waiting (approximate where the provider only estimates), or None if unknown."""
        return None


def copy_tree(store, prefix, folder):
    """Download every key under prefix into folder (the rest of the key as the path)."""
    folder = Path(folder)
    for key in store.list(prefix):
        rel = key[len(prefix):].lstrip("/")
        if rel:
            store.get_file(key, folder / rel)


def upload_tree(store, prefix, folder):
    """Upload every file under folder to prefix/<relative path>."""
    folder = Path(folder)
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            store.put_file(f"{prefix.rstrip('/')}/{p.relative_to(folder).as_posix()}", p)

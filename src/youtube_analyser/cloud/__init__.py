"""Swappable storage and queues (see base.py): open_store(url) and open_queue(url).

    Store URLs                               Queue URLs
    /path/to/folder, file:///path            memory://              (one process)
    memory://      (one process)             file:///path           (processes sharing a folder)
    azblob://<account>/<container>           azqueue://<account>/<queue>

Other providers plug in with register_store / register_queue, or through the package entry
points "youtube_analyser.stores" and "youtube_analyser.queues" (name: the URL scheme; value: a
callable taking the URL and returning a Store or Queue).
"""
from importlib.metadata import entry_points
from urllib.parse import urlparse

from .base import Message, Queue, Store, copy_tree, upload_tree
from .local import FileQueue, FileStore, MemoryQueue, MemoryStore, file_url_path

__all__ = ["Store", "Queue", "Message", "FileStore", "FileQueue", "MemoryQueue", "MemoryStore", "open_store", "open_queue",
           "register_store", "register_queue", "copy_tree", "upload_tree"]


def _azure_store(url):
    from .azure import AzureBlobStore
    u = urlparse(url)
    return AzureBlobStore(u.path.strip("/"), account=u.netloc)


def _azure_queue(url):
    from .azure import AzureQueue
    u = urlparse(url)
    return AzureQueue(u.path.strip("/"), account=u.netloc)


_STORES = {"file": lambda url: FileStore(file_url_path(url)), "memory": lambda url: MemoryStore(), "azblob": _azure_store}
_QUEUES = {"memory": lambda url: MemoryQueue(), "file": lambda url: FileQueue(file_url_path(url)), "azqueue": _azure_queue}


def register_store(scheme, factory):
    _STORES[scheme] = factory


def register_queue(scheme, factory):
    _QUEUES[scheme] = factory


def _open(url, known, group, what):
    scheme = urlparse(str(url)).scheme if "://" in str(url) else "file"
    if scheme not in known:
        for ep in entry_points(group=group):
            if ep.name == scheme:
                known[scheme] = ep.load()
    if scheme not in known:
        raise ValueError(f"no {what} for {url!r}; known: {', '.join(sorted(known))}")
    return known[scheme](str(url))


def open_store(url):
    """A Store from a URL (a plain path is a folder)."""
    return _open(url, _STORES, "youtube_analyser.stores", "store")


def open_queue(url):
    """A Queue from a URL."""
    return _open(url, _QUEUES, "youtube_analyser.queues", "queue")

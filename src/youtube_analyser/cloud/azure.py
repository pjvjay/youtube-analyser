"""The Store and Queue on Azure: Blob Storage and Queue Storage.

    azblob://<account>/<container>     files in a blob container
    azqueue://<account>/<queue>        messages in a storage queue

They sign in with AZURE_STORAGE_CONNECTION_STRING when it is set (account keys, or the Azurite
emulator for local testing), and otherwise with the machine's Azure identity (a container app's
managed identity, or `az login` on a laptop) through azure-identity's DefaultAzureCredential:
give that identity the "Storage Blob Data Contributor" and "Storage Queue Data Message Processor"
/ "Sender" roles on the account. Needs the [azure] extra: pip install "youtube-analyser[azure]".
"""
import json
import os
from pathlib import Path

from .base import Message, Queue, Store


def _credential():
    from azure.identity import DefaultAzureCredential
    return DefaultAzureCredential()


class AzureBlobStore(Store):
    """Keys are blob names in one container (created if missing)."""

    def __init__(self, container, account=None, connection_string=None, credential=None, client=None):
        from azure.core.exceptions import ResourceExistsError
        from azure.storage.blob import ContainerClient
        connection_string = connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if client is None:
            if connection_string:
                client = ContainerClient.from_connection_string(connection_string, container)
            else:
                client = ContainerClient(f"https://{account}.blob.core.windows.net", container,
                                         credential=credential or _credential())
        self.client = client
        try:
            self.client.create_container()
        except ResourceExistsError:
            pass

    def put(self, key, data):
        self.client.upload_blob(key, data, overwrite=True)

    def put_file(self, key, path):
        with open(path, "rb") as f:
            self.client.upload_blob(key, f, overwrite=True)

    def put_stream(self, key, chunks):
        counted = _Counted(chunks)
        self.client.upload_blob(key, counted, overwrite=True)
        return counted.size

    def get(self, key):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            return self.client.download_blob(key).readall()
        except ResourceNotFoundError:
            raise KeyError(key) from None

    def get_file(self, key, path):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            downloader = self.client.download_blob(key)
        except ResourceNotFoundError:
            raise KeyError(key) from None
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            downloader.readinto(f)

    def exists(self, key):
        return self.client.get_blob_client(key).exists()

    def delete(self, key):
        from azure.core.exceptions import ResourceNotFoundError
        try:
            self.client.delete_blob(key)
        except ResourceNotFoundError:
            pass

    def list(self, prefix=""):
        return sorted(b.name for b in self.client.list_blobs(name_starts_with=prefix or None))


class _Counted:
    """An iterator of byte chunks that counts them (upload_blob takes any iterable)."""

    def __init__(self, chunks):
        self.chunks, self.size = iter(chunks), 0

    def __iter__(self):
        return self

    def __next__(self):
        chunk = next(self.chunks)
        self.size += len(chunk)
        return chunk


class AzureQueue(Queue):
    """A storage queue (created if missing). Messages are JSON text, up to 64 KB."""

    def __init__(self, name, account=None, connection_string=None, credential=None, client=None):
        from azure.core.exceptions import ResourceExistsError
        from azure.storage.queue import QueueClient
        connection_string = connection_string or os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if client is None:
            if connection_string:
                client = QueueClient.from_connection_string(connection_string, name)
            else:
                client = QueueClient(f"https://{account}.queue.core.windows.net", name, credential=credential or _credential())
        self.client = client
        try:
            self.client.create_queue()
        except ResourceExistsError:
            pass

    def send(self, body):
        self.client.send_message(json.dumps(body))

    def receive(self, visibility_s=600):
        for m in self.client.receive_messages(max_messages=1, visibility_timeout=int(visibility_s)):
            return Message(m.id, json.loads(m.content), receipt=m.pop_receipt, dequeue_count=m.dequeue_count)
        return None

    def done(self, message):
        self.client.delete_message(message.id, message.receipt)

    def retry(self, message):
        self.client.update_message(message.id, message.receipt, visibility_timeout=0)

    def extend(self, message, visibility_s):
        updated = self.client.update_message(message.id, message.receipt, visibility_timeout=int(visibility_s))
        message.receipt = updated.pop_receipt   # the old receipt is no longer valid

    def count(self):
        return self.client.get_queue_properties().approximate_message_count

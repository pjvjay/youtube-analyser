"""pytest setup: the Azure Storage emulator (Azurite) for the cloud tests.

The Azure tests run against Azurite, started here when it is installed (`npm install -g azurite`,
or AZURITE=/path/to/azurite), or against the account in AZURE_STORAGE_CONNECTION_STRING when that
is set. Without either they are skipped.
"""
import os
import shutil
import socket
import subprocess
import tempfile
import time

import pytest

ACCOUNT_KEY = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="  # Azurite's well-known key


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def azure_connection():
    """A connection string for Azure Storage (Azurite, started for the session), or skip."""
    if os.environ.get("AZURE_STORAGE_CONNECTION_STRING"):
        yield os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        return
    pytest.importorskip("azure.storage.blob")
    binary = os.environ.get("AZURITE") or shutil.which("azurite") or next(
        (p for p in ("/tmp/az/node_modules/.bin/azurite", os.path.expanduser("~/node_modules/.bin/azurite"))
         if os.path.exists(p)), None)
    if not binary:
        pytest.skip("Azurite is not installed (npm install -g azurite)")
    blob, queue = _free_port(), _free_port()
    with tempfile.TemporaryDirectory() as data:
        proc = subprocess.Popen([binary, "--silent", "--skipApiVersionCheck", "--location", data, "--blobPort", str(blob),
                                 "--queuePort", str(queue), "--tablePort", str(_free_port())],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                try:
                    socket.create_connection(("127.0.0.1", queue), timeout=0.2).close()
                    socket.create_connection(("127.0.0.1", blob), timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.1)
            yield (f"DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey={ACCOUNT_KEY};"
                   f"BlobEndpoint=http://127.0.0.1:{blob}/devstoreaccount1;"
                   f"QueueEndpoint=http://127.0.0.1:{queue}/devstoreaccount1;")
        finally:
            proc.terminate()
            proc.wait(10)


@pytest.fixture
def azure_env(azure_connection, monkeypatch):
    """AZURE_STORAGE_CONNECTION_STRING set for the test (what the azblob:// and azqueue:// URLs use)."""
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", azure_connection)
    return azure_connection

"""The swappable cloud pieces (cloud/, jobs.py): every Store and Queue keeps the same promises, and
the same Worker does the same work whichever is behind it. Azure's run against the Azurite
emulator (see conftest.py).

    pytest tests/test_cloud.py
"""
import json
import os
import sys
import time
import uuid

import pytest

from tests import helpers as h
from tests.helpers import mf
from youtube_analyser import channel, cloud, jobs
from youtube_analyser.cloud import azure as az
from youtube_analyser.youtube import YtDlp

FAKE = f"{sys.executable} {h.REPO / 'tests' / 'fake_yt_dlp.py'}"


def name():
    return "t" + uuid.uuid4().hex[:12]


@pytest.fixture(params=["file", "memory", "azure"])
def store(request, tmp_path):
    if request.param == "file":
        return cloud.FileStore(tmp_path / "store")
    if request.param == "memory":
        return cloud.open_store("memory://")
    request.getfixturevalue("azure_env")
    return cloud.open_store(f"azblob://devstoreaccount1/{name()}")


@pytest.fixture(params=["memory", "file", "azure"])
def task_queue(request, tmp_path):
    if request.param == "memory":
        return cloud.MemoryQueue()
    if request.param == "file":
        return cloud.open_queue(f"file://{tmp_path / 'queue'}")
    request.getfixturevalue("azure_env")
    return cloud.open_queue(f"azqueue://devstoreaccount1/{name()}")


# ---- stores ------------------------------------------------------------------------------------

def test_a_store_keeps_files_by_key(store, tmp_path):
    store.put("a/b/one.txt", b"one")
    store.put("a/b/one.txt", b"one again")                 # replaced
    store.put_json("a/two.json", {"n": 2, "s": {3, 1}})
    assert store.get("a/b/one.txt") == b"one again"
    assert store.get_json("a/two.json") == {"n": 2, "s": [1, 3]}
    assert store.exists("a/two.json") and not store.exists("a/three")
    with pytest.raises(KeyError):
        store.get("a/three")
    with pytest.raises(KeyError):
        store.get_file("a/three", tmp_path / "x")
    src = tmp_path / "big.bin"
    src.write_bytes(os.urandom(3 << 20))
    store.put_file("files/big.bin", src)
    store.get_file("files/big.bin", tmp_path / "back" / "big.bin")
    assert (tmp_path / "back" / "big.bin").read_bytes() == src.read_bytes()
    assert store.put_stream("files/streamed", iter([b"ab", b"cd", b""])) == 4
    assert store.get("files/streamed") == b"abcd"
    assert store.list("a/") == ["a/b/one.txt", "a/two.json"]
    assert store.list("files/") == ["files/big.bin", "files/streamed"]
    assert store.list("a/b/on") == ["a/b/one.txt"]
    assert store.list("nothing/") == []
    cloud.copy_tree(store, "a", tmp_path / "copy")
    assert (tmp_path / "copy" / "b" / "one.txt").read_bytes() == b"one again"
    cloud.upload_tree(store, "up", tmp_path / "copy")
    assert store.list("up/") == ["up/b/one.txt", "up/two.json"]
    store.delete("a/b/one.txt")
    store.delete("a/b/one.txt")                             # deleting twice is fine
    assert store.list("a/") == ["a/two.json"]


def test_a_folder_store_stays_in_its_folder(tmp_path):
    s = cloud.FileStore(tmp_path / "s")
    with pytest.raises(KeyError):
        s.put("../outside", b"x")
    with pytest.raises(KeyError):
        s.get("../../etc/passwd")
    s.put("deep/er/f", b"x")
    s.delete("deep/er/f")
    assert not (tmp_path / "s" / "deep").exists()           # emptied folders are tidied away
    with pytest.raises(ValueError):                         # a failed upload leaves nothing behind
        s.put_stream("bad", (b if b else (_ for _ in ()).throw(ValueError("boom")) for b in [b"a", b""]))
    assert s.list() == []


# ---- queues ------------------------------------------------------------------------------------

def test_a_queue_hands_out_each_message_once(task_queue):
    assert task_queue.receive() is None
    task_queue.send({"n": 1})
    task_queue.send({"n": 2})
    assert task_queue.count() in (2, None)
    a = task_queue.receive(visibility_s=60)
    b = task_queue.receive(visibility_s=60)
    assert {a.body["n"], b.body["n"]} == {1, 2}
    assert task_queue.receive() is None                     # both taken
    task_queue.done(a)
    task_queue.retry(b)                                     # given back at once
    again = task_queue.receive(visibility_s=1)
    assert again.body == b.body and again.dequeue_count == 2
    time.sleep(2.1)                                         # its taker died: it comes back
    late = task_queue.receive(visibility_s=60)
    assert late is not None and late.body == b.body
    task_queue.done(late)
    assert task_queue.receive() is None


def test_urls_and_plugins(tmp_path, monkeypatch):
    assert isinstance(cloud.open_store(str(tmp_path / "s")), cloud.FileStore)
    assert isinstance(cloud.open_store(f"file://{tmp_path / 's2'}"), cloud.FileStore)
    assert isinstance(cloud.open_queue("memory://"), cloud.MemoryQueue)
    with pytest.raises(ValueError, match="no store for"):
        cloud.open_store("s3://bucket")
    with pytest.raises(ValueError, match="no queue for"):
        cloud.open_queue("sqs://queue")
    made = []
    cloud.register_store("mem", lambda url: made.append(url) or cloud.FileStore(tmp_path / "mem"))
    assert isinstance(cloud.open_store("mem://x"), cloud.FileStore) and made == ["mem://x"]

    class EntryPoint:
        name = "plugged"

        def load(self):
            return lambda url: cloud.MemoryQueue()
    monkeypatch.setattr(cloud, "entry_points", lambda group: [EntryPoint()] if group == "youtube_analyser.queues" else [])
    assert isinstance(cloud.open_queue("plugged://anything"), cloud.MemoryQueue)


def test_azure_without_a_connection_string_signs_in_with_the_identity(monkeypatch):
    """Without AZURE_STORAGE_CONNECTION_STRING the account's address and the machine's identity are
    used (checked with stand-ins for the Azure clients)."""
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    seen = {}

    class Client:
        def __init__(self, url, name, credential=None):
            seen[name] = (url, credential)

        def create_container(self):
            from azure.core.exceptions import ResourceExistsError
            raise ResourceExistsError("there already")

        create_queue = create_container

    monkeypatch.setattr("azure.storage.blob.ContainerClient", Client)
    monkeypatch.setattr("azure.storage.queue.QueueClient", Client)
    monkeypatch.setattr(az, "_credential", lambda: "managed identity")
    az.AzureBlobStore("videos", account="acct")
    az.AzureQueue("work", account="acct")
    assert seen == {"videos": ("https://acct.blob.core.windows.net", "managed identity"),
                    "work": ("https://acct.queue.core.windows.net", "managed identity")}


# ---- the same worker, whatever is behind it -----------------------------------------------------

def upload_job(store, task_queue, clip, action):
    js = jobs.JobStore(store)
    job = js.new(f"{clip.name}", action, size=clip.stat().st_size)
    store.put_file(f"uploads/{job['job_id']}", clip)
    task_queue.send({"kind": "analyse", "job": job["job_id"]})
    return js, job["job_id"]


def test_an_uploaded_video_is_analysed_by_a_worker(store, task_queue):
    js, job_id = upload_job(store, task_queue, mf.build("walk_stop_walk"), "walking")
    task_queue.send({"kind": "mystery"})                   # unknown work is dropped, and said so
    said = h.Captured()
    jobs.Worker(store, task_queue, h.SETTINGS, log=said).run(until_empty=True)
    status = js.get(job_id)
    assert status["status"] == "done" and status["video_deleted"], status
    assert not store.exists(f"uploads/{job_id}")
    h.run("walking", ["walk_stop_walk"])
    got = [(r["start_s"], r["end_s"]) for r in h.csv_rows_text(js.result(job_id, "sections.csv"))]
    assert got == [(r["start_s"], r["end_s"]) for r in h.sections("walk_stop_walk", "walking")]
    assert "unknown kind of work" in said.stderr
    assert task_queue.receive() is None


def test_a_job_deleted_before_it_starts_is_skipped(tmp_path):
    store, q = cloud.FileStore(tmp_path / "s"), cloud.MemoryQueue()
    js, job_id = upload_job(store, q, mf.build("walk_stop_walk"), "walking")
    js.delete(job_id)
    jobs.Worker(store, q, h.SETTINGS).run(until_empty=True)
    assert store.list() == []


# ---- a channel search spread over workers -------------------------------------------------------

CHANNEL = [("duoPurple01", "Guest with purple hair | EP 3", "20240301", "duo_purple"),
           ("duoNatural1", "Another guest | EP 2", "20240215", "duo_natural"),
           ("jumpThree01", "Jumping about | EP 1", "20240101", "jump_three")]


@pytest.fixture
def fake_channel(tmp_path, monkeypatch):
    videos = [{"id": i, "title": t, "upload_date": d, "clip": str(mf.build(c))} for i, t, d, c in CHANNEL]
    (tmp_path / "channel.json").write_text(json.dumps({"videos": videos}))
    monkeypatch.setenv("FAKE_CHANNEL", str(tmp_path / "channel.json"))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.log"))
    return tmp_path


def test_a_channel_search_on_workers_matches_one_machine(fake_channel, store, tmp_path, monkeypatch):
    q = cloud.FileQueue(tmp_path / "queue")
    cq = channel.ChannelQuery("woman with purple hair talking", intros="keep")
    search_id = channel.enqueue_channel("https://www.youtube.com/@Fake", cq, store, q, YtDlp(FAKE), log=h.Captured())
    assert q.count() == 3
    monkeypatch.setenv("FAKE_FAIL", "jumpThree01")         # this one's download fails every time
    worker = jobs.Worker(store, q, h.SETTINGS, yt=YtDlp(FAKE), log=h.Captured())
    worker.run(until_empty=True)
    assert q.count() == 0
    failed = store.list(f"searches/{search_id}/failed/")
    assert failed == [f"searches/{search_id}/failed/jumpThree01.json"]
    done, total = channel.collect_channel(store, search_id, tmp_path / "report", log=h.Captured())
    assert (done, total) == (2, 3)
    rows = h.csv_rows(tmp_path / "report" / "matches.csv")
    assert {r["video_id"] for r in rows} == {"duoPurple01"}
    assert all((tmp_path / "report" / "thumbs" / r["thumb"]).exists() for r in rows)
    # the same search on one machine gives the same moments
    monkeypatch.setenv("FAKE_FAIL", "")
    local, _ = channel.run_channel("https://www.youtube.com/@Fake", cq, tmp_path / "work", YtDlp(FAKE), log=h.Captured())
    assert [(r["start_s"], r["end_s"]) for r in local["duoPurple01"]["rows"]] == [(float(r["start_s"]), float(r["end_s"])) for r in rows]
    # enqueueing again sends only what is not done yet
    assert channel.enqueue_channel("https://www.youtube.com/@Fake", cq, store, q, YtDlp(FAKE), log=h.Captured()) == search_id
    assert q.count() == 1


def test_enrolled_voices_and_the_intro_bank_travel_with_the_search(fake_channel, tmp_path, monkeypatch):
    """Names are fingerprinted where the search starts (the workers have no voice recordings) and
    the intro bank is built there from the first long videos."""
    store, q = cloud.FileStore(tmp_path / "s"), cloud.MemoryQueue()
    jane = tmp_path / "jane.wav"
    import subprocess
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(mf.build("duo_purple")), "-t", "8", "-vn", str(jane)], check=True)
    cq = channel.ChannelQuery("Jane talking", config_text=f'[[people]]\nname = "Jane"\nvoice_file = "{jane}"\n')
    search_id = channel.enqueue_channel("https://www.youtube.com/@Fake", cq, store, q, YtDlp(FAKE), log=h.Captured())
    spec = store.get_json(f"searches/{search_id}/spec.json")
    assert list(spec["enrolled"]) == ["Jane"] and spec["bank"] is None     # the test clips are too short for a bank
    jobs.Worker(store, q, h.SETTINGS, yt=YtDlp(FAKE), log=h.Captured()).run(until_empty=True)
    results = channel.done_results(store, f"searches/{search_id}")
    assert [r["evidence"] for r in results["duoPurple01"]["rows"]] == ["voice of Jane [yes]"] * 2
    assert results["duoNatural1"]["rows"] == []


def test_the_command_line_starts_and_collects_a_cloud_search(fake_channel, tmp_path):
    store, queue_dir = tmp_path / "store", tmp_path / "queue"
    r = h.yta("channel", "https://www.youtube.com/@Fake", "people jumping", "--store", store, "--queue",
              f"file://{queue_dir}", "--yt-dlp", FAKE, "--intros", "keep")
    assert r.code == 0, r.stderr
    search_id = r.stdout.split("yta collect ")[1].split()[0]
    r = h.yta("worker", "--store", store, "--queue", f"file://{queue_dir}", "--until-empty", "--yt-dlp", FAKE, *h.OPTIONS)
    assert r.code == 0, r.stderr
    r = h.yta("collect", search_id, "--store", store, "--out", tmp_path / "report")
    assert "3 of 3 videos searched; 3 results" in r.stdout
    assert "jumpThree01&t=3s" in (tmp_path / "report" / "report.md").read_text()
    r = h.yta("channel", "https://www.youtube.com/@Fake", "people jumping", "--store", store)
    assert r.code == 2 and "go together" in r.stderr

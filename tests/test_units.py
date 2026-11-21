"""Small pieces on their own (no models): file names and links, the intro bank's edge cases,
yt-dlp's listings and downloads (with a stand-in), the report writer, queries and votes.

    pytest tests/test_units.py
"""
import json
import subprocess
import sys

import numpy as np
import pytest

from tests import helpers as h
from youtube_analyser import audio, channel, intro, media, people, query, youtube
from youtube_analyser.cloud import FileStore

FAKE = f"{sys.executable} {h.REPO / 'tests' / 'fake_yt_dlp.py'}"


def test_names_and_links():
    assert media.youtube_id("20240105_abcdefghijk.mp4") == "abcdefghijk"
    assert media.upload_date("x/20240105_abcdefghijk.webm") == "20240105"
    assert media.youtube_url("20240105_abcdefghijk.mkv", 61.9) == "https://www.youtube.com/watch?v=abcdefghijk&t=61s"
    assert media.youtube_url("holiday.mp4", 3) == "" and media.youtube_id("holiday.mp4") is None
    assert media.is_video("a.MP4") and not media.is_video("a.f251.webm") and not media.is_video("notes.txt")
    assert media.hms(3725.4) == "1:02:05"
    assert media.picture_time(10, 1.0) == 10.5


def test_probing_what_is_not_a_video(tmp_path):
    (tmp_path / "x.mp4").write_bytes(b"nope")
    with pytest.raises(ValueError, match="no readable video stream"):
        media.probe(tmp_path / "x.mp4")
    sound = tmp_path / "sound.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=d=2", str(sound)], check=True)
    with pytest.raises(ValueError):
        media.probe(sound)
    assert media.duration(tmp_path / "x.mp4") == 0.0
    assert media.load_audio(tmp_path / "x.mp4") is None


def test_a_sideways_phone_video_is_read_upright(tmp_path):
    src, rotated = tmp_path / "src.mp4", tmp_path / "rotated.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=256x144:d=2:r=25", str(src)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(src), "-c", "copy", str(rotated)], check=True)
    assert media.probe(src) == (256, 144)
    w, h_ = media.probe(rotated)
    assert (w, h_) == (144, 256)
    frames = list(media.read_frames(rotated, 1.0, w, h_))
    assert len(frames) == 2 and frames[0][1].shape == (256, 144, 3)
    # every sample is a real picture: the test pattern's colour bars, not scrambled rows
    assert frames[0][1].std() > 30
    assert media.grab_frame(rotated, 99, 1.0, w, h_).max() == 0   # past the end: black


def test_keyframes_are_spaced(tmp_path):
    v = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=256x144:d=20:r=25", "-g", "25", str(v)], check=True)
    frames = list(media.keyframes(v, 256, 144, every=4))
    assert 4 <= len(frames) <= 6                          # a keyframe every second, at most one every 4 s
    assert len(list(media.keyframes(v, 256, 144, every=1))) >= 10


def test_the_intro_bank_without_shared_titles():
    empty = intro.build_bank({})
    assert len(empty["thumbs"]) == 0 and intro.title_frames(empty) == 0 and intro.title_frames(None) == 0
    rng = np.random.default_rng(0)
    # four long openings with nothing in common: nothing is title material
    openings = {f"v{k}": rng.integers(0, 255, (intro.OPENING_S, intro.H, intro.W), np.uint8) for k in range(4)}
    assert intro.title_frames(intro.build_bank(openings)) == 0
    assert intro.start_row(h.REPO / "missing.mp4", None)["how"] == "intro kept"


def test_votes_are_decided_the_same_every_time():
    # two colours seen equally often: the first seen wins, whatever Python's hash seed
    assert people._vote(["purple", "blue", "blue", "purple"], 2) == ("purple", 0.5, 4)
    assert people._vote(["purple"], 2) == (None, 0.0, 1)
    code = ("from youtube_analyser import people;"
            "print(people._vote(['blue','red','red','blue','green','green'], 2)[0])")
    seen = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"PYTHONHASHSEED": str(s), "PATH": ""}).stdout.strip() for s in range(6)}
    assert seen == {"blue"}


def test_hearing_with_no_measured_sound():
    assert audio.heard(None, 0, 1) == (False, 0.0, [])
    assert audio.heard({"snr_db": np.zeros(0), "speech": np.zeros(0, bool)}, 0, 1) == (False, 0.0, [])


def test_a_config_from_text():
    c = people.load_config(text='[look]\nmin_samples = 7\n[[people]]\nname = "Sam"\n')
    assert c["look"]["min_samples"] == 7 and c["people"][0]["name"] == "Sam"
    req = query.parse("Sam smiling", c, ["Sam"])
    assert req["name"] == "Sam" and req["action"] == "smiling"


# ---- yt-dlp ---------------------------------------------------------------------------------

def test_channel_addresses():
    assert youtube.channel_tabs("https://www.youtube.com/@Show") == ["https://www.youtube.com/@Show/videos"]
    assert youtube.channel_tabs("https://youtube.com/@Show/featured", shorts=True) == [
        "https://youtube.com/@Show/videos", "https://youtube.com/@Show/shorts"]
    assert youtube.channel_tabs("https://www.youtube.com/channel/UC123/") == ["https://www.youtube.com/channel/UC123/videos"]
    for other in ("https://www.youtube.com/playlist?list=PL1", "https://www.youtube.com/watch?v=abc", "https://www.youtube.com/@Show/streams"):
        assert youtube.channel_tabs(other) == [other]
    nested = {"_type": "playlist", "entries": [{"_type": "playlist", "entries": [{"id": "a"}, None]}, {"id": "b"}]}
    assert [e["id"] for e in youtube._entries(nested)] == ["a", "b"]
    assert youtube.entry_date({"timestamp": 1704153600}) == "20240102"
    assert youtube.entry_date({"upload_date": "20230101"}) == "20230101" and youtube.entry_date({}) is None
    assert youtube.in_dates(None, "20240101") and youtube.in_dates("20240105", "20240101", "20240131")
    assert not youtube.in_dates("20231231", "20240101") and not youtube.in_dates("20240201", None, "20240131")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=256x144:d=2:r=25", "-f", "lavfi", "-i",
                    "sine=d=2", "-shortest", str(clip)], check=True)
    videos = [{"id": "newestvid01", "title": 'Tab\there "quoted"', "upload_date": "20240301", "clip": str(clip), "listed_date": True},
              {"id": "livenow0001", "title": "Live", "upload_date": "20240302", "clip": str(clip), "live": "is_live"},
              {"id": "oldvideo001", "title": "Old", "upload_date": "20200101", "clip": str(clip)},
              {"id": "shortvid001", "title": "A Short", "upload_date": "20240301", "clip": str(clip), "tab": "shorts"}]
    (tmp_path / "channel.json").write_text(json.dumps({"videos": videos}))
    monkeypatch.setenv("FAKE_CHANNEL", str(tmp_path / "channel.json"))
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "calls.log"))
    return tmp_path


def test_listing_and_downloading(fake):
    yt = youtube.YtDlp(FAKE, cookies=fake / "cookies.txt", cookies_from_browser="firefox", extra=["--sleep-requests 1"])
    entries = yt.list("https://www.youtube.com/@Fake", shorts=True)
    assert [e["id"] for e in entries] == ["newestvid01", "oldvideo001", "shortvid001"]   # the live stream left out
    assert entries[0]["date"] == "20240301" and entries[1]["date"] is None
    calls = [json.loads(x) for x in (fake / "calls.log").read_text().splitlines()]
    assert calls[0][:6] == ["--cookies", str(fake / "cookies.txt"), "--cookies-from-browser", "firefox", "--sleep-requests", "1"]
    path, info, note = yt.download(entries[0], fake / "dl", sound=True)
    assert path.name == "20240301_newestvid01.mp4" and note == ""
    assert info == {"upload_date": "20240301", "title": 'Tab\there "quoted"', "duration": pytest.approx(2.0, abs=0.1)}
    assert yt.download(entries[0], fake / "dl", sound=True, had_sound=True)[2] == "already downloaded"
    assert yt.download(entries[1], fake / "dl", sound=False, since="20230101") == (
        None, {"upload_date": "20200101"}, "outside the dates asked for")
    (fake / "dl" / "20240301_newestvid01.f251.webm").write_bytes(b"part")
    assert youtube.find_download(fake / "dl", "newestvid01").name == "20240301_newestvid01.mp4"


def test_a_silent_copy_is_downloaded_again_for_its_sound(fake, monkeypatch):
    yt = youtube.YtDlp(FAKE)
    entry = yt.list("https://www.youtube.com/@Fake")[0]
    path, _, _ = yt.download(entry, fake / "dl", sound=False)
    assert not media.has_audio(path)
    path, _, note = yt.download(entry, fake / "dl", sound=True)      # asked for without sound before
    assert note == "" and media.has_audio(path)
    monkeypatch.setenv("FAKE_FAIL", "newestvid01")
    path.unlink()
    path, _, note = yt.download(entry, fake / "dl", sound=True)
    assert path is None and note.startswith("download failed: ERROR: [youtube] newestvid01: Video unavailable")


def test_yt_dlp_problems(fake, monkeypatch):
    with pytest.raises(youtube.YouTubeError, match="could not list"):
        youtube.YtDlp([sys.executable, "-c", "import sys; sys.exit('blocked')"]).list("https://www.youtube.com/@Fake")
    monkeypatch.setattr(youtube.shutil, "which", lambda name: None)
    assert youtube.YtDlp().base == [sys.executable, "-m", "yt_dlp"]
    monkeypatch.setattr(youtube.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(youtube.YouTubeError, match="yt-dlp is not installed"):
        youtube.YtDlp()
    monkeypatch.setattr(youtube.shutil, "which", lambda name: "/usr/bin/yt-dlp")
    assert youtube.YtDlp().base == ["/usr/bin/yt-dlp"]
    # a download that prints nothing useful and leaves no file
    silent = youtube.YtDlp([sys.executable, "-c", "print('NA\\tNA\\tNA\\tNA'); print('not json\\tx\\ty\\tz')"])
    assert silent.download({"id": "abcdefghijk", "url": "u"}, fake / "none", sound=False) == (None, {}, "download failed: not json\tx\ty\tz")


# ---- the channel report -------------------------------------------------------------------------

def test_the_report_of_a_channel_search(tmp_path):
    store, prefix = FileStore(tmp_path), "searches/s"
    row = {"video": "20240301_abcdefghijk.mp4", "person": 0, "start_s": 3.2, "end_s": 3.4, "start": "0:00:03", "end": "0:00:03",
           "duration_s": 0.2, "action": "jumping", "match": "possible", "evidence": "e", "thumb": "t.jpg", "in_intro": False,
           "youtube": "https://www.youtube.com/watch?v=abcdefghijk&t=3s"}
    store.put_json(f"{prefix}/done/abcdefghijk.json", {"video": row["video"], "video_id": "abcdefghijk", "title": "T",
                                                       "upload_date": "2024-03-01", "duration_s": 7200, "note": "", "rows":
                                                       [row, dict(row, start_s=60, end="0:01:05", start="0:01:00", in_intro=True)]})
    store.put_json(f"{prefix}/done/zzzzzzzzzzz.json", {"video": "other.mp4", "video_id": "zzzzzzzzzzz", "title": "",
                                                       "upload_date": "", "duration_s": 0, "note": "nobody jumping", "rows": []})
    store.put(f"{prefix}/done/broken.json", b"{half")               # another run writing: skipped
    req = query.parse("people jumping", people.load_config())
    channel.write_report(store, prefix, "people jumping", req, "https://www.youtube.com/@X", 3, 0, ["a problem"])
    md = store.get(f"{prefix}/report.md").decode()
    assert "1 result in 1 video (and 1 in the intros, listed at the end); 2 of 3 videos searched (2.0 hours of video)" in md
    assert "- [0:00:03](https://www.youtube.com/watch?v=abcdefghijk&t=3s) jumping (possible): e" in md
    assert md.index("0:01:00-0:01:05") > md.index("# In the intros")
    page = store.get(f"{prefix}/report.html").decode()
    assert "nobody jumping" in page and "a problem" in page and 'class="tag">possible' in page
    assert len(h.csv_rows(tmp_path / prefix / "matches.csv")) == 2
    assert channel._eta(0, 0, 10, []) == "" and "about" in channel._eta(3600, 1, 60, [{"duration": 3600}, {}])
    assert channel._took(30) == "30s" and channel._took(600) == "10.0 min" and channel._took(7200) == "2.0 h"
    with pytest.raises(ValueError):
        channel.ChannelQuery("x", speed="warp")
    with pytest.raises(ValueError):
        channel.ChannelQuery("x", intros="maybe")
    with pytest.raises(ValueError, match="nothing to search for"):
        channel.ChannelQuery("wibble").request()


def test_where_the_interview_starts_without_a_shared_title(monkeypatch):
    """Episodes with a title of their own: the start is the first run of camera angles seen again
    later, or else footage shared with one other episode early on."""
    rng = np.random.default_rng(1)
    opening = rng.integers(0, 255, (intro.OPENING_S, intro.H, intro.W)).astype(np.uint8)
    bank = {"thumbs": opening[5:9].copy(), "count": np.ones(4, np.int32), "second": np.arange(5, 9, dtype=np.int32)}
    monkeypatch.setattr(intro, "body", lambda path: opening[40:60].copy())     # angles from 40 s recur later
    r = intro.interview_start("v.mp4", bank, first=opening)
    assert (r["how"], r["interview_start_s"]) == ("angles", 40)
    monkeypatch.setattr(intro, "body", lambda path: np.zeros((0, intro.H, intro.W), np.uint8))
    r = intro.interview_start("v.mp4", bank, first=opening)
    assert (r["how"], r["interview_start_s"]) == ("shared", 9)               # after the shared seconds 5-8
    r = intro.interview_start("v.mp4", {"thumbs": bank["thumbs"][:0], "count": bank["count"][:0],
                                         "second": bank["second"][:0]}, first=opening)
    assert r["how"] == "none"
    assert intro.interview_start("v.mp4", bank, first=opening[:100])["how"] == "short video"
    assert intro.interview_start("v.mp4", None)["how"] == "none"


def test_a_long_section_is_checked_for_stillness_a_piece_at_a_time(tmp_path):
    """Sitting can last most of an episode: a section longer than STILL_CHUNK_S is read in pieces,
    and only until one shows the figure moving."""
    from youtube_analyser import detector
    v = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=256x144:d=20:r=25", str(v)], check=True)
    box = [[60.0, 20.0, 120.0, 100.0]]
    long = {"boxes": [(float(t), box) for t in range(0, 700)]}
    assert detector.still_diff(v, long, 1.0, 256, 144) >= detector.STILL_DIFF     # the test pattern's counter moves
    still = {"boxes": [(float(t), box) for t in range(400, 1000)]}                  # past the end: nothing to compare
    assert detector.still_diff(v, still, 1.0, 256, 144) is None


def test_videos_held_back_for_the_intro_bank_are_searched_when_the_last_video_fails(fake):
    """The channel's videos wait for the intro bank until enough long ones are in; when the last
    one to arrive is too old (or fails), the ones waiting are still searched."""
    cq = channel.ChannelQuery("people jumping")                         # intros labelled: a bank is wanted
    done, _ = channel.run_channel("https://www.youtube.com/@Fake", cq, fake / "work", youtube.YtDlp(FAKE),
                                  since="20230101", downloads=1, log=h.Captured())
    assert list(done) == ["newestvid01"]                                # searched, though it was held back
    assert done["newestvid01"]["note"] in ("nobody in view", "nobody jumping")


# ---- dates, names and odd files (found before publishing) ---------------------------------------

def test_dates_hold_for_videos_downloaded_before(fake):
    """--since and --until apply to videos already downloaded by an earlier search (whose
    listing gave no dates), and a search run again with dates reports only those videos."""
    yt, url = youtube.YtDlp(FAKE), "https://www.youtube.com/@Fake"
    quiet = h.Captured()
    channel.run_channel(url, channel.ChannelQuery("people standing", intros="keep"), fake / "w", yt, log=quiet)
    assert sorted(p.name for p in (fake / "w" / "videos").iterdir()) == ["20200101_oldvideo001.mp4", "20240301_newestvid01.mp4"]
    done, _ = channel.run_channel(url, channel.ChannelQuery("people jumping", intros="keep"), fake / "w", yt,
                                  since="20230101", log=quiet)
    assert list(done) == ["newestvid01"]                                  # the old copy is not searched
    done, folder = channel.run_channel(url, channel.ChannelQuery("people standing", intros="keep"), fake / "w", yt,
                                       since="2023-01-01", log=quiet)
    assert list(done) == ["newestvid01"]                                  # nor reported from the earlier run
    assert "1 of 1 video searched" in (folder / "report.md").read_text()
    done, _ = channel.run_channel(url, channel.ChannelQuery("people standing", intros="keep"), fake / "w", yt,
                                  until="20211231", log=quiet)
    assert list(done) == ["oldvideo001"]
    # yt-dlp's side: a copy already there, outside the dates asked for
    entry = {"id": "oldvideo001", "url": "https://www.youtube.com/watch?v=oldvideo001"}
    assert yt.download(entry, fake / "w" / "videos", sound=False, since="20230101") == (
        None, {"upload_date": "20200101"}, youtube.OUTSIDE_DATES)
    assert yt.download(entry, fake / "w" / "videos", sound=False, until="20211231")[2] == "already downloaded"


def test_a_date_range_with_both_ends(fake):
    """--since with --until on a channel listed without dates, nothing downloaded yet. A channel
    is listed newest first, so the videos newer than --until come first: they do not count
    towards "5 in a row older than --since", and the videos between the dates are reached. The
    run still stops once five in a row are older than --since."""
    clip = str(fake / "clip.mp4")
    videos = ([{"id": f"newer{k:06d}", "title": f"Newer {k}", "upload_date": f"2025{k:02d}01", "clip": clip} for k in range(6, 0, -1)]
              + [{"id": "september01", "title": "September", "upload_date": "20240915", "clip": clip},
                 {"id": "march000001", "title": "March", "upload_date": "20240312", "clip": clip}]
              + [{"id": f"older{k:06d}", "title": f"Older {k}", "upload_date": f"2023{k:02d}01", "clip": clip} for k in range(8, 0, -1)])
    (fake / "channel.json").write_text(json.dumps({"videos": videos}))
    said = h.Captured()
    done, folder = channel.run_channel("https://www.youtube.com/@Fake", channel.ChannelQuery("people standing", intros="keep"),
                                       fake / "w", youtube.YtDlp(FAKE), since="20240101", until="2024-12-31", downloads=1, ahead=1,
                                       log=said)
    assert sorted(done) == ["march000001", "september01"]                 # the six newer videos did not end the run
    assert "2 of 16 videos searched" in (folder / "report.md").read_text()
    assert "download failed" not in said.stdout                           # left out for their dates, not failures
    assert "5 videos in a row older than --since" in said.stdout          # the older videos still end it
    asked = [json.loads(x)[-1][-11:] for x in (fake / "calls.log").read_text().splitlines()[1:]]   # after the listing
    assert asked[:13] == [v["id"] for v in videos[:13]]                   # six newer, two in the dates, five older
    assert not {"older000002", "older000001"} & set(asked)                # the last ones were never asked for


def test_a_video_left_out_for_its_date(fake):
    """yt-dlp says nothing about a video it leaves out for its date when --print is used (and has
    one message for too new and too old when it does speak): the date it prints first tells such
    a video from a failed download, and which side of the dates it is on."""
    yt = youtube.YtDlp(FAKE)
    newest, old = yt.list("https://www.youtube.com/@Fake")
    out = youtube.OUTSIDE_DATES
    assert yt.download(old, fake / "dl", sound=False, since="20230101", until="20241231") == (None, {"upload_date": "20200101"}, out)
    assert yt.download(newest, fake / "dl", sound=False, since="20210101", until="20231231") == (None, {"upload_date": "20240301"}, out)
    # yt-dlp's own message, for someone who runs it with --no-quiet, and on its own
    loud = youtube.YtDlp(FAKE, extra=["--no-quiet"])
    assert loud.download(newest, fake / "dl", sound=False, until="20231231") == (None, {"upload_date": "20240301"}, out)
    entry = {"id": "abcdefghijk", "url": "u"}
    printing = lambda text: youtube.YtDlp([sys.executable, "-c", f"print({text!r})"])
    message = "[download] 2025-03-01 upload date is not in range 2024-01-01 to 2024-12-31"
    assert printing(message).download(entry, fake / "none", sound=False, since="20240101", until="20241231") == (
        None, {"upload_date": "20250301"}, out)
    assert printing("upload date is not in range").download(entry, fake / "none", sound=False, until="20241231") == (None, {}, out)
    # a download that fails is still a failure, and the date printed first is not taken for the reason
    for first in ('upload_date\t"20240601"', "upload_date\tNA", 'upload_date\t"soon"'):
        assert printing(first).download(entry, fake / "none", sound=False, since="20240101") == (None, {}, "download failed: no file")
    # which side: the date when it is known; without one, older only when there is no --until
    assert youtube.older_than_since("20231231", "20240101", "20241231")
    assert not youtube.older_than_since("20250101", "20240101", "20241231")
    assert youtube.older_than_since(None, "20240101") and not youtube.older_than_since(None, "20240101", "20241231")
    assert not youtube.older_than_since("20200101", None, "20241231")


def test_dates_and_addresses():
    assert youtube.parse_date("2024-03-01") == "20240301" == youtube.parse_date("20240301")
    assert youtube.parse_date(None) is None and youtube.parse_date("") is None
    for bad in ("2024-13-01", "yesterday", "240301"):
        with pytest.raises(ValueError, match="not a date"):
            youtube.parse_date(bad)
    assert youtube.in_dates("2024-03-01", "20240301", "20240301") and not youtube.in_dates("2024-02-29", "20240301")
    assert youtube.entry_url({"_type": "url", "id": "a", "url": "https://www.youtube.com/shorts/a"}) == "https://www.youtube.com/shorts/a"
    assert youtube.entry_url({"id": "a", "url": "https://rr1.googlevideo.com/videoplayback?x",      # a single video's media file
                              "webpage_url": "https://www.youtube.com/watch?v=a"}) == "https://www.youtube.com/watch?v=a"
    assert youtube.entry_url({"id": "abcdefghijk"}) == "https://www.youtube.com/watch?v=abcdefghijk"
    assert media.youtube_id("20240101_family trip") is None              # not an 11-character YouTube ID
    for option in (["--since", "2024-13-01"], ["--shard", "3/2"], ["--max-videos", "0"]):
        with pytest.raises(SystemExit):
            h.yta("channel", "x", "y", *option)
    for option in (["--conf", "1.5"], ["--imgsz", "16"], ["--fps", "0"], ["--threads", "-1"], ["--max-gap", "x"]):
        with pytest.raises(SystemExit):
            h.yta("detect", "a.mp4", *option)
    with pytest.raises(SystemExit):
        h.yta("review", "--width", "641")


def test_awkward_file_names_and_cover_art(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=256x144:d=2:r=25", str(src)], check=True)
    for name in ("Interview: part 1.mp4", "-starts-with-a-dash.mp4", "crypto:x.mp4"):
        (tmp_path / name).write_bytes(src.read_bytes())
    import os
    here = os.getcwd()
    os.chdir(tmp_path)                          # relative names, as a user would type them
    try:
        for name in ("Interview: part 1.mp4", "-starts-with-a-dash.mp4", "crypto:x.mp4"):
            assert media.probe(name) == (256, 144), name
            assert media.duration(name) == pytest.approx(2.0, abs=0.1), name
            assert len(list(media.read_frames(name, 1.0, 256, 144))) == 2, name
            assert len(list(media.keyframes(name, 256, 144))) >= 1, name
            assert media.grab_frame(name, 1.0, 1.0, 256, 144).std() > 30, name
    finally:
        os.chdir(here)
    # a cover picture stored as the first video stream: the real video is read, not the picture
    cover = tmp_path / "cover.png"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=640x360", "-frames:v", "1", str(cover)], check=True)
    both = tmp_path / "with_cover.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(cover), "-i", str(src), "-map", "0", "-map", "1", "-c", "copy",
                    str(both)], check=True)
    assert media.video_stream(both) == (1, 256, 144)
    frames = [img for _, img in media.read_frames(both, 1.0, 256, 144)]
    assert len(frames) == 2 and frames[0].std() > 30                      # the test pattern, not a red picture
    picture_only = tmp_path / "picture.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(cover), "-c", "copy", str(picture_only)], check=True)
    assert media.video_stream(picture_only)[1:] == (640, 360)             # a picture alone is still read
    assert len(intro.opening(tmp_path / "notes.txt")) == 0                # not a video: no thumbnails, no crash


def test_one_video_file_as_the_source(tmp_path):
    folder = tmp_path / "v"
    folder.mkdir()
    for date in ("20240101", "20240202"):
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=256x144:d=2:r=25",
                        str(folder / f"{date}_abcdefghij{date[-1]}.mp4")], check=True)
    done, _ = channel.run_channel(folder / "20240202_abcdefghij2.mp4", channel.ChannelQuery("people jumping", intros="keep"),
                                  tmp_path / "w", log=h.Captured())
    assert list(done) == ["abcdefghij2"]


def test_an_inventory_someone_has_edited(tmp_path):
    inv = tmp_path / "videos.csv"
    inv.write_text("file,upload_date,video_id,title,kind,duration_min,my notes\n"
                   "a.mp4,2024-01-01,aaaaaaaaaaa,A,long,60,keep this\n"
                   ",,,,,,\n"                                              # an empty row
                   "b.mp4,2024-01-02,bbbbbbbbbbb,B,long,61,x,an extra value with no column\n")
    i = channel.Inventory(inv)
    assert sorted(i.rows) == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    i.add("ccccccccccc", tmp_path / "20240103_ccccccccccc.mp4", {"title": "C", "duration": 30.0}, {}, sound=True)
    rows = h.csv_rows(inv)
    assert [r["video_id"] for r in rows] == ["ccccccccccc", "bbbbbbbbbbb", "aaaaaaaaaaa"]
    assert rows[2]["my notes"] == "keep this" and rows[0]["upload_date"] == "2024-01-03" and rows[0]["asked_for_sound"] == "yes"


def test_a_users_colours_and_review_files(tmp_path):
    c = people.load_config(text='[colours.hair]\nteal = "#2a7f7f"\n')     # one example without brackets
    assert "teal" in c["colours"]["hair"]
    with pytest.raises(ValueError, match="#rrggbb"):
        people.load_config(text='[colours.hair]\nteal = ["green-ish"]\n')
    from youtube_analyser import report
    verdicts = tmp_path / "v.csv"
    verdicts.write_text("\ufeffvideo,start_s,verdict\nx.mp4,1.0, confirmed\n")   # a spreadsheet's BOM, a stray space
    assert report.load(verdicts, ("video", "start_s", "verdict"))[0]["video"] == "x.mp4"
    verdicts.write_text("video,start,verdict\nx.mp4,1.0,confirmed\n")
    with pytest.raises(ValueError, match="no start_s column"):
        report.load(verdicts, ("video", "start_s", "verdict"))

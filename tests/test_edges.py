"""Edge cases with the models: videos with nobody in them, portrait videos, saved model output
from older versions or for a replaced video, models shared between threads, enrolment problems,
searches that cannot hear or see enough, and channel options on a folder of videos.

    pytest tests/test_edges.py
"""
import subprocess
import threading

import numpy as np
import pytest

from tests import helpers as h
from tests.helpers import mf
from youtube_analyser import analysis, channel, faces, inference, people, query, search
from youtube_analyser.analysis import Settings, VideoAnalysis


@pytest.fixture(scope="module")
def empty_video(tmp_path_factory):
    v = tmp_path_factory.mktemp("empty") / "20240101_nobodyhere1.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=256x144:d=6:r=25", "-g", "25", str(v)], check=True)
    return v


def test_nobody_in_the_video(empty_video, tmp_path):
    r = h.yta("detect", empty_video, "--out", tmp_path, "--keep-intro", "--action", "jumping")
    assert r.code == 0 and "0 jumping sections" in r.stdout
    config = people.load_config()
    s = search.Searcher(config, h.SETTINGS, tmp_path)
    assert s.search(empty_video, query.parse("people jumping", config)) == []
    assert s.note in ("nobody jumping", "nobody in view")
    assert s.search(empty_video, query.parse("woman with purple hair", config)) == []
    assert s.note == "nobody in view"
    assert s.scout(empty_video, query.parse("woman with purple hair", config)) is False
    assert s.note.startswith("quick look: nobody in")
    with pytest.raises(ValueError, match="needs a person or an action"):
        s.anyone(s.analysis(mf.build("walk_stop_walk")), None)


def test_a_portrait_video(tmp_path):
    """Shorts are upright: the pose model gets a taller input, and everything else follows."""
    v = tmp_path / "portrait.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(mf.build("walk_stop_walk")), "-vf", "crop=81:144:88:0,scale=144:256",
                    "-c:v", "libx264", str(v)], check=True)
    r = h.yta("detect", v, "--out", tmp_path / "out", "--keep-intro")
    assert r.code == 0, r.stderr
    va = VideoAnalysis(v, h.SETTINGS, cache=tmp_path / "out")
    assert tuple(va.poses["size"]) == (144, 256) and len(va.poses["frame"]) > 0


def test_saved_output_of_an_older_version_is_used_and_a_replaced_video_is_not(tmp_path):
    clip = tmp_path / "20240101_walkstopwal.mp4"
    clip.write_bytes(mf.build("walk_stop_walk").read_bytes())
    va = VideoAnalysis(clip, Settings(), cache=tmp_path / "out")
    poses = va.poses
    saved = tmp_path / "out" / "poses" / f"{clip.stem}.npz"
    # an older version's file: the same arrays, without the record of what made them
    legacy = {k: v for k, v in np.load(saved).items() if k != "_key"}
    np.savez_compressed(saved, **legacy)
    again = VideoAnalysis(clip, Settings(), cache=tmp_path / "out")
    assert again.poses is not None and again.pose_note == "saved poses"
    other = VideoAnalysis(clip, Settings(imgsz=320), cache=tmp_path / "out")
    assert other.poses is not None and other.pose_note.endswith("ms/frame")   # another size: made again
    # the video replaced by another under the same name: everything is made again
    clip.write_bytes(mf.build("two_jumpers").read_bytes())
    fresh = VideoAnalysis(clip, Settings(imgsz=320), cache=tmp_path / "out")
    assert len(fresh.poses["t"]) == 10 != len(poses["t"]) and fresh.pose_note.endswith("ms/frame")


def test_models_shared_between_threads_give_the_same_answers():
    models = Settings().models()
    poses = VideoAnalysis(mf.build("three_talkers"), h.SETTINGS, cache=h.OUT).poses
    frames = [img for _, img in h.media.read_frames(mf.path("three_talkers"), 1.0, 256, 144)][:6]
    geos = [(f, faces.face_geometry(poses["kp"][j])) for j, f in enumerate(poses["frame"]) if f < 6]
    geos = [(f, g) for f, g in geos if g is not None]
    alone = [models.faces.face(frames[f], g)[1:] for f, g in geos]
    together, errors = [None] * len(geos), []

    def work(i):
        try:
            f, g = geos[i]
            together[i] = models.faces.face(frames[f], g)[1:]
        except Exception as e:  # pragma: no cover - the failure the test is for
            errors.append(e)
    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(geos))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and np.allclose(together, alone)
    assert inference.Models.shared() is inference.Models.shared()
    with pytest.raises(ValueError):
        inference.runtime_name("cuda")
    single = inference.PoseModel(analysis.Settings().model_path, 640, runtime="onnxruntime")
    assert len(single(frames[0], 0.35)[0]) == len(list(single.run([frames[0]], 0.35))[0][0])


def test_enrolment_problems(tmp_path):
    quiet = tmp_path / "quiet.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "3", str(quiet)], check=True)
    config = people.load_config(text=f'''
[[people]]
name = "Nobody"
voice_file = "{tmp_path / 'missing.wav'}"
[[people]]
name = "Quiet"
voice_file = "{quiet}"
[[people]]
name = "Jane"
voice_from = {{ video = "duo_purple.mp4", start = 0.0, end = 8.0 }}
[[people]]
name = "Undescribed"
''')
    said = h.Captured()
    s = search.Searcher(config, h.SETTINGS, h.OUT, videos=[mf.build("duo_purple")], log=said)
    assert list(s.enrol()) == ["Jane"]                     # found among the videos by its name
    assert "Nobody: no sound" in said.stderr and "Quiet: too little speech" in said.stderr
    assert s.enrol() is s.enrolled                          # once


def test_searches_that_cannot_hear_or_see_enough():
    config = people.load_config(text="[models]\ngender_age = false\n")
    s = search.Searcher(config, h.SETTINGS, h.OUT, possible=True)
    # no sound track: nobody's gender can come from a voice, and the face model is switched off
    rows = s.search(mf.build("three_talkers_no_sound"), query.parse("man talking", config))
    assert rows and all(r["match"] == "possible" for r in rows)
    assert "gender not seen" in rows[0]["evidence"]
    # a name nobody has enrolled is nobody
    assert s.search(mf.build("duo_purple"), query.parse("Jane talking", config, ["Jane"])) == []


def test_a_channel_folder_with_dates_shards_and_a_changed_query(tmp_path):
    folder = tmp_path / "videos"
    folder.mkdir()
    for date, name in (("20240301", "duo_purple"), ("20240101", "jump_three")):
        (folder / f"{date}_{name[:11].ljust(11, 'x')}.mp4").symlink_to(mf.build(name))
    (folder / "notes.txt").write_text("not a video")
    cq = channel.ChannelQuery("people jumping", intros="keep")
    done, where = channel.run_channel(folder, cq, tmp_path / "w", since="20240201", log=h.Captured())
    assert list(done) == ["duo_purplex"]                     # the January video is older than --since
    done, _ = channel.run_channel(folder, cq, tmp_path / "w", shard="2/2", delete_videos=True, log=h.Captured())
    assert set(done) == {"duo_purplex", "jump_threex"}      # the second of two shards: the other video
    assert len(list(folder.glob("*.mp4"))) == 2              # a folder's videos are never deleted
    said = h.Captured()
    channel.run_channel(folder, channel.ChannelQuery("people jumping", intros="keep", possible=True), tmp_path / "w", log=said)
    assert "starting it again" in said.stdout
    with pytest.raises(ValueError):
        channel.run_channel(folder, cq, tmp_path / "w", shard="3/2", log=h.Captured())

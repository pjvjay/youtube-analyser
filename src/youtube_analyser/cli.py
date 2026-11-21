"""The `yta` command.

    yta detect *.mp4 --action talking          sections where someone does an action
    yta search "woman with purple hair" *.mp4   a described person (doing an action)
    yta channel <channel URL or folder> "people jumping"
                                               a whole channel: download, search, report with links
    yta intro --build-bank *.mp4               the bank of title frames that marks the intros
    yta report --action talking                the results files from a detector run
    yta review --action talking                one video of every section, to review them
    yta serve                                  the HTTP API
    yta worker --store ... --queue ...         a worker for the API or for channel searches
    yta collect <search id> --store ...        the report of a channel search run by workers
    yta models                                 the models and the runtime in use
    yta config > my_people.toml                the search's words, colours and people, to change

`yta <command> --help` lists each command's options.
"""
import argparse
import sys
from pathlib import Path

from . import __version__
from .actions import ACTIONS
from .analysis import Settings
from .motion import JUMP_FPS


def _model_options(ap, fps=True):
    ap.add_argument("--model", default=Settings().pose_model,
                    help="pose model: a file name in the models folder, or a path (default %(default)s)")
    ap.add_argument("--imgsz", type=int, default=640, help="longest side fed to the pose model (default 640)")
    if fps:
        ap.add_argument("--fps", type=float, default=1.0, help="frames sampled per second of video (default 1)")
    ap.add_argument("--jump-fps", type=float, default=JUMP_FPS,
                    help="running and jumping: frames per second sampled where a whole body is in view (default 5)")
    ap.add_argument("--conf", type=float, default=0.35, help="minimum person confidence (default 0.35)")
    ap.add_argument("--runtime", choices=["auto", "openvino", "onnxruntime"], default="auto",
                    help="auto = OpenVINO if installed, else onnxruntime")
    ap.add_argument("--threads", type=int, default=0, help="inference threads (0 = automatic)")


def _settings(a):
    return Settings(pose_model=a.model, imgsz=a.imgsz, fps=getattr(a, "fps", 1.0), conf=a.conf, jump_fps=a.jump_fps,
                    runtime=a.runtime, threads=a.threads)


def _bank(a, log):
    from . import intro
    if a.keep_intro:
        return None
    bank = intro.load_bank(a.intro_bank)
    if bank is None:
        log(f"no intro bank at {a.intro_bank}: the intro is not separated (build one with yta intro --build-bank)",
            file=sys.stderr)
    return bank


def _intro_options(ap):
    from .intro import DEFAULT_BANK
    ap.add_argument("--intro-bank", type=Path, default=DEFAULT_BANK, help="title frames that mark the intro (yta intro)")
    ap.add_argument("--keep-intro", action="store_true", help="treat the whole video as the interview")


def _yt_dlp_options(ap):
    ap.add_argument("--cookies", type=Path, help="cookies.txt for yt-dlp (if YouTube asks you to sign in)")
    ap.add_argument("--cookies-from-browser", help="e.g. firefox or chrome: yt-dlp reads the browser's cookies")
    ap.add_argument("--yt-dlp", help="the yt-dlp command (default: yt-dlp on the PATH, else python -m yt_dlp)")
    ap.add_argument("--yt-dlp-arg", action="append", help="more yt-dlp options, e.g. '--sleep-requests 1'")


def _yt(a):
    from .youtube import YtDlp
    return YtDlp(a.yt_dlp, a.cookies, a.cookies_from_browser, a.yt_dlp_arg)


def build_parser():
    ap = argparse.ArgumentParser(prog="yta", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("detect", help="sections where someone does an action",
                       description="Find the sections of each video where someone does an action (default standing).")
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--action", choices=list(ACTIONS), default="standing")
    p.add_argument("--out", type=Path, default=Path("output"),
                   help="output folder; actions other than standing write to a subfolder named after the action, "
                        "and all share its model output (default ./output)")
    _model_options(p)
    p.add_argument("--min-frames", type=int, help="frames showing the action needed for a section "
                   "(default 2; 1 for jumping, 3 for running)")
    p.add_argument("--max-gap", type=int, help="frames without it allowed inside a section "
                   "(default 2; 1 for walking and silent; 4 for jumping, whose frames are 0.2 s apart)")
    p.add_argument("--motion", choices=["any", "still", "moving"],
                   help="standing and sitting: only people standing or sitting still (or moving)")
    p.add_argument("--no-sound", action="store_true", help="talking and silent: judge from the faces alone")
    p.add_argument("--reclassify", action="store_true", help="redo sections for videos already done, from the saved output")
    p.add_argument("--force", action="store_true", help="redo everything, the pose model included")
    _intro_options(p)

    p = sub.add_parser("search", help="a described person doing an action",
                       description="Search videos for a described person doing an action (see docs/search.md).")
    p.add_argument("query", help='who to look for, e.g. "woman with purple hair talking" (or "people jumping")')
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--action", choices=list(ACTIONS) + ["visible"], help="what they should be doing (default: from the query)")
    p.add_argument("--out", type=Path, default=Path("output"), help="the model output folder (shared with yta detect)")
    p.add_argument("--results", type=Path, default=Path("results") / "search")
    p.add_argument("--config", type=Path, help="your people.toml (merged over the defaults)")
    p.add_argument("--possible", action="store_true", help="also list people who might match (some part not seen)")
    p.add_argument("--explain", action="store_true", help="print what was seen of every person and why they matched or not")
    p.add_argument("--quick", action="store_true", help="with a colour in the query, look at each video's keyframes first "
                   "and skip the video if nobody there has it (can miss someone on screen for under about 10 s)")
    _model_options(p)
    _intro_options(p)

    p = sub.add_parser("channel", help="a whole channel: download, search, report with links",
                       description="Search a whole YouTube channel (or a folder of videos) and report every moment "
                                   "found, with a link to it (see docs/channel.md).")
    p.add_argument("source", help="a YouTube channel, playlist or video address, or a folder of downloaded videos")
    p.add_argument("query", help='what to look for, e.g. "people jumping" or "woman with purple hair talking"')
    p.add_argument("--action", choices=list(ACTIONS) + ["visible"], help="instead of an action in the query")
    p.add_argument("--work", type=Path, help="where videos, model output and reports go (default ./channels/<channel>)")
    p.add_argument("--speed", choices=["quick", "thorough", "fastest"], default="quick",
                   help="quick (default): a quick look at keyframes first when the query has a colour; "
                        "thorough: every second of every video; fastest: quick, and a smaller pose model input")
    p.add_argument("--max-videos", type=int, help="only the newest N videos")
    p.add_argument("--since", help="only videos uploaded on or after this date (YYYYMMDD)")
    p.add_argument("--until", help="only videos uploaded on or before this date (YYYYMMDD)")
    p.add_argument("--shorts", action="store_true", help="include the channel's Shorts")
    p.add_argument("--downloads", type=int, default=3, help="videos downloaded at once (default 3)")
    p.add_argument("--ahead", type=int, default=6, help="at most this many downloaded videos waiting to be searched")
    p.add_argument("--delete-videos", action="store_true", help="delete each video once it has been searched")
    p.add_argument("--shard", help="K/N: search only every Nth video, starting with the Kth (N runs share a channel)")
    p.add_argument("--intros", choices=["label", "keep"], default="label",
                   help="label (default): set results in each episode's intro apart; keep: each video is one whole")
    p.add_argument("--config", type=Path, help="your people.toml (merged over the defaults)")
    p.add_argument("--possible", action="store_true", help="also list people who might match")
    p.add_argument("--explain", action="store_true", help="print what was seen of every person")
    _yt_dlp_options(p)
    p.add_argument("--store", help="with --queue: run the search on cloud workers, keeping results in this store "
                                   "(e.g. azblob://account/container); see docs/cloud.md")
    p.add_argument("--queue", help="with --store: the queue the workers take work from (e.g. azqueue://account/queue)")
    p.add_argument("--runtime", choices=["auto", "openvino", "onnxruntime"], default="auto")
    p.add_argument("--threads", type=int, default=0)

    p = sub.add_parser("collect", help="the report of a channel search run by cloud workers")
    p.add_argument("search", help="the search id yta channel --store --queue printed")
    p.add_argument("--store", required=True, help="the store the workers write to")
    p.add_argument("--out", type=Path, help="also download the report and thumbnails into this folder")

    p = sub.add_parser("intro", help="the bank of title frames, and where interviews start",
                       description="Build the bank of title frames from the videos (--build-bank), or print where "
                                   "each video's interview starts.")
    p.add_argument("videos", nargs="+", type=Path)
    p.add_argument("--bank", type=Path, default=None, help="the bank file (default ./intro_bank.npz)")
    p.add_argument("--build-bank", action="store_true", help="build the bank from these videos")

    p = sub.add_parser("report", help="the results files from a detector run")
    p.add_argument("--action", choices=list(ACTIONS), default="standing")
    p.add_argument("--detector-output", type=Path, default=Path("output"), help="yta detect's --out (default ./output)")
    p.add_argument("--results", type=Path, default=Path("results"), help="where to write (default ./results)")
    p.add_argument("--inventory", type=Path, default=Path("analysis") / "videos.csv",
                   help="the video list with titles (default ./analysis/videos.csv)")

    p = sub.add_parser("review", help="one video of every section, to review them")
    p.add_argument("--action", choices=list(ACTIONS), default="standing")
    p.add_argument("--sections", type=Path, help="default results/standing_sections.csv or results/<action>/<action>_sections.csv")
    p.add_argument("--videos-dir", type=Path, default=Path("."), help="folder holding the videos (default .)")
    p.add_argument("--verdicts", help='comma-separated verdicts, or "all" (default "confirmed,unclear", plus '
                   '"not reviewed" for actions other than standing)')
    p.add_argument("--pad", type=float, default=2.0, help="seconds of context before and after each section")
    p.add_argument("--out", type=Path, help="default output/review.mp4 or output/<action>/review.mp4")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)

    p = sub.add_parser("serve", help="the HTTP API (settings from the environment, see docs/api.md)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    p = sub.add_parser("worker", help="take work from a queue: uploads to the API, channel searches")
    p.add_argument("--store", required=True, help="where videos and results are (a folder, or e.g. azblob://account/container)")
    p.add_argument("--queue", required=True, help="where the work is (e.g. file:///path or azqueue://account/queue)")
    p.add_argument("--until-empty", action="store_true", help="stop when the queue is empty (for a cloud job)")
    p.add_argument("--config", type=Path, help="your people.toml")
    _model_options(p)
    p.add_argument("--intro-bank", type=Path, help="title frames that mark the intro of uploaded videos")
    _yt_dlp_options(p)

    sub.add_parser("models", help="the models and the runtime in use")
    sub.add_parser("config", help="print the default people.toml (the words, colours and thresholds of searches)")
    return ap


def main(argv=None, log=print):
    a = build_parser().parse_args(argv)
    try:
        return _run(a, log) or 0
    except ImportError as e:   # an optional part that is not installed
        top = (e.name or "").split(".")[0]
        extra = {"fastapi": "service", "uvicorn": "service", "starlette": "service", "azure": "azure"}.get(top)
        log(f"yta {a.command}: {e.name or e} is not installed" + (f': pip install "youtube-analyser[{extra}]"' if extra else ""),
            file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        log(f"yta {a.command}: {e}", file=sys.stderr)
        return 2


def _run(a, log):
    from . import people
    if a.command == "detect":
        from .detector import detect_videos
        detect_videos(a.videos, a.action, a.out, _settings(a), a.min_frames, a.max_gap, a.motion, not a.no_sound,
                      a.reclassify, a.force, _bank(a, log), log=log)
    elif a.command == "search":
        from .search import search_videos
        search_videos(a.query, a.videos, people.load_config(a.config), a.action, _settings(a), a.out, a.results,
                      _bank(a, log), a.possible, a.explain, a.quick, log=log)
    elif a.command == "channel":
        return _channel(a, log)
    elif a.command == "collect":
        from . import cloud
        from .channel import collect_channel
        collect_channel(cloud.open_store(a.store), a.search, a.out, log=log)
    elif a.command == "intro":
        return _intro(a, log)
    elif a.command == "report":
        from .report import make_report
        make_report(a.action, a.detector_output, a.results, a.inventory, log=log)
    elif a.command == "review":
        from .review import make_review
        make_review(a.action, a.sections, a.videos_dir, a.verdicts, a.pad, a.out, a.width, a.height, log=log)
    elif a.command == "serve":  # pragma: no cover - runs until stopped; the app is tested directly
        import uvicorn
        uvicorn.run("youtube_analyser.service:app", factory=True, host=a.host, port=a.port)
    elif a.command == "worker":
        from . import cloud, intro
        from .jobs import Worker
        bank = intro.load_bank(a.intro_bank) if a.intro_bank else None
        worker = Worker(cloud.open_store(a.store), cloud.open_queue(a.queue), _settings(a), people.load_config(a.config),
                        bank, log=log, yt=_yt(a))
        log(f"worker: taking work from {a.queue}", flush=True)
        worker.run(until_empty=a.until_empty)
    elif a.command == "config":
        from . import assets
        log(assets.PEOPLE_TOML.read_text(), end="")
    elif a.command == "models":
        from . import assets, inference
        log(f"runtime: {inference.runtime_name('auto')}; models in {assets.models_dir()}")
        for p in sorted(assets.models_dir().glob("*.onnx")):
            log(f"  {p.name}  {p.stat().st_size / 1e6:.1f} MB")


def _channel(a, log):
    from . import cloud
    from .channel import ChannelQuery, enqueue_channel, run_channel
    config_text = a.config.read_text() if a.config else ""
    cq = ChannelQuery(a.query, a.action, a.speed, a.possible, a.intros, config_text)
    yt = _yt(a)
    if a.store or a.queue:
        if not (a.store and a.queue):
            raise ValueError("--store and --queue go together (a cloud search); leave both out to search here")
        search_id = enqueue_channel(a.source, cq, cloud.open_store(a.store), cloud.open_queue(a.queue), yt, a.max_videos,
                                    a.since, a.until, a.shorts, log=log)
        log(f"when the workers are done: yta collect {search_id} --store {a.store} --out <folder>")
        return 0
    run_channel(a.source, cq, a.work, yt, a.max_videos, a.since, a.until, a.shorts, a.downloads, a.ahead,
                a.delete_videos, a.shard, a.runtime, a.threads, a.explain, log=log)


def _intro(a, log):
    from . import intro
    bank_path = a.bank or intro.DEFAULT_BANK
    if a.build_bank:
        bank, used = intro.build_bank_from(a.videos)
        intro.save_bank(bank_path, bank)
        log(f"{len(bank['count'])} thumbnails ({intro.title_frames(bank)} title) from {used} videos of "
            f"{intro.OPENING_S // 60} minutes or more -> {bank_path}")
        return 0
    bank = intro.load_bank(bank_path)
    if bank is None:
        raise FileNotFoundError(f"no bank at {bank_path}; build one with yta intro --build-bank")
    for v in a.videos:
        r = intro.interview_start(v, bank)
        log(f"{v.name}\t{r['interview_start_s']}\t{r['how']}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

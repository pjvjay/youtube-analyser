"""Find people doing actions in (YouTube) videos, and search a whole channel for them.

The pieces, bottom up:

    media       ffmpeg: frames, keyframes, single pictures, the sound
    inference   the neural networks on OpenVINO or onnxruntime (Models)
    pose        standing / sitting from body points       motion   the camera, walking, running, jumping
    faces       smiling, talking, silent                   audio    speech, voices, pitch
    actions     the actions and who does one in each frame
    analysis    VideoAnalysis: everything computed about one video, made once and saved
    detector    sections of an action (yta detect)         intro    where the interview starts
    people      hair, clothes, gender, age, voices          query    search queries
    search      a described person doing an action (yta search)
    youtube     listing and downloading with yt-dlp        channel  a whole channel (yta channel)
    cloud       swappable storage and queues (a folder, Azure, ...)
    jobs        the work queue's workers                    service  the HTTP API (yta serve)
    report, review   results files and review videos

Typical use from Python:

    from youtube_analyser import Searcher, parse_query, load_people
    config = load_people()
    searcher = Searcher(config, cache="output")
    rows = searcher.search(Path("video.mp4"), parse_query("woman with purple hair talking", config))
"""
__version__ = "2.0.0"

from .analysis import Settings, VideoAnalysis  # noqa: E402
from .people import load_config as load_people  # noqa: E402
from .query import parse as parse_query  # noqa: E402
from .search import Searcher  # noqa: E402

__all__ = ["Settings", "VideoAnalysis", "Searcher", "parse_query", "load_people", "__version__"]

"""Command-line entry point: ``playertrack analyze`` and ``playertrack landmarks``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .homography import PitchHomography, pitch_landmarks
from .pipeline import AnalysisPipeline, TargetSelection
from .video import iter_frames, read_video_info


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="playertrack",
        description="Track one football player in a match video and compute their metrics.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="Analyse one player in a video")
    analyze.add_argument("video", type=Path, help="Match video file")
    analyze.add_argument(
        "--calibration", type=Path, required=True, help="Pitch calibration JSON file"
    )
    analyze.add_argument(
        "--click",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        required=True,
        help="Pixel position of the player to follow",
    )
    analyze.add_argument("--click-frame", type=int, default=0, help="Frame the click refers to")
    analyze.add_argument("--weights", default="yolo11m.pt", help="YOLO weights file")
    analyze.add_argument("--stride", type=int, default=1, help="Process every N-th frame")
    analyze.add_argument("--device", default=None, help="Inference device, e.g. cpu or 0")
    analyze.add_argument("--out", type=Path, help="Write the JSON result here instead of stdout")

    commands.add_parser("landmarks", help="List the pitch landmark names for calibration files")

    serve = commands.add_parser("serve", help="Start the HTTP API used by the web frontend")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--weights", default="yolo11m.pt", help="YOLO weights file")
    serve.add_argument("--stride", type=int, default=1, help="Process every N-th frame")
    serve.add_argument("--device", default=None, help="Inference device, e.g. cpu or 0")
    return parser


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .api.app import create_app, make_default_analyzer
    except ImportError:
        print("The API needs the 'api' extra: pip install -e 'backend[api]'", file=sys.stderr)
        return 1
    app = create_app(make_default_analyzer(args.weights, args.stride, args.device))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _analyze(args: argparse.Namespace) -> int:
    from .detection import YoloDetector

    info = read_video_info(args.video)
    homography = PitchHomography.from_calibration_file(args.calibration)
    pipeline = AnalysisPipeline(YoloDetector(args.weights, device=args.device), homography)
    total = max(1, info.frame_count // args.stride)

    def report(done: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"\rprocessed {done}/{total} frames", end="", file=sys.stderr, flush=True)

    result = pipeline.run(
        iter_frames(args.video, stride=args.stride, start=args.click_frame),
        fps=info.fps,
        selection=TargetSelection(args.click_frame, (args.click[0], args.click[1])),
        progress=report,
    )
    print(file=sys.stderr)
    if not result.target_found:
        print("No player found at the clicked position.", file=sys.stderr)
        return 1

    payload = json.dumps(result.to_dict(), indent=2)
    if args.out:
        args.out.write_text(payload, encoding="utf-8")
        movement = result.movement
        print(
            f"distance {movement.distance_m:.1f} m, top speed {movement.top_speed_kmh:.1f} km/h,"
            f" passes {result.possession.passes_attempted} -> {args.out}",
            file=sys.stderr,
        )
    else:
        print(payload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "landmarks":
        for name, (x, y) in pitch_landmarks().items():
            print(f"{name:36s} ({x:6.2f}, {y:6.2f})")
        return 0
    if args.command == "serve":
        return _serve(args)
    return _analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())

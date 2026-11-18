# Licenta 2027 – Single-player tracking in football video

[![CI](https://github.com/craciunnicolas10/Licenta-2027/actions/workflows/ci.yml/badge.svg)](https://github.com/craciunnicolas10/Licenta-2027/actions/workflows/ci.yml)

Bachelor's thesis project (Computer Science, Transilvania University of Brașov).

Pick one player in a football match video and get a report on what they did:
heatmap, distance covered, top speed, sprints, time on the ball and passes.
The system detects people and the ball with YOLO, tracks them across frames,
keeps hold of the selected player through occlusions, and projects image
positions onto a real-size pitch with a homography so the numbers come out in
metres and km/h.

## How it works

```
video frame ─► YOLO detection ─► IoU tracker ─► target follower ─► homography ─► pitch positions (m)
                (players, ball)   (track IDs)    (re-acquires the                       │
                                                  selected player)          ┌───────────┴───────────┐
                                                                            ▼                       ▼
                                                                   distance, speed,        possession spells,
                                                                   sprints, heatmap        passes
```

| Stage | Module | Notes |
|---|---|---|
| Video I/O | `video.py` | frame iterator with stride (skipped frames are not decoded) |
| Detection | `detection.py` | Ultralytics YOLO behind a `Detector` protocol; works with COCO or football-specific weights |
| Tracking | `tracking.py` | ByteTrack-style two-stage IoU association with constant-velocity prediction |
| Target selection | `selection.py` | click-to-select, re-acquisition when the tracker assigns a new ID |
| Pitch mapping | `homography.py` | image → pitch homography from ≥4 clicked landmarks, optional RANSAC |
| Movement metrics | `metrics.py` | smoothed trajectory, distance, top / average speed, sprints (> 25.2 km/h for ≥ 1 s), heatmap |
| Possession | `possession.py` | closest player within 1.5 m of the ball, flicker filtering, pass detection |
| Orchestration | `pipeline.py`, `cli.py` | end-to-end run, JSON result |

## Getting started

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e "backend[detect,dev]"
```

Run the tests (they use a synthetic scene, so no weights or videos are needed):

```bash
cd backend
pytest
```

## Usage

1. Save a frame and read pixel coordinates from it (any image viewer that shows
   the cursor position works, e.g. Paint):

   ```bash
   playertrack snapshot match.mp4 --frame 0 --out frame.png
   ```

2. Calibrate the pitch: pick at least four pitch landmarks in that frame and
   write them to a JSON file – see [docs/calibration.md](docs/calibration.md) and
   [examples/calibration.example.json](examples/calibration.example.json).
   Check it by drawing the projected pitch lines over the frame; they should sit
   on the real markings:

   ```bash
   playertrack snapshot match.mp4 --calibration calibration.json --out check.png
   ```

3. Find the player: `--detect` draws every detection with the point to click on.

   ```bash
   playertrack snapshot match.mp4 --detect --out players.png
   ```

4. Run the analysis:

   ```bash
   playertrack analyze match.mp4 --calibration calibration.json --click 812 455 --out result.json
   ```

`result.json` contains the movement metrics, a 21 × 14 heatmap, time spent in
each third, possession summary, the list of passes and the full trajectory.

```bash
playertrack landmarks   # list the landmark names usable in calibration files
```

### HTTP API

```bash
pip install -e "backend[detect,api]"
playertrack serve --port 8000        # interactive docs at http://127.0.0.1:8000/docs
```

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/analyses` | multipart form: `video`, `calibration` (JSON string), `click_x`, `click_y`, optional `click_frame`. Returns `202` with the job. |
| `GET` | `/api/analyses/{id}` | `status` (`queued` / `running` / `done` / `failed`), `frames_processed`, and `result` or `error` |
| `GET` | `/api/health` | liveness check |

Jobs run in the background one at a time and are kept in memory. Allowed
browser origins are set with `PLAYERTRACK_CORS_ORIGINS` (default
`http://localhost:5173`, the Vite dev server).

## Project layout

```
backend/
  src/playertrack/   analysis library and CLI
  tests/             pytest suite (synthetic scenes)
docs/                design notes
examples/            sample configuration files
```

## Roadmap

Work happens on feature branches that are merged into `main` through pull requests.

- [x] Detection, tracking and target re-acquisition
- [x] Homography from manual calibration
- [x] Distance, speed, sprints, heatmap
- [x] Possession and pass detection
- [ ] `feature/api` – FastAPI service: upload a video, run the analysis as a background job, poll for the result
- [ ] `feature/frontend` – React app: upload, click on the player, interactive heatmap and stats dashboard
- [ ] `feature/team-classification` – split players into teams by jersey colour so passes can be marked completed / intercepted
- [ ] `feature/pitch-keypoints` – detect pitch keypoints in every frame to support moving broadcast cameras
- [ ] Evaluation on annotated footage (e.g. SoccerNet tracking) and comparison with ByteTrack / BoT-SORT

## Tech stack

Python · NumPy · OpenCV · Ultralytics YOLO · FastAPI · React · GitHub Actions

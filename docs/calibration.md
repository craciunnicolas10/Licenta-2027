# Pitch calibration

The analysis needs to know how image pixels map onto the pitch. You provide this
as a JSON file with at least four points that you can see in the video, each one
given both in image pixels and as a position on the pitch.

```json
{
  "pitch": { "length": 105, "width": 68 },
  "ransac_threshold_m": 1.0,
  "points": [
    { "image": [312, 188], "landmark": "halfway_top" },
    { "image": [1590, 214], "landmark": "right_penalty_box_top_goal_line" },
    { "image": [1402, 640], "landmark": "right_penalty_box_bottom_corner" },
    { "image": [240, 1010], "landmark": "halfway_bottom" },
    { "image": [903, 512], "pitch": [70.0, 34.0] }
  ]
}
```

- `pitch` – real pitch size in metres. Defaults to 105 x 68.
- `points[].image` – pixel position in the video frame (x to the right, y down).
- `points[].landmark` – a named pitch point; run `playertrack landmarks` for the list.
- `points[].pitch` – or give the pitch position in metres directly.
- `ransac_threshold_m` – optional. With more than four points, RANSAC drops
  points whose error exceeds this many metres (useful for a mis-click).

## Pitch coordinates

The origin is the top-left corner flag as seen from the main camera. `x` runs
along the near/far touchline from 0 to `length`, `y` runs along the goal line
from 0 to `width`.

```
(0,0)  ┌─────────────────┬─────────────────┐  (105,0)
       │                 │                 │
       │                 │                 │
       │            (52.5,34)              │
       │                 │                 │
       │                 │                 │
(0,68) └─────────────────┴─────────────────┘  (105,68)
```

## Tips

- Spread the points over the whole visible area; four points bunched in one
  corner give large errors on the other side of the pitch.
- Never pick three points on one straight line (for example three points on the
  halfway line) — the homography becomes degenerate.
- The current version assumes the camera does not move. For broadcast footage
  with panning and zoom, calibrate on a short clip from a static wide shot.
  Automatic per-frame calibration is on the roadmap.

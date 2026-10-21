"""Draw Cosmos3's controls for one pose on a CARLA map, without CARLA or the model.

    check_controls.py <map.xodr> <out dir> [x y yaw_deg]

Exports the map to RDS-HQ with roadgen, puts a camera (front wide, 1.6 m up)
and a Pandar128 (1.94 m up) at the pose (OpenDRIVE frame), a car 15 m ahead,
and writes the WSM frame, the LiDAR control's range image and the timings.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np


def pose(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    m = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    m[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    m[:3, 3] = [x, y, z]
    return m


def main() -> None:
    xodr, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    import roadgen

    from cosmos3_canvas.controls import Box, LidarControl, WsmCamera, camera_pose_cv

    t = time.perf_counter()
    road_map = roadgen.read_opendrive(str(xodr), sampling=0.5)
    clip = road_map.export_rdshq(str(out / "rds_hq"), trace=False)
    timings = {"export": time.perf_counter() - t}
    if len(sys.argv) > 3:
        x, y, yaw = (float(v) for v in sys.argv[3:6])
    else:  # the first lane-line vertex, along it
        from cosmos_auto.wm.minimap_utils import simplify_minimap

        lines, _ = simplify_minimap(str(out / "rds_hq" / "3d_lanelines" / f"{clip}.tar"))
        line = max(lines, key=len)
        x, y = line[len(line) // 2][:2]
        dx, dy = (line[len(line) // 2 + 1] - line[len(line) // 2])[:2]
        yaw = math.degrees(math.atan2(dy, dx))
    ego = pose(x, y, 0.0, math.radians(yaw))
    z = float(road_map_height(out / "rds_hq", clip, x, y))
    ego[2, 3] = z
    camera = ego @ pose(1.5, 0.0, 1.6, 0.0)
    lidar = ego @ pose(0.0, 0.0, 1.94, 0.0)
    car = Box("1", "Car", np.array([4.5, 1.9, 1.5]), ego @ pose(15.0, 0.0, 0.75, 0.0))

    t = time.perf_counter()
    wsm = WsmCamera(out / "rds_hq", clip, "camera_front_wide_120fov")
    frame = wsm.render(camera_pose_cv(camera), [car])
    timings["wsm_first"] = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(5):
        wsm.render(camera_pose_cv(camera), [car])
    timings["wsm_frame"] = (time.perf_counter() - t) / 5

    t = time.perf_counter()
    lidar_control = LidarControl(out / "rds_hq", clip)
    timings["lidar_static"] = time.perf_counter() - t
    timings["lidar_points"] = int(lidar_control.static.shape[0])
    t = time.perf_counter()
    ranges = lidar_control.render(lidar, lidar @ pose(1.0, 0.0, 0.0, 0.0), [car])
    timings["lidar_sweep"] = time.perf_counter() - t

    import cv2

    cv2.imwrite(str(out / "wsm.png"), frame[:, :, ::-1])
    image = np.clip(ranges / 60.0, 0, 1)
    cv2.imwrite(str(out / "lidar_range.png"), (255 * (1 - image) * (ranges > 0)).astype(np.uint8))
    timings.update(pose=[x, y, z, yaw], wsm_nonblack=float((frame.sum(-1) > 0).mean()),
                   lidar_hits=float((ranges > 0).mean()))
    (out / "timings.json").write_text(json.dumps(timings, indent=1))
    print(json.dumps(timings, indent=1))


def road_map_height(root: Path, clip: str, x: float, y: float) -> float:
    """The nearest lane-line vertex's height (the map's ground there)."""
    from cosmos_auto.wm.minimap_utils import simplify_minimap

    lines, _ = simplify_minimap(str(root / "3d_lanelines" / f"{clip}.tar"))
    points = np.concatenate(lines)
    return points[np.argmin(np.hypot(points[:, 0] - x, points[:, 1] - y)), 2]


if __name__ == "__main__":
    main()

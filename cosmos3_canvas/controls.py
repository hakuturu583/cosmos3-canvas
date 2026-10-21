"""Cosmos3-Nano-Sim-Auto's controls, drawn frame by frame from a live scene.

The model is conditioned on two control streams, both rendered by the cookbook's
``cosmos_auto`` package from an RDS-HQ clip (scene files on disk, every frame
known in advance):

* WSM v4 control videos (``cosmos_auto.wm``): the map's lane lines, road
  boundaries, crosswalks, signs, lights and the actors' boxes, drawn in the
  camera's F-theta lens at 1280x720;
* LiDAR HD-map range maps (``cosmos_auto.lidar``): the same map layers and box
  edges as dense points, rolled into the Pandar128's range image with its
  rolling shutter.

Here the map comes from roadgen's RDS-HQ export of the CARLA map's OpenDRIVE
(the layers are static; their tars are read as the CLI reads them), and the
per-frame inputs (camera / LiDAR poses, boxes) from the simulator, so one frame
or sweep is drawn at a time with the CLI's own drawing functions. Three steps
of the CLI read the whole clip and are replaced by causal ones:

* the below-ego-level ground plane (a +-10 frame moving average of the camera
  pose's plane) is averaged over the last 10 frames only;
* static boxes are not averaged over the clip (``fix_static_objects``), nor are
  boxes interpolated between labels (every frame has its own);
* traffic lights have no per-frame state (carla_canvas does not send light
  states): they are drawn in the "unknown" colour.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

WSM_SIZE = (1280, 720)  # (W, H)
#: FLU body axes -> OpenCV camera axes (x right, y down, z forward): camera-to-body rotation
CV_TO_FLU = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
HYPERION_CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_front_tele_30fov",
)


def camera_pose_cv(sensor_to_world_flu: np.ndarray) -> np.ndarray:
    """4x4 OpenCV-axes camera-to-world from a FLU sensor-to-world pose."""
    pose = np.array(sensor_to_world_flu, np.float64)
    pose[:3, :3] = pose[:3, :3] @ CV_TO_FLU
    return pose


@dataclass
class Box:
    """One actor's box in the map frame (right-handed, z up)."""

    track_id: str
    object_type: str  # Car, Truck, Pedestrian, Cyclist, Others
    lwh: np.ndarray  # (3,)
    box_to_world: np.ndarray  # (4, 4): the box centre, x along its length


def objects_dict(boxes: list[Box]) -> dict[str, dict[str, Any]]:
    """RDS-HQ's per-frame ``all_object_info`` record."""
    return {
        box.track_id: {
            "object_type": box.object_type,
            "object_is_moving": True,
            "object_lwh": [float(v) for v in box.lwh],
            "object_to_world": np.asarray(box.box_to_world, np.float64).tolist(),
        }
        for box in boxes
    }


def default_ftheta(camera_key: str):
    """The cookbook's default Hyperion 8 F-theta intrinsics (1924x1084), rescaled to 720 rows."""
    from cosmos_auto.wm.camera.ftheta import FThetaCamera
    from cosmos_auto.wm.wds_utils import get_sample

    import cosmos_auto.wm

    tar = Path(cosmos_auto.wm.__file__).parent / "assets" / "default_ftheta_intrinsic.tar"
    intrinsics = get_sample(str(tar))[f"ftheta_intrinsic.{camera_key}.npy"]
    camera = FThetaCamera.from_numpy(intrinsics, device="cpu")
    camera.rescale(WSM_SIZE[1] / camera.height)
    return camera


def wsm_config(input_root: str | Path):
    """The CLI's effective configuration (conf/render.yaml, version v4)."""
    from hydra import compose, initialize_config_dir

    import cosmos_auto.wm

    conf = Path(cosmos_auto.wm.__file__).parent / "conf"
    with initialize_config_dir(config_dir=str(conf), version_base=None):
        return compose(config_name="render", overrides=[f"input_root={input_root}", "version=v4",
                                                       "output_root=/dev/null"])


class WsmCamera:
    """One camera's WSM v4 control frames, drawn one at a time (``_render_sample_hdmap_v4_impl``'s loop)."""

    def __init__(self, input_root: str | Path, clip: str, camera_key: str, plane_window: int = 10) -> None:
        self.input_root, self.clip = str(input_root), clip
        self.camera_key = camera_key
        self.cfg = wsm_config(input_root)
        self.camera = default_ftheta(camera_key)
        self.plane_window = plane_window
        self._static: dict[str, Any] | None = None
        self._planes: deque[dict[str, np.ndarray]] = deque(maxlen=plane_window)
        self.pose_init: np.ndarray | None = None

    def _layer(self, name: str) -> str:
        return str(Path(self.input_root) / name / f"{self.clip}.tar")

    def _build_static(self, pose_init: np.ndarray) -> None:
        from cosmos_auto.wm import cli
        from cosmos_auto.wm.bbox_utils import build_per_vertex_color_map

        cfg, camera = self.cfg, self.camera
        minimap_types = [t for t in cfg.minimap_spec.minimap_types
                         if t not in ["lanelines", "traffic_lights", "traffic_signs", "road_markings"]]
        minimap_types, colored_boundaries = cli.route_road_boundary_layer(minimap_types, cfg.use_road_boundary_colors)
        minimap_types = [t for t in minimap_types if Path(self._layer(f"3d_{t}")).exists()]
        minimap_data, minimap_to_rgb = cli.prepare_minimap_data([self._layer(f"3d_{t}") for t in minimap_types], "v3")
        lanelines = cli.prepare_laneline_type_geometry_data(self._layer("3d_lanelines"))
        tl_positions, _, tl_to_rgb, tl_legacy = cli.prepare_traffic_light_status_data(
            self._layer("3d_traffic_lights"), self._layer("__no_light_states__"), "v4")
        signs, sign_to_rgb, other_sign_rgb = cli.prepare_traffic_sign_data_v4(
            self._layer("3d_traffic_signs"), sign_color_version="v4", hdmap_color_version="v3")
        markings = cli.prepare_road_marking_data_v4(
            self._layer("3d_road_markings"), hdmap_color_version="v3", road_marking_color_version="v4",
            use_arrow_colors=cfg.use_road_marking_arrow_colors, use_arrow_shapes=cfg.use_road_marking_arrow_shapes,
            arrow_size_scale=cfg.road_marking_arrow_size_scale, use_text_colors=cfg.use_road_marking_text_colors,
            curve_subdivide=cfg.curve_subdivide, use_interior_tessellation=cfg.use_road_marking_interior_tessellation,
            interior_tessellation_max_edge_m=cfg.road_marking_interior_tessellation_max_edge_m)
        below = cfg.filter_below_ego_level
        minimap = cli.preprocess_minimap_for_rendering(minimap_data, minimap_to_rgb, curve_subdivide=cfg.curve_subdivide)
        geometry = cli.build_worldspace_minimap_geometry(minimap, camera, pose_init, below_level_filter=below)
        geometry += cli.build_worldspace_laneline_geometry(lanelines, camera, pose_init, below_level_filter=below)
        if colored_boundaries and Path(self._layer("3d_road_boundaries")).exists():
            boundaries = cli.load_colored_road_boundaries(self._layer("3d_road_boundaries"))
            geometry += cli.build_worldspace_road_boundaries(boundaries, camera, pose_init, below_level_filter=below)
        if below:
            geometry += cli.build_worldspace_road_marking_geometry(markings, camera, below_level_filter=True)
        self._static = {
            "geometry": geometry, "markings": markings, "tl": (tl_positions, tl_to_rgb, tl_legacy),
            "signs": (signs, sign_to_rgb, other_sign_rgb), "bbox_colors": build_per_vertex_color_map("v3"),
        }
        self.pose_init = pose_init

    def _level_plane(self, pose: np.ndarray) -> dict[str, np.ndarray]:
        from cosmos_auto.wm.occupancy_utils import extract_ground_plane_from_pose

        self._planes.append(extract_ground_plane_from_pose(pose, self.pose_init))
        normal = np.mean([p["normal"] for p in self._planes], axis=0)
        return {"normal": normal / np.linalg.norm(normal),
                "ground_point": np.mean([p["ground_point"] for p in self._planes], axis=0)}

    def render(self, camera_pose: np.ndarray, boxes: list[Box]) -> np.ndarray:
        """(720, 1280, 3) uint8: the control frame for an OpenCV camera-to-world pose (map frame)."""
        import cv2
        from cosmos_auto.wm import cli

        if self._static is None:
            self._build_static(camera_pose)
        static, cfg, camera = self._static, self.cfg, self.camera
        plane = self._level_plane(camera_pose) if cfg.filter_below_ego_level else None
        world_to_camera = np.linalg.inv(camera_pose).astype(np.float32)
        geometry = list(static["geometry"])
        for geom in geometry:
            geom.set_world_to_camera(world_to_camera)
            if plane is not None:
                geom.set_level_plane(plane)
        tl_positions, tl_to_rgb, tl_legacy = static["tl"]
        geometry += cli.create_traffic_light_status_geometry_objects_from_data_v4(
            tl_positions, None, 0, camera_pose, camera, tl_to_rgb, legacy_state_map=tl_legacy, level_plane=plane)
        signs, sign_to_rgb, other_sign_rgb = static["signs"]
        geometry += cli.create_traffic_sign_geometry_objects_from_data_v4(
            signs, camera_pose, camera, sign_to_rgb, other_sign_rgb,
            single_sided_categories=cfg.single_sided_sign_categories, level_plane=plane)
        if not cfg.filter_below_ego_level:
            geometry += cli.create_road_marking_geometry_objects_from_data_v4(static["markings"], camera_pose, camera)
        geometry += cli.create_bbox_geometry_objects_for_frame(
            objects_dict(boxes), camera_pose, camera, fill_face=cfg.bbox_fill_face, fill_face_style="solid",
            object_type_to_per_vertex_color=static["bbox_colors"], color_version="v3", line_width=4,
            edge_color=[200, 200, 200], curve_subdivide=cfg.curve_subdivide)
        frame = cli.render_geometries(geometry, camera.height, camera.width, depth_max=200,
                                      depth_gradient=cfg.depth_gradient)
        if (frame.shape[1], frame.shape[0]) != WSM_SIZE:
            frame = cv2.resize(frame, WSM_SIZE, interpolation=cv2.INTER_LINEAR)
        return frame


class LidarControl:
    """HD-map range maps of the Pandar128, one sweep at a time (``render_hdmap_range_map_for_sweep``)."""

    def __init__(self, input_root: str | Path, clip: str, device: str = "cuda", crop_radius_m: float = 130.0) -> None:
        import torch
        from cosmos_auto.lidar.hdmap_points import load_static_map_points
        from cosmos_auto.lidar.render import DEFAULT_MINIMAP_TYPES
        from cosmos_auto.lidar.sensor_model import SpinningLidarModel

        self.model = SpinningLidarModel.default()
        self.device = torch.device(device)
        layers = load_static_map_points(input_root, clip, DEFAULT_MINIMAP_TYPES, missing_ok=True)
        points = [torch.as_tensor(p, dtype=torch.float32) for p in layers.values() if len(p)]
        self.static = (torch.cat(points) if points else torch.zeros((0, 3))).to(self.device)
        self.crop_radius = crop_radius_m

    def render(self, lidar_to_world_start: np.ndarray, lidar_to_world_end: np.ndarray,
               boxes: list[Box]) -> np.ndarray:
        """(128, 1800) float32 metres, 0: nothing (``save_control``'s pooling of the 3600 columns)."""
        import torch
        from cosmos_auto.lidar.hdmap_points import bbox_points_for_frames
        from cosmos_auto.lidar.render import render_hdmap_range_map_for_sweep

        centre = torch.as_tensor(lidar_to_world_start[:3, 3], dtype=torch.float32, device=self.device)
        near = self.static[(self.static[:, :2] - centre[:2]).norm(dim=1) < self.crop_radius]
        box_points = bbox_points_for_frames({"000000.all_object_info.json": objects_dict(boxes)}, [0])[0]
        points = torch.cat([near, torch.as_tensor(box_points, dtype=torch.float32, device=self.device).reshape(-1, 3)])
        ranges = render_hdmap_range_map_for_sweep(self.model, points, lidar_to_world_start, lidar_to_world_end)
        ranges = ranges.float().cpu().numpy()
        ranges = np.where(np.isfinite(ranges) & (ranges > 0), ranges, 0.0)
        groups = ranges.reshape(ranges.shape[0], 1800, ranges.shape[1] // 1800)
        nearest = np.where(groups > 0, groups, np.inf).min(axis=-1)
        return np.where(np.isfinite(nearest), nearest, 0.0).astype(np.float32)


def lidar_control_frames(ranges: np.ndarray) -> np.ndarray:
    """[S,128,1800] metres -> the model's [3,S,128,1800] control (range, hit, valid)."""
    occupied = (ranges > 0).astype(np.float32)
    return np.stack([ranges.astype(np.float32), occupied, occupied])

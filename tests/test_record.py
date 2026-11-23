"""Recording a rollout's WSM and generated frames, and laying them out."""

from __future__ import annotations

import numpy as np

from cosmos3_canvas.record import Recorder, compose, overlay


def test_overlay_draws_only_the_wsm_lines() -> None:
    rgb = np.full((2, 2, 3), 100, np.uint8)
    wsm = np.zeros((2, 2, 3), np.uint8)
    wsm[0, 0] = (255, 0, 0)
    out = overlay(rgb, wsm, alpha=0.5)
    assert tuple(out[0, 0]) == (178, 50, 50)  # rounded, 177.5 -> 178
    assert (out[1:] == 100).all() and tuple(out[0, 1]) == (100, 100, 100)


def test_recorded_frames_compose_into_one_image(tmp_path) -> None:
    cameras = ["camera_front_wide_120fov", "camera_front_tele_30fov"]
    recorder = Recorder(tmp_path, cameras, (48, 80))
    recorder.new_rollout()
    recorder.put("wsm", range(1, 3), np.zeros((2, 2, 72, 128, 3), np.uint8))  # shrunk to 48x80
    recorder.put("rgb", range(1, 3), np.full((2, 2, 48, 80, 3), 128, np.uint8))
    recorder.flush()
    assert sorted(p.name for p in recorder.dir.iterdir())[:2] == [
        "rgb_camera_front_tele_30fov_00001.jpg", "rgb_camera_front_tele_30fov_00002.jpg"]
    image = compose(recorder.dir, 2, cameras)
    # two panels (generated, overlay), each a 3-wide grid of the layout's two rows used, 8 px apart
    assert image.shape == (2 * 48, 2 * 3 * 80 + 8, 3)
    assert compose(recorder.dir, 2, cameras, ("wsm",)).shape == (2 * 48, 3 * 80, 3)


def test_carla_panels_show_the_carla_frame_and_the_wsm_over_it(tmp_path) -> None:
    cameras = ["camera_front_wide_120fov"]
    recorder = Recorder(tmp_path, cameras, (48, 80))
    recorder.new_rollout({"map": "Town10HD_Opt"})
    wsm = np.zeros((1, 1, 48, 80, 3), np.uint8)
    wsm[..., 36, :, :] = 255  # a line, below the labels
    recorder.put("wsm", range(0, 1), wsm)
    recorder.put("rgb", range(0, 1), np.full((1, 1, 48, 80, 3), 128, np.uint8))
    recorder.put("carla", range(0, 1), np.full((1, 1, 48, 80, 3), 30, np.uint8))
    recorder.poses(0.05, {"ego": {"id": 1, "type_id": "vehicle.x", "pose": [0, 0, 0, 0, 0, 0]},
                          "actors": {}, "cameras": {}})
    recorder.flush()
    assert (recorder.dir / "meta.json").exists() and (recorder.dir / "poses.jsonl").read_text().count("\n") == 1
    image = compose(recorder.dir, 0, cameras, ("carla", "carla_overlay"))
    tile = 80  # one camera's column in the 3-wide grid
    carla, over = image[:, tile:2 * tile], image[:, 3 * tile + 8 + tile:3 * tile + 8 + 2 * tile]
    assert abs(int(carla[30, 40, 0]) - 30) <= 3 and abs(int(over[30, 40, 0]) - 30) <= 3
    assert over[36, 40, 0] > 120  # the WSM line drawn over the CARLA frame


def test_frames_are_framed_in_their_phase_colour() -> None:
    from cosmos3_canvas.record import PHASE_COLORS, frame_phase, parse_phases

    phases = parse_phases(["6.95:policy", "0:warm-up"])
    assert phases == [(0.0, "warm-up"), (6.95, "policy")]
    image = np.zeros((200, 400, 3), np.uint8)
    assert tuple(frame_phase(image, 1.0, phases)[199, 0]) == PHASE_COLORS[0]
    assert tuple(frame_phase(image, 7.0, phases)[199, 0]) == PHASE_COLORS[1]
    assert (frame_phase(image, 1.0, [(2.0, "later")]) == 0).all()  # before any phase: as it was


def test_a_path_projects_through_the_recorded_lens() -> None:
    from cosmos3_canvas.record import project

    camera = {"image_width": 832, "image_height": 480,
              "lens": {"principal_point": [416.0, 240.0], "angle_to_pixeldist_poly": [0.0, 400.0],
                       "max_angle": 1.2, "linear_cde": [1.0, 0.0, 0.0]}}
    pose = [0.0, 0.0, 1.5, 0.0, 0.0, 0.0]  # 1.5 m up, looking along +x
    points = np.array([[10.0, 0.0, 1.5],   # straight ahead: the principal point
                       [10.0, 2.0, 1.5],   # CARLA's +y is to the right
                       [10.0, 0.0, 0.0],   # on the ground: below the centre
                       [-10.0, 0.0, 1.5]])  # behind
    pixels, valid = project(points, pose, camera)
    np.testing.assert_allclose(pixels[0], [416.0, 240.0], atol=1e-9)
    assert pixels[1, 0] > 416.0 and pixels[2, 1] > 240.0
    assert valid.tolist() == [True, True, True, False]


def test_the_latest_plan_by_a_time_is_drawn(tmp_path) -> None:
    import json

    from cosmos3_canvas.record import Trajectories

    path = tmp_path / "plans.jsonl"
    path.write_text("".join(json.dumps({"t": t, "points": [[t, 0, 0]]}) + "\n" for t in (7.0, 7.1)))
    plans = Trajectories(path)
    assert plans.at(6.9) is None
    assert plans.at(7.05)[0, 0] == 7.0 and plans.at(7.1)[0, 0] == 7.1

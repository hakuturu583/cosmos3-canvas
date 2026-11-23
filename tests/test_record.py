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

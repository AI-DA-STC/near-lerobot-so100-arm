from __future__ import annotations

"""Run inference in MuJoCo simulation using a deployed LeRobot policy server.

This script connects to a running instance of `deploy.py`, extracts observations
(two cameras + joint states) from a MuJoCo simulation, queries the inference
API for an action and applies the returned joint positions back to the
simulation.

Usage
-----
python lerobot/scripts/run_inference_sim.py \
    --server-url http://127.0.0.1:8000 \
    --sim-config mujoco/model/so_arm100.xml \
    --fps 30 \
    --num-episodes 5
"""

import argparse
import base64
import logging
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import requests
import torch
import ast

# Some users may want MuJoCo's interactive viewer instead of OpenCV windows.
try:
    
    import mujoco.viewer as mjviewer  # available in mujoco>=2.3.6
    import mujoco
except ImportError:
    mjviewer = None  # fallback handled later

logger = logging.getLogger(__name__)


class MujocoInferenceRunner:
    def __init__(
        self,
        sim_xml_path: Path | str,
        server_url: str,
        fps: int = 30,
        max_steps_per_episode: int | None = None,
        camera_names: List[str] | None = None,
        camera_key_mapping: Dict[str, str] | None = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.fps = fps
        self.dt = 1.0 / fps if fps else None
        self.max_steps = max_steps_per_episode
        self.camera_names = camera_names or ["camera0", "camera1"]
        # Mapping from camera name to policy input key (e.g. side_view -> observation.images.black)
        default_mapping = {"side_view": "observation.images.black", "top_down": "observation.images.white"}
        self.camera_key_mapping = camera_key_mapping or default_mapping

        logger.info("Loading MuJoCo model from %s", sim_xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(sim_xml_path))
        # Allocate a MuJoCo data object
        self.data = mujoco.MjData(self.model)

        # Off-screen renderer for image feeds
        self.renderer = mujoco.Renderer(self.model, height=720, width=1280)

        # Interactive viewer (initialized later if requested and available)
        self.mj_viewer = None

        # GUI display flag will be set later via attribute
        self.display = False  # OpenCV window
        self.use_mjviewer = False  # interactive MuJoCo viewer

    # ---------------------------------------------------------------------
    # Observation helpers
    # ---------------------------------------------------------------------
    def _get_joint_positions(self) -> np.ndarray:
        # Return a copy of qpos (joint positions)
        return np.copy(self.data.qpos)

    def _render_camera(self, camera_name: str) -> np.ndarray:
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id == -1:
            raise ValueError(f"Camera name '{camera_name}' not found in model.")
        # mujoco.Renderer.update_scene signature: update_scene(data, camera=None)
        self.renderer.update_scene(self.data, cam_id)
        img = self.renderer.render()
        return img  # RGB uint8 HxWx3

    def _encode_image_b64(self, img: np.ndarray) -> str:
        _, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _build_observation(self) -> Dict[str, object]:
        obs: Dict[str, object] = {}
        # Add joint positions (first 6 joints expected by policy)
        obs["observation.state"] = self._get_joint_positions()[:6].astype(float).tolist()

        # Add camera images mapped to expected keys
        for cam in self.camera_names:
            try:
                img = self._render_camera(cam)
            except ValueError as e:
                logger.warning(str(e))
                continue
            key = self.camera_key_mapping.get(cam, f"observation.images.{cam}")
            obs[key] = self._encode_image_b64(img)
        return obs

    # ------------------------------------------------------------------
    # API interaction
    # ------------------------------------------------------------------
    def _query_action(self, observation: Dict[str, object]) -> np.ndarray:
        payload = {"observations": observation}
        resp = requests.post(f"{self.server_url}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        action = np.asarray(resp.json()["action"], dtype=np.float32)
        return action

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, num_episodes: int) -> None:
        for ep in range(num_episodes):
            logger.info("Starting episode %d/%d", ep + 1, num_episodes)
            mujoco.mj_resetData(self.model, self.data)
            step_idx = 0
            start_ep = time.perf_counter()
            while True:
                obs = self._build_observation()
                action = self._query_action(obs)

                # Apply action as target joint positions into actuators (simplistic)
                # If number of actuators differs, adjust accordingly
                act_dim = self.model.nu
                if action.shape[0] != act_dim:
                    raise ValueError(f"Action dim {action.shape[0]} != #actuators {act_dim}")
                self.data.ctrl[:] = action

                # Step simulation for one frame duration (may require multiple substeps)
                mujoco.mj_step(self.model, self.data)

                if self.use_mjviewer and self.mj_viewer is not None:
                    self.mj_viewer.sync()

                if self.display:
                    # render from default viewer camera (cam_id 0)
                    self.renderer.update_scene(self.data)
                    frame = self.renderer.render()
                    cv2.imshow("MuJoCo", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)

                # throttle
                if self.dt is not None:
                    elapsed = time.perf_counter() - start_ep - step_idx * self.dt
                    if elapsed < self.dt:
                        time.sleep(self.dt - elapsed)

                step_idx += 1
                if (self.max_steps and step_idx >= self.max_steps) or (
                    self.dt is None and step_idx >= self.fps * 10
                ):
                    break
            logger.info("Episode %d finished after %d steps", ep + 1, step_idx)

        if self.display:
            cv2.destroyAllWindows()

        if self.use_mjviewer and self.mj_viewer is not None:
            self.mj_viewer.close()


# -----------------------------------------------------------------------
# Command line interface
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="URL of the inference server started with deploy.py",
    )
    parser.add_argument(
        "--sim-config",
        type=Path,
        required=True,
        help="Path to MuJoCo .xml model file to load",
    )
    parser.add_argument("--fps", type=int, default=30, help="Control loop frequency")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of pick-and-place episodes to execute",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=300,
        help="Maximum number of control steps per episode (optional)",
    )
    parser.add_argument(
        "--camera-names",
        type=str,
        nargs="*",
        default=["camera0", "camera1"],
        help="Names of MuJoCo cameras used as observations (e.g. --camera-names top_down side_view or --camera-names\n        \"[top_down,side_view]\")",
    )

    parser.add_argument("--display", action="store_true", help="Show MuJoCo rendering window")
    parser.add_argument("--mjviewer", action="store_true", help="Use MuJoCo interactive viewer (GLFW). Overrides --display if set.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Allow users passing list in a single quoted string (e.g. "[top_down,side_view]")
    cam_names = args.camera_names
    if len(cam_names) == 1:
        single = cam_names[0]
        if "," in single or (single.startswith("[") and single.endswith("]")):
            try:
                # Try literal eval first: handles "['a','b']" or "[a,b]"
                parsed = ast.literal_eval(single)
                if isinstance(parsed, (list, tuple)):
                    cam_names = list(parsed)
                else:
                    cam_names = [p.strip() for p in single.strip("[]").split(",") if p.strip()]
            except Exception:
                cam_names = [p.strip().strip("'\"") for p in single.strip("[]").split(",") if p.strip()]

    runner = MujocoInferenceRunner(
        sim_xml_path=args.sim_config,
        server_url=args.server_url,
        fps=args.fps,
        max_steps_per_episode=args.episode_length,
        camera_names=cam_names,
    )

    runner.display = args.display and (not args.mjviewer)
    runner.use_mjviewer = args.mjviewer and (mjviewer is not None)

    if runner.use_mjviewer:
        if mjviewer is None:
            logger.warning("MuJoCo.viewer not available in this mujoco version; falling back to OpenCV display.")
            runner.use_mjviewer = False
            runner.display = True
        else:
            runner.mj_viewer = mjviewer.launch_passive(runner.model, runner.data)

    runner.run(args.num_episodes) 
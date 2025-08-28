from __future__ import annotations

"""Deployment script to host a LeRobot policy as a REST server.

Example
-------
python lerobot/scripts/deploy.py --repo-id <hf_username/model_name> --host 0.0.0.0 --port 8000

The server exposes a single `/predict` endpoint expecting a JSON payload with:
- `observations`: dictionary with keys exactly matching the policy inputs. Each value must be
  - for images: a base64 encoded RGB image string (H, W, 3, uint8)
  - for vector observations (e.g. joint states): a list of floats

The endpoint returns
```json
{"action": [float, ...]}
```
corresponding to the predicted joint command.
"""

import argparse
import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.factory import get_policy_class
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.robot_devices.control_utils import predict_action

logger = logging.getLogger(__name__)


class ObservationPayload(BaseModel):
    observations: Dict[str, Any]


class DeployedPolicyServer:
    def __init__(self, repo_id: str, device: str | None = None, prompt: str | None = None):
        # Determine the concrete policy class from the config file on the Hub
        cfg = PreTrainedConfig.from_pretrained(repo_id, local_files_only=False)
        policy_cls = get_policy_class(cfg.type)
        self.policy: PreTrainedPolicy = policy_cls.from_pretrained(repo_id, config=cfg)  # type: ignore[arg-type]
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy.to(device)
        self.device = torch.device(device)
        self.policy.eval()
        logger.info("Policy loaded on %s", device)
        self.prompt = prompt

    def _process_observations(self, obs_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert JSON serialisable observations into tensors suitable for the policy.

        Keys containing the substring "image" are treated as base-64 PNG images.
        Other numeric lists are mapped to float32 tensors.
        String prompts are left unchanged and handled later (if supported by the policy).
        """
        processed: Dict[str, Any] = {}
        for key, val in obs_dict.items():
            if isinstance(val, str) and ("image" in key):
                # base64-encoded PNG -> RGB uint8 tensor (H,W,C)
                try:
                    img_bytes = base64.b64decode(val)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(img)
                except Exception as e:
                    raise ValueError(f"Failed to decode image for key '{key}': {e}") from e
                processed[key] = tensor
            elif isinstance(val, (list, tuple)):
                processed[key] = torch.tensor(val, dtype=torch.float32)
            elif isinstance(val, (int, float)):
                processed[key] = torch.tensor([val], dtype=torch.float32)
            else:
                # keep other data (e.g. prompt) as-is
                processed[key] = val
        return processed  # type: ignore[return-value]

    @torch.inference_mode()
    def predict(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        if self.prompt is not None:
            logger.debug("Prompt provided but language-conditioning not implemented; ignoring prompt.")

        processed = self._process_observations(obs_dict)
        action_tensor = predict_action(processed, self.policy, self.device, use_amp=False)
        return action_tensor.numpy().tolist()  # type: ignore[return-value]


def build_app(server: DeployedPolicyServer) -> FastAPI:
    app = FastAPI(title="LeRobot Policy Inference Server")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/predict")
    async def predict(payload: ObservationPayload):  # noqa: D401
        try:
            action = server.predict(payload.observations)
            return {"action": action}
        except Exception as e:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=str(e))

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="Hugging Face repository id of the pretrained policy")
    parser.add_argument("--host", default="127.0.0.1", help="Host to expose the API")
    parser.add_argument("--port", type=int, default=8000, help="Port to expose the API")
    parser.add_argument("--device", type=str, default=None, help="Override device, defaults to cuda if available")
    parser.add_argument("--prompt", type=str, default=None, help="Optional language prompt to inject in every prediction request.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    policy_server = DeployedPolicyServer(args.repo_id, device=args.device, prompt=args.prompt)
    app = build_app(policy_server)

    uvicorn.run(app, host=args.host, port=args.port) 
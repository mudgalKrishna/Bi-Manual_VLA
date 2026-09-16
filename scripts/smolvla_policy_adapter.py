#!/usr/bin/env python3
"""Reference SmolVLA adapter for Intel-side inference.

This is the PyTorch reference path. It deliberately comes before OpenVINO
export so that preprocessing, language tokens, and action postprocessing are
defined in one place and can later be compared with an exported graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch
from transformers import AutoTokenizer

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


CAMERAS = (
    "observation.images.front",
    "observation.images.arm_a",
    "observation.images.arm_b",
    "observation.images.top",
)


class SmolVLAAdapter:
    """Load a trained checkpoint and predict one 12-D action."""

    def __init__(self, checkpoint: str | Path, instruction: str,
                 device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.policy = SmolVLAPolicy.from_pretrained(
            str(checkpoint), local_files_only=True)
        self.policy.to(self.device)
        self.policy.eval()
        self.policy.reset()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        self.instruction = instruction

    def _tokens(self) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [self.instruction],
            padding="max_length",
            max_length=self.policy.config.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "observation.language.tokens": encoded["input_ids"].to(self.device),
            "observation.language.attention_mask": encoded[
                "attention_mask"].bool().to(self.device),
        }

    def predict(self, images: Mapping[str, torch.Tensor],
                state: torch.Tensor) -> torch.Tensor:
        """Return one action tensor shaped ``(1, 12)``.

        Images must be float tensors shaped ``(1, 3, 320, 320)`` with the
        same camera keys as the training dataset. State must be ``(1, 12)``.
        """
        missing = [camera for camera in CAMERAS if camera not in images]
        if missing:
            raise KeyError(f"Missing camera observations: {missing}")
        batch = {camera: images[camera].to(self.device) for camera in CAMERAS}
        batch["observation.state"] = state.to(self.device)
        batch.update(self._tokens())
        with torch.inference_mode():
            return self.policy.select_action(batch)


def dummy_observation() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    image = torch.zeros((1, 3, 320, 320), dtype=torch.float32)
    state = torch.zeros((1, 12), dtype=torch.float32)
    return {camera: image for camera in CAMERAS}, state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instruction", default="set the dinner table")
    args = parser.parse_args()
    adapter = SmolVLAAdapter(args.checkpoint, args.instruction)
    images, state = dummy_observation()
    action = adapter.predict(images, state)
    print({"action_shape": list(action.shape), "action": action.tolist()})

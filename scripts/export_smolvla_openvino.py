#!/usr/bin/env python3
"""Attempt an explicit SmolVLA action-graph export.

This is separate from generic Optimum export because SmolVLA is a custom
LeRobot action policy. The script fixes the language-token interface and
exports the action path with four images, state, and token tensors.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoTokenizer

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


class ActionGraph(torch.nn.Module):
    def __init__(self, policy: SmolVLAPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, front, arm_a, arm_b, top, state, language_tokens,
                language_attention_mask):
        batch = {
            "observation.images.front": front,
            "observation.images.arm_a": arm_a,
            "observation.images.arm_b": arm_b,
            "observation.images.top": top,
            "observation.state": state,
            "observation.language.tokens": language_tokens,
            "observation.language.attention_mask": language_attention_mask,
        }
        return self.policy.select_action(batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--instruction", default="set the dinner table")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "smolvla_action.onnx"

    policy = SmolVLAPolicy.from_pretrained(
        str(checkpoint), local_files_only=True)
    policy.eval()
    policy.reset()
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    tokens = tokenizer(
        [args.instruction],
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    image = torch.zeros((1, 3, 320, 320), dtype=torch.float32)
    state = torch.zeros((1, 12), dtype=torch.float32)
    language_tokens = tokens["input_ids"].to(torch.int64)
    language_mask = tokens["attention_mask"].bool()
    example = (image, image, image, image, state, language_tokens, language_mask)

    with torch.inference_mode():
        reference_action = ActionGraph(policy)(*example)
    print("Reference action shape:", list(reference_action.shape))

    try:
        torch.onnx.export(
            ActionGraph(policy),
            example,
            str(onnx_path),
            input_names=[
                "front", "arm_a", "arm_b", "top", "state",
                "language_tokens", "language_attention_mask",
            ],
            output_names=["action"],
            opset_version=18,
            do_constant_folding=False,
            dynamo=False,
        )
    except Exception as error:
        raise SystemExit(
            "SmolVLA ONNX export was blocked by a model/frontend operation. "
            "The PyTorch reference path is valid; inspect this error before "
            "choosing a supported subgraph boundary.\n"
            f"{type(error).__name__}: {error}"
        ) from error

    print("ONNX written:", onnx_path)
    try:
        import openvino as ov
        ov_model = ov.convert_model(str(onnx_path))
        xml_path = output / "smolvla_action.xml"
        ov.save_model(ov_model, str(xml_path), compress_to_fp16=False)
        print("OpenVINO IR written:", xml_path)
    except Exception as error:
        raise SystemExit(
            "ONNX export succeeded, but OpenVINO conversion failed. "
            "Keep the ONNX file and inspect this frontend error.\n"
            f"{type(error).__name__}: {error}"
        ) from error

    (output / "export_report.json").write_text(
        json.dumps({
            "checkpoint": str(checkpoint),
            "instruction": args.instruction,
            "reference_action_shape": list(reference_action.shape),
            "onnx": str(onnx_path),
            "openvino_xml": str(output / "smolvla_action.xml"),
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

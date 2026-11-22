"""Export the SigLIP2 vision tower with a zero-shot head to ONNX (#151).

The graph takes the project's ``edge_detector.preprocess`` blob (RGB, squash-resized square, 0-1, ImageNet mean/std,
NCHW float32) and outputs ``(1, 2)`` logits ``[no_fire, fire]`` for ``classifier_fire_probability`` (softmax index 1):

1. ``x_siglip = x * 2 * std_in + (2 * mean_in - 1)``: undoes ImageNet normalisation and applies SigLIP's 0.5/0.5;
2. vision tower -> pooled image embedding, L2-normalised;
3. cosine similarity with the precomputed, L2-normalised class text embeddings (a constant in the graph), times
   ``exp(logit_scale)``;
4. ``fire = logsumexp(fire classes)``, ``no_fire = logsumexp(other classes)``.

So ``softmax([no_fire, fire])[1]`` is the CLIP-style softmax probability mass of the fire classes over all classes (not
SigLIP's independent per-prompt sigmoid). ``logit_bias`` is a constant added to every class and cancels in the 2-way
softmax, so it is omitted. Unequal class counts only shift ``fire - no_fire`` by a constant (``log(n_neg / n_fire)`` for
equally similar classes), which changes neither ranking nor recall at the "best safe point" threshold.

Text embeddings use ``padding="max_length", max_length=64`` (how SigLIP text towers were trained).

Example:
    python -m tools.zero_shot_eval.export --model google/siglip2-base-patch16-224 --prompt-set rich --out m.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModel
from transformers import AutoProcessor

from tools.zero_shot_eval.prompts import FIRE_CLASSES
from tools.zero_shot_eval.prompts import PROMPT_SETS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _features(out: torch.Tensor | object) -> torch.Tensor:
    """Return the embedding tensor from ``get_*_features`` (a tensor or an output object, depending on version)."""
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


@torch.no_grad()
def class_embeddings(model: torch.nn.Module, processor: object, prompt_set: dict[str, list[str]]) -> torch.Tensor:
    """Compute L2-normalised class embeddings (mean of the normalised prompt embeddings, renormalised).

    Args:
        model: The SigLIP model.
        processor: Its processor.
        prompt_set: Class name -> prompts.

    Returns:
        Tensor of shape ``(n_classes, dim)`` in the order of ``prompt_set``.
    """
    rows = []
    for prompts in prompt_set.values():
        tok = processor(text=prompts, padding="max_length", max_length=64, return_tensors="pt")
        emb = _features(model.get_text_features(input_ids=tok["input_ids"]))
        emb = torch.nn.functional.normalize(emb, dim=-1).mean(0)
        rows.append(torch.nn.functional.normalize(emb, dim=-1))
    return torch.stack(rows)


class ZeroShotHead(torch.nn.Module):
    """Vision tower + constant text embeddings -> ``[no_fire, fire]`` logits."""

    def __init__(self, model: torch.nn.Module, text_emb: torch.Tensor, fire_mask: list[bool]) -> None:
        """Builds the adapter.

        Args:
            model: The SigLIP model (only the vision tower is kept).
            text_emb: L2-normalised class embeddings ``(n_classes, dim)``.
            fire_mask: Per class, whether it belongs to the fire side.
        """
        super().__init__()
        self.vision = model.vision_model
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("in_scale", 2 * std)
        self.register_buffer("in_shift", 2 * mean - 1)
        scaled = text_emb * model.logit_scale.exp().item()
        fire = torch.tensor(fire_mask)
        self.register_buffer("w_fire", scaled[fire].T.contiguous())
        self.register_buffer("w_other", scaled[~fire].T.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Maps an ImageNet-normalised blob ``(1, 3, S, S)`` to logits ``(1, 2)``."""
        pooled = self.vision(pixel_values=x * self.in_scale + self.in_shift).pooler_output
        img = pooled / pooled.norm(dim=-1, keepdim=True)
        fire = torch.logsumexp(img @ self.w_fire, dim=-1, keepdim=True)
        other = torch.logsumexp(img @ self.w_other, dim=-1, keepdim=True)
        return torch.cat([other, fire], dim=-1)


def build(model_id: str, prompt_set_name: str) -> tuple[ZeroShotHead, int, dict]:
    """Load the model and build the head.

    Args:
        model_id: Hugging Face model id.
        prompt_set_name: Key of ``PROMPT_SETS``.

    Returns:
        ``(head, input_size, meta)``.
    """
    model = AutoModel.from_pretrained(model_id, attn_implementation="eager").eval()
    processor = AutoProcessor.from_pretrained(model_id)
    prompt_set = PROMPT_SETS[prompt_set_name]
    mask = [name in FIRE_CLASSES for name in prompt_set]
    head = ZeroShotHead(model, class_embeddings(model, processor, prompt_set), mask).eval()
    size = int(model.config.vision_config.image_size)
    meta = {
        "model": model_id,
        "prompt_set": prompt_set_name,
        "input_size": size,
        "fire_classes": [n for n, m in zip(prompt_set, mask, strict=True) if m],
        "other_classes": [n for n, m in zip(prompt_set, mask, strict=True) if not m],
        "logit_scale_exp": model.logit_scale.exp().item(),
    }
    return head, size, meta


def main() -> None:
    """Export one (model, prompt set) pair."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-set", required=True, choices=sorted(PROMPT_SETS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    head, size, meta = build(args.model, args.prompt_set)
    dummy = torch.zeros(1, 3, size, size)
    with torch.no_grad():
        torch.onnx.export(
            head,
            (dummy,),
            str(args.out),
            input_names=["input"],
            output_names=["logits"],
            opset_version=args.opset,
            dynamo=False,
            external_data=False,
        )
    meta["opset"] = args.opset
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

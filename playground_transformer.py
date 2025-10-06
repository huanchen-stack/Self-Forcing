"""CUDA graph playground using a randomly initialized CausalWanModel."""

from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, Hashable

import numpy as np
import torch

from wan.modules.causal_model import CausalWanModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_kv_cache(
    num_layers: int,
    batch: int,
    kv_len: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[Dict[str, torch.Tensor]]:
    cache = []
    for _ in range(num_layers):
        cache.append({
            "k": torch.zeros(batch, kv_len, num_heads, head_dim, device=device, dtype=dtype),
            "v": torch.zeros(batch, kv_len, num_heads, head_dim, device=device, dtype=dtype),
            "global_end_index": torch.zeros(1, device=device, dtype=torch.long),
            "local_end_index": torch.zeros(1, device=device, dtype=torch.long),
        })
    return cache


class RandomWanTransformer(torch.nn.Module):
    """Small Wan causal model with random weights for experimentation."""

    def __init__(
        self,
        in_dim: int = 8,
        out_dim: int = 8,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        frames_per_block: int = 2,
        height: int = 16,
        width: int = 16,
        local_attn_size: int = 4,
        sink_size: int = 1,
        text_len: int = 32,
        text_dim: int = 128,
        freq_dim: int = 64,
        ffn_dim: int = 256,
    ) -> None:
        super().__init__()

        self.frames_per_block = frames_per_block
        self.height = height
        self.width = width
        self.patch_size = (1, 2, 2)
        self.frame_seq_len = (height // self.patch_size[1]) * (width // self.patch_size[2])
        self.seq_len = self.frame_seq_len * frames_per_block

        self.model = CausalWanModel(
            model_type="t2v",
            patch_size=self.patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=True,
            cross_attn_norm=True,
        )
        self.model.num_frame_per_block = frames_per_block

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.text_len = text_len
        self.text_dim = text_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.kv_len = max(local_attn_size * self.frame_seq_len, self.seq_len)

    def prepare_caches(self, batch: int, device: torch.device, dtype: torch.dtype) -> list[Dict[str, torch.Tensor]]:
        return init_kv_cache(
            num_layers=self.num_layers,
            batch=batch,
            kv_len=self.kv_len,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        kv_cache: list[Dict[str, torch.Tensor]],
        current_start: int,
        **_: Any,
    ) -> tuple[None, torch.Tensor]:
        # Convert [B, F, C, H, W] -> list of [C, F, H, W]
        x_list = [sample.permute(1, 0, 2, 3) for sample in noisy_image_or_video]
        t = timestep[:, 0]
        context_list = [conditional_dict["prompt_embeds"][i] for i in range(timestep.shape[0])]

        outputs = self.model(
            x=x_list,
            t=t,
            context=context_list,
            seq_len=self.seq_len,
            kv_cache=kv_cache,
            crossattn_cache=None,
            current_start=current_start,
        )

        denoised = torch.stack([
            out.permute(1, 0, 2, 3) for out in outputs
        ])
        return None, denoised.to(dtype=noisy_image_or_video.dtype)


class TransformerGraphCache:
    """Capture and replay transformer calls keyed by (block, step)."""

    def __init__(self, transformer: RandomWanTransformer) -> None:
        self.transformer = transformer
        self.entries: Dict[Hashable, Dict[str, Any]] = {}

    def capture(
        self,
        key: Hashable,
        noisy_sample: torch.Tensor,
        timestep_sample: torch.Tensor,
        call_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        entry = {
            "graph": torch.cuda.CUDAGraph(),
            "noisy_static": torch.empty_like(noisy_sample),
            "timestep_static": torch.empty_like(timestep_sample),
            "output_static": torch.empty_like(noisy_sample),
            "kwargs": call_kwargs,
        }

        entry["noisy_static"].copy_(noisy_sample)
        entry["timestep_static"].copy_(timestep_sample)

        torch.cuda.synchronize()
        with torch.cuda.graph(entry["graph"]):
            _, outputs = self.transformer(
                noisy_image_or_video=entry["noisy_static"],
                timestep=entry["timestep_static"],
                **entry["kwargs"],
            )
            entry["output_static"].copy_(outputs)

        self.entries[key] = entry
        return entry["output_static"]

    def replay(
        self,
        key: Hashable,
        noisy_sample: torch.Tensor,
        timestep_sample: torch.Tensor,
    ) -> torch.Tensor:
        entry = self.entries[key]
        entry["noisy_static"].copy_(noisy_sample)
        entry["timestep_static"].copy_(timestep_sample)
        entry["graph"].replay()
        return entry["output_static"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Wan CUDA-graph playground")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--frames-per-block", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--capture-block", type=int, default=6)
    parser.add_argument("--capture-step", type=int, default=-1,
                        help="Step index within block to capture (-1 = last)")
    parser.add_argument("--cudagraph", action="store_true",
                        help="Enable CUDA graph capture")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device is required for this playground")

    device = torch.device("cuda")
    dtype = torch.float16

    set_seed(args.seed)

    if args.cudagraph:
        os.environ["PRINT_ENABLED"] = "0"
    else:
        os.environ.setdefault("PRINT_ENABLED", "1")

    transformer = RandomWanTransformer(
        frames_per_block=args.frames_per_block,
    ).to(device=device, dtype=dtype)
    transformer.eval()
    transformer.requires_grad_(False)

    kv_cache = transformer.prepare_caches(batch=1, device=device, dtype=dtype)

    text_embeds = torch.randn(1, transformer.text_len, transformer.text_dim, device=device, dtype=dtype)
    conditional_dict = {"prompt_embeds": text_embeds}

    total_frames = args.num_blocks * args.frames_per_block
    noise = torch.randn(1, total_frames, transformer.in_dim, transformer.height, transformer.width,
                        device=device, dtype=dtype)

    if args.capture_step < 0:
        capture_step = args.num_steps - 1
    else:
        capture_step = args.capture_step

    if capture_step < 0 or capture_step >= args.num_steps:
        raise SystemExit("capture_step must fall within [0, num_steps)")

    graph_cache = TransformerGraphCache(transformer) if args.cudagraph else None

    current_start_frame = 0
    capture_key = (args.capture_block, capture_step)

    for block_idx in range(args.num_blocks):
        frame_slice = slice(current_start_frame, current_start_frame + args.frames_per_block)
        noisy_input = noise[:, frame_slice]

        for step_idx in range(args.num_steps):
            timestep = torch.full(
                (1, args.frames_per_block),
                step_idx,
                device=device,
                dtype=torch.int64,
            )

            call_kwargs = {
                "conditional_dict": conditional_dict,
                "kv_cache": kv_cache,
                "current_start": current_start_frame * transformer.frame_seq_len,
            }

            key = (block_idx, step_idx)
            if args.cudagraph and key == capture_key:
                if key in graph_cache.entries:
                    denoised_pred = graph_cache.replay(key, noisy_input, timestep)
                    print(f"Replayed graph for block {block_idx}, step {step_idx}")
                else:
                    denoised_pred = graph_cache.capture(key, noisy_input, timestep, call_kwargs)
                    print(f"Captured graph for block {block_idx}, step {step_idx}")
            else:
                _, denoised_pred = transformer(
                    noisy_image_or_video=noisy_input,
                    timestep=timestep,
                    **call_kwargs,
                )

            if step_idx < args.num_steps - 1:
                noise_term = torch.randn_like(denoised_pred)
                noisy_input = denoised_pred + 0.1 * noise_term

        current_start_frame += args.frames_per_block

    print("Playground completed without CUDA graph capture errors.")


if __name__ == "__main__":
    main()

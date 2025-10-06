"""Quick CUDA-graph smoke test for the Wan causal transformer."""

import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from pipeline import CausalInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _prepare_text_condition(device: torch.device, prompt: str) -> dict:
    text_encoder = WanTextEncoder().to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    conditional_dict = text_encoder(text_prompts=[prompt])
    for key, value in conditional_dict.items():
        conditional_dict[key] = value.to(device=device, dtype=torch.float16)

    return text_encoder, conditional_dict


def _load_transformer(device: torch.device, checkpoint_path: str) -> WanDiffusionWrapper:
    transformer = WanDiffusionWrapper(is_causal=True)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    transformer.load_state_dict(state_dict["generator_ema"])
    transformer.eval()
    transformer.to(device=device, dtype=torch.float16)
    transformer.requires_grad_(False)
    return transformer


class CUDAGraphHarness:
    """Capture and replay a single transformer call."""

    def __init__(self, transformer: WanDiffusionWrapper):
        self.transformer = transformer
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.noisy_static: Optional[torch.Tensor] = None
        self.timestep_static: Optional[torch.Tensor] = None
        self.output_static: Optional[torch.Tensor] = None

    def capture(
        self,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Record the transformer execution for the supplied tensors."""

        self.noisy_static = torch.empty_like(noisy_input)
        self.timestep_static = torch.empty_like(timestep)
        self.output_static = torch.empty_like(noisy_input)

        self.noisy_static.copy_(noisy_input)
        self.timestep_static.copy_(timestep)

        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            result = self.transformer(
                noisy_image_or_video=self.noisy_static,
                timestep=self.timestep_static,
                **kwargs,
            )
            if isinstance(result, tuple):
                result = result[1]
            self.output_static.copy_(result)

        self.graph = graph
        return self.output_static

    def replay(self, noisy_input: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        assert self.graph is not None, "capture() must be called before replay()"
        assert self.noisy_static is not None and self.timestep_static is not None
        assert self.output_static is not None

        self.noisy_static.copy_(noisy_input)
        self.timestep_static.copy_(timestep)
        self.graph.replay()
        return self.output_static


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA-graph playground for Wan transformer")
    parser.add_argument("--checkpoint_path", default="./checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--config_path", default="./configs/self_forcing_dmd.yaml")
    parser.add_argument("--prompt", default="A serene mountain landscape at sunrise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_blocks", type=int, default=8,
                        help="Total number of diffusion blocks to simulate")
    parser.add_argument("--capture_block", type=int, default=7,
                        help="Block index whose transformer call should be captured when using CUDA graphs")
    parser.add_argument("--capture_step", type=int, default=-1,
                        help="Denoising step to capture (-1 for final step) when using CUDA graphs")
    parser.add_argument("--cudagraph", action="store_true",
                        help="Enable CUDA graph capture for the selected block/step")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA device is required for this playground")

    device = torch.device("cuda")
    _set_seed(args.seed)

    if args.cudagraph:
        if args.capture_block < 0 or args.capture_block >= args.num_blocks:
            raise SystemExit("capture_block must be within [0, num_blocks)")
        os.environ["PRINT_ENABLED"] = "0"
    else:
        os.environ.setdefault("PRINT_ENABLED", "1")

    text_encoder, conditional_dict = _prepare_text_condition(device, args.prompt)
    transformer = _load_transformer(device, args.checkpoint_path)

    default_cfg = OmegaConf.load("configs/default_config.yaml")
    user_cfg = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(default_cfg, user_cfg)

    pipeline = CausalInferencePipeline(
        config,
        device=device,
        generator=transformer,
        text_encoder=text_encoder,
        vae=None,
    )

    pipeline._initialize_kv_cache(batch_size=1, dtype=torch.float16, device=device)
    pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.float16, device=device)

    # Prime kernels, then reset caches so the playground starts clean.
    warm_noise = torch.zeros(
        [1, pipeline.num_frame_per_block, 16, 60, 104],
        device=device,
        dtype=torch.float16,
    )
    warm_step = torch.zeros(
        [1, pipeline.num_frame_per_block],
        device=device,
        dtype=torch.int64,
    )
    transformer(
        noisy_image_or_video=warm_noise,
        conditional_dict=conditional_dict,
        timestep=warm_step,
        kv_cache=pipeline.kv_cache1,
        crossattn_cache=pipeline.crossattn_cache,
        current_start=0,
    )
    pipeline._initialize_kv_cache(batch_size=1, dtype=torch.float16, device=device)
    pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.float16, device=device)

    num_blocks = args.num_blocks
    frames_per_block = pipeline.num_frame_per_block
    total_frames = num_blocks * frames_per_block

    noise = torch.randn([1, total_frames, 16, 60, 104], device=device, dtype=torch.float16)

    denoising_steps = list(pipeline.denoising_step_list)
    if args.cudagraph and args.capture_step >= len(denoising_steps):
        raise SystemExit("capture_step must be within the denoising step range")

    if args.capture_step < 0:
        capture_step = len(denoising_steps) - 1
    else:
        capture_step = args.capture_step

    graph_harness = CUDAGraphHarness(transformer) if args.cudagraph else None

    current_start_frame = 0
    for block_idx in range(num_blocks):
        current_frames = frames_per_block
        noisy_input = noise[:, current_start_frame:current_start_frame + current_frames]

        for step_idx, current_timestep in enumerate(denoising_steps):
            timestep = torch.full(
                (1, current_frames),
                int(current_timestep),
                device=device,
                dtype=torch.int64,
            )

            kwargs = dict(
                conditional_dict=conditional_dict,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )

            if args.cudagraph and block_idx == args.capture_block and step_idx == capture_step:
                if graph_harness.graph is None:
                    denoised_pred = graph_harness.capture(
                        noisy_input.clone(),
                        timestep.clone(),
                        **kwargs,
                    )
                    print("Captured CUDA graph for block", block_idx, "step", step_idx)
                else:
                    denoised_pred = graph_harness.replay(noisy_input, timestep)
                    print("Replayed CUDA graph for block", block_idx, "step", step_idx)
            else:
                _, denoised_pred = transformer(
                    noisy_image_or_video=noisy_input,
                    timestep=timestep,
                    **kwargs,
                )

            if step_idx < (len(denoising_steps) - 1):
                noise_term = torch.randn_like(denoised_pred.flatten(0, 1))
                next_timestep = denoising_steps[step_idx + 1]
                noisy_input = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    noise_term,
                    torch.full(
                        (current_frames,),
                        int(next_timestep),
                        device=device,
                        dtype=torch.long,
                    ),
                ).unflatten(0, denoised_pred.shape[:2])

        # Update KV cache with the clean context for the next block.
        zero_step = torch.zeros_like(timestep)
        transformer(
            noisy_image_or_video=denoised_pred,
            timestep=zero_step,
            **kwargs,
        )

        current_start_frame += current_frames

    print("Playground run completed without CUDA graph capture errors.")


if __name__ == "__main__":
    main()

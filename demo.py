"""
Demo for Self-Forcing.
"""

import os
import re
import random
import time
import base64
import argparse
import hashlib
import subprocess
import urllib.request
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from omegaconf import OmegaConf
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import queue
from threading import Thread, Event

from pipeline import CausalInferencePipeline
from demo_utils.constant import ZERO_VAE_CACHE
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from demo_utils.utils import generate_timestamp
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation

from torch.profiler import profile, record_function, ProfilerActivity

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=5001)
parser.add_argument('--host', type=str, default='0.0.0.0')
parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/self_forcing_dmd.pt')
parser.add_argument("--config_path", type=str, default='./configs/self_forcing_dmd.yaml')
parser.add_argument('--trt', action='store_true')
args = parser.parse_args()

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

# Load models
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

text_encoder = WanTextEncoder()

# Global variables for dynamic model switching
current_vae_decoder = None
current_use_taehv = False
fp8_applied = False
torch_compile_applied = False
global frame_number
frame_number = 0
anim_name = ""
frame_rate = 6

def initialize_vae_decoder(use_taehv=False, use_trt=False):
    """Initialize VAE decoder based on the selected option"""
    global current_vae_decoder, current_use_taehv

    if use_trt:
        from demo_utils.vae import VAETRTWrapper
        current_vae_decoder = VAETRTWrapper()
        return current_vae_decoder

    if use_taehv:
        from demo_utils.taehv import TAEHV
        # Check if taew2_1.pth exists in checkpoints folder, download if missing
        taehv_checkpoint_path = "checkpoints/taew2_1.pth"
        if not os.path.exists(taehv_checkpoint_path):
            print(f"taew2_1.pth not found in checkpoints folder {taehv_checkpoint_path}. Downloading...")
            os.makedirs("checkpoints", exist_ok=True)
            download_url = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"
            try:
                urllib.request.urlretrieve(download_url, taehv_checkpoint_path)
                print(f"Successfully downloaded taew2_1.pth to {taehv_checkpoint_path}")
            except Exception as e:
                print(f"Failed to download taew2_1.pth: {e}")
                raise

        class DotDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        class TAEHVDiffusersWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dtype = torch.float16
                self.taehv = TAEHV(checkpoint_path=taehv_checkpoint_path).to(self.dtype)
                self.config = DotDict(scaling_factor=1.0)

            def decode(self, latents, return_dict=None):
                # n, c, t, h, w = latents.shape
                # low-memory, set parallel=True for faster + higher memory
                return self.taehv.decode_video(latents, parallel=False).mul_(2).sub_(1)

        current_vae_decoder = TAEHVDiffusersWrapper()
    else:
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load('wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth', map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)

    current_vae_decoder.eval()
    current_vae_decoder.to(dtype=torch.float16)
    current_vae_decoder.requires_grad_(False)
    current_vae_decoder.to(gpu)
    current_use_taehv = use_taehv

    print(f"✅ VAE decoder initialized with {'TAEHV' if use_taehv else 'default VAE'}")
    return current_vae_decoder


# Initialize with default VAE
vae_decoder = initialize_vae_decoder(use_taehv=False, use_trt=args.trt)

transformer = WanDiffusionWrapper(is_causal=True)
state_dict = torch.load(args.checkpoint_path, map_location="cpu")
transformer.load_state_dict(state_dict['generator_ema'])

text_encoder.eval()
transformer.eval()

transformer.to(dtype=torch.float16)
text_encoder.to(dtype=torch.bfloat16)

text_encoder.requires_grad_(False)
transformer.requires_grad_(False)

pipeline = CausalInferencePipeline(
    config,
    device=gpu,
    generator=transformer,
    text_encoder=text_encoder,
    vae=vae_decoder
)

if low_memory:
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
transformer.to(gpu)

# Flask and SocketIO setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'frontend_buffered_demo'
socketio = SocketIO(app, cors_allowed_origins="*")

generation_active = False
stop_event = Event()
frame_send_queue = queue.Queue()
sender_thread = None
models_compiled = False


def tensor_to_base64_frame(frame_tensor):
    """Convert a single frame tensor to base64 image string."""
    global frame_number, anim_name
    # Clamp and normalize to 0-255
    frame = torch.clamp(frame_tensor.float(), -1., 1.) * 127.5 + 127.5
    frame = frame.to(torch.uint8).cpu().numpy()

    # CHW -> HWC
    if len(frame.shape) == 3:
        frame = np.transpose(frame, (1, 2, 0))

    # Convert to PIL Image
    if frame.shape[2] == 3:  # RGB
        image = Image.fromarray(frame, 'RGB')
    else:  # Handle other formats
        image = Image.fromarray(frame)

    # Convert to base64
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=100)
    if not os.path.exists("./images/%s" % anim_name):
        os.makedirs("./images/%s" % anim_name)
    frame_number += 1
    image.save("./images/%s/%s_%03d.jpg" % (anim_name, anim_name, frame_number))
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def frame_sender_worker():
    """Background thread that processes frame send queue non-blocking."""
    global frame_send_queue, generation_active, stop_event

    print("📡 Frame sender thread started")

    while True:
        frame_data = None
        try:
            # Get frame data from queue
            frame_data = frame_send_queue.get(timeout=1.0)

            if frame_data is None:  # Shutdown signal
                frame_send_queue.task_done()  # Mark shutdown signal as done
                break

            frame_tensor, frame_index, block_index, job_id = frame_data

            # Convert tensor to base64
            base64_frame = tensor_to_base64_frame(frame_tensor)

            # Send via SocketIO
            try:
                socketio.emit('frame_ready', {
                    'data': base64_frame,
                    'frame_index': frame_index,
                    'block_index': block_index,
                    'job_id': job_id
                })
            except Exception as e:
                print(f"⚠️ Failed to send frame {frame_index}: {e}")

            frame_send_queue.task_done()

        except queue.Empty:
            # Check if we should continue running
            if not generation_active and frame_send_queue.empty():
                break
        except Exception as e:
            print(f"❌ Frame sender error: {e}")
            # Make sure to mark task as done even if there's an error
            if frame_data is not None:
                try:
                    frame_send_queue.task_done()
                except Exception as e:
                    print(f"❌ Failed to mark frame task as done: {e}")
            break

    print("📡 Frame sender thread stopped")

@torch.no_grad()
def generate_video_stream___prof___(prompt, seed, enable_torch_compile=False, enable_fp8=False, use_taehv=False):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        with record_function("generate_video_stream"):
            generate_video_stream(prompt, seed, enable_torch_compile, enable_fp8, use_taehv)

    print("============ prof export ============")
    prof.export_chrome_trace(f"trace_{time.time() :8f}.json")
    print("============  prof done  ============")

@torch.no_grad()
def generate_video_stream(prompt, seed, enable_torch_compile=False, enable_fp8=False, use_taehv=False):
    """Generate video and push frames immediately to frontend."""
    global generation_active, stop_event, frame_send_queue, sender_thread, models_compiled, torch_compile_applied, fp8_applied, current_vae_decoder, current_use_taehv, frame_rate, anim_name

    try:
        generation_active = True
        stop_event.clear()
        job_id = generate_timestamp()

        # Start frame sender thread if not already running
        if sender_thread is None or not sender_thread.is_alive():
            sender_thread = Thread(target=frame_sender_worker, daemon=True)
            sender_thread.start()

        # Emit progress updates
        def emit_progress(message, progress):
            try:
                socketio.emit('progress', {
                    'message': message,
                    'progress': progress,
                    'job_id': job_id
                })
            except Exception as e:
                print(f"❌ Failed to emit progress: {e}")

        emit_progress('Starting generation...', 0)

        # Handle VAE decoder switching
        if use_taehv != current_use_taehv:
            emit_progress('Switching VAE decoder...', 2)
            print(f"🔄 Switching VAE decoder to {'TAEHV' if use_taehv else 'default VAE'}")
            current_vae_decoder = initialize_vae_decoder(use_taehv=use_taehv)
            # Update pipeline with new VAE decoder
            pipeline.vae = current_vae_decoder

        # Handle FP8 quantization
        if enable_fp8 and not fp8_applied:
            emit_progress('Applying FP8 quantization...', 3)
            print("🔧 Applying FP8 quantization to transformer")
            from torchao.quantization.quant_api import quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor
            quantize_(transformer, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
            fp8_applied = True

        # Text encoding
        emit_progress('Encoding text prompt...', 8)
        conditional_dict = text_encoder(text_prompts=[prompt])
        
        print(f"Prompt: {prompt}")
        print(f"Conditional dict keys: {list(conditional_dict.keys())}")
        for key, value in conditional_dict.items():
            conditional_dict[key] = value.to(dtype=torch.float16)
            print(f"\t {key} shape: {value.shape}, dtype: {value.dtype}")

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                text_encoder,target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # Handle torch.compile if enabled
        torch_compile_applied = enable_torch_compile
        if enable_torch_compile and not models_compiled:
            # Compile transformer and decoder
            transformer.compile(mode="max-autotune-no-cudagraphs")
            if not current_use_taehv and not low_memory and not args.trt:
                current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")

        # Initialize generation
        emit_progress('Initializing generation...', 12)

        print(f"Seed: {seed}")
        rnd = torch.Generator(gpu).manual_seed(seed)
        print(f"rnd: {rnd} (based on gpu {gpu})")
        # all_latents = torch.zeros([1, 21, 16, 60, 104], device=gpu, dtype=torch.bfloat16)

        pipeline._initialize_kv_cache(batch_size=1, dtype=torch.float16, device=gpu)
        pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.float16, device=gpu)

        # TODO: see if 21 = 3 * 7 (3 = pipeline.num_frame_per_block, 7 = num_blocks);
        #   if so, if we were to change total number of blocks, we need to change noise shape too

        noise = torch.randn([1, 21, 16, 60, 104], device=gpu, dtype=torch.float16, generator=rnd)

        # Generation parameters
        
        # TODO: see if num_blocks and pipeline.num_frame_per_block can be adjusted for different lengths
        
        num_blocks = 7
        current_start_frame = 0
        num_input_frames = 0
        all_num_frames = [pipeline.num_frame_per_block] * num_blocks
        
        print(f"Frames status: total {sum(all_num_frames)}, per block {all_num_frames}")
        
        if current_use_taehv:
            vae_cache = None
        else:
            vae_cache = ZERO_VAE_CACHE
            for i in range(len(vae_cache)):
                vae_cache[i] = vae_cache[i].to(device=gpu, dtype=torch.float16)

        total_frames_sent = 0
        generation_start_time = time.time()

        emit_progress('Generating frames... (frontend handles timing)', 15)

        profile = True
        # profile = False
        if profile:
            _start = torch.cuda.Event(enable_timing=True)
            _end = torch.cuda.Event(enable_timing=True)
            denoising_start = torch.cuda.Event(enable_timing=True)
            denoising_end = torch.cuda.Event(enable_timing=True)
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            transformer_start = torch.cuda.Event(enable_timing=True)
            transformer_end = torch.cuda.Event(enable_timing=True)
            decoding_start = torch.cuda.Event(enable_timing=True)
            decoding_end = torch.cuda.Event(enable_timing=True)


        if profile:
            _start.record()

        for idx, current_num_frames in enumerate(all_num_frames):
            
            print(f"In loop: idx {idx}, current_num_frames {current_num_frames}, current_start_frame {current_start_frame}")
            if not generation_active or stop_event.is_set():
                break

            progress = int(((idx + 1) / len(all_num_frames)) * 80) + 15

            # Special message for first block with torch.compile
            if idx == 0 and torch_compile_applied and not models_compiled:
                emit_progress(
                    f'Processing block 1/{len(all_num_frames)} - Compiling models (may take 5-10 minutes)...', progress)
                print(f"🔥 Processing block {idx+1}/{len(all_num_frames)}")
                models_compiled = True
            else:
                emit_progress(f'Processing block {idx+1}/{len(all_num_frames)}...', progress)
                print(f"🔄 Processing block {idx+1}/{len(all_num_frames)}")

            noisy_input = noise[:, current_start_frame - num_input_frames : current_start_frame + current_num_frames - num_input_frames]
            # print(f"noisy_input shape: {noisy_input.shape}, sharding: {current_start_frame - num_input_frames} to {current_start_frame + current_num_frames - num_input_frames}")
            # print(f"\t current_start_frame: {current_start_frame}, num_input_frames: {num_input_frames}, current_num_frames: {current_num_frames}")            

            # print(f"pipeline denoising steps: {pipeline.denoising_step_list}")

            if profile:
                block_start.record()
                denoising_start.record()

            for index, current_timestep in enumerate(pipeline.denoising_step_list):
                if not generation_active or stop_event.is_set():
                    break

                timestep = torch.ones([1, current_num_frames], device=noise.device,
                                      dtype=torch.int64) * current_timestep
                # print(f"\t timestep shape: {timestep.shape}, value: {timestep}")

                if index < len(pipeline.denoising_step_list) - 1:

                    if profile:
                        transformer_start.record()
                    
                    _, denoised_pred = transformer(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=pipeline.kv_cache1,
                        crossattn_cache=pipeline.crossattn_cache,
                        current_start=current_start_frame * pipeline.frame_seq_length
                    )

                    if profile:
                        transformer_end.record()
                        torch.cuda.synchronize()
                        transformer_time = transformer_start.elapsed_time(transformer_end)

                        print(f"\t Transformer denoising step {index+1}/{len(pipeline.denoising_step_list)} completed in {transformer_time:.2f} ms")
                    
                    next_timestep = pipeline.denoising_step_list[index + 1]
                    
                    noisy_input = pipeline.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([1 * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])

                else:
                    
                    if profile:
                        transformer_start.record()

                    _, denoised_pred = transformer(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=pipeline.kv_cache1,
                        crossattn_cache=pipeline.crossattn_cache,
                        current_start=current_start_frame * pipeline.frame_seq_length
                    )

                    if profile:
                        transformer_end.record()
                        torch.cuda.synchronize()
                        transformer_time = transformer_start.elapsed_time(transformer_end)

                        print(f"\t Transformer final denoising step {index+1}/{len(pipeline.denoising_step_list)} completed in {transformer_time:.2f} ms")

            if not generation_active or stop_event.is_set():
                break

            if profile:
                denoising_end.record()
                torch.cuda.Event()
                denoising_time = denoising_start.elapsed_time(denoising_end)

                print(f"⚡ Block {idx+1} denoising completed in {denoising_time:.2f} ms")

            # Record output
            # all_latents[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Update KV cache for next block
            if idx != len(all_num_frames) - 1:

                if profile:
                    transformer_start.record()

                transformer(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=torch.zeros_like(timestep),
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length,
                )

                if profile:
                    transformer_end.record()
                    torch.cuda.synchronize()
                    kv_update_time = transformer_start.elapsed_time(transformer_end)

                    print(f"KV cache update for next block completed in {kv_update_time:.2f} .s")
                    print(f"denoised_pred shape: {denoised_pred.shape}, dtype: {denoised_pred.dtype}")

            print(f"decoding args: args.trt {args.trt}, current_use_taehv {current_use_taehv}, vae_cache is None {vae_cache is None if not args.trt else 'N/A'}")

            # Decode to pixels and send frames immediately
            print(f"🎨 Decoding block {idx+1} to pixels...")
            
            if profile:
                decoding_start.record()

            if args.trt:
                all_current_pixels = []
                for i in range(denoised_pred.shape[1]):
                    is_first_frame = torch.tensor(1.0).cuda().half() if idx == 0 and i == 0 else \
                        torch.tensor(0.0).cuda().half()
                    outputs = vae_decoder.forward(denoised_pred[:, i:i + 1, :, :, :].half(), is_first_frame, *vae_cache)
                    # outputs = vae_decoder.forward(denoised_pred.float(), *vae_cache)
                    current_pixels, vae_cache = outputs[0], outputs[1:]
                    print(current_pixels.max(), current_pixels.min())
                    all_current_pixels.append(current_pixels.clone())
                pixels = torch.cat(all_current_pixels, dim=1)
                if idx == 0:
                    pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block
            else:
                if current_use_taehv:
                    if vae_cache is None:
                        vae_cache = denoised_pred
                    else:
                        denoised_pred = torch.cat([vae_cache, denoised_pred], dim=1)
                        vae_cache = denoised_pred[:, -3:, :, :, :]
                    pixels = current_vae_decoder.decode(denoised_pred)
                    print(f"denoised_pred shape: {denoised_pred.shape}")
                    print(f"pixels shape: {pixels.shape}")
                    if idx == 0:
                        pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block
                    else:
                        pixels = pixels[:, 12:, :, :, :]

                else:
                    """
                    rand pixels to see if vae is causing "cold start"
                    pixels = torch.rand(1, 12, 3, 480, 832)
                    """
                    pixels, vae_cache = current_vae_decoder(denoised_pred.half(), *vae_cache)
                    
                    # print(f"denoised_pred shape: {denoised_pred.shape}")
                    # print(f"pixels shape: {pixels.shape}")
                    # print(f"vae_cache length: {len(vae_cache)}")
                    # for vae_c in vae_cache:
                    #     print(f"\t vae_c shape: {vae_c.shape}")
                    if idx == 0:
                        pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block

            if profile:
                decoding_end.record()
                torch.cuda.synchronize()
                decoding_time = decoding_start.elapsed_time(decoding_end)
                print(f"🎨 Block {idx+1} VAE decoding completed in {decoding_time:.2f} ms")

            # Queue frames for non-blocking sending
            block_frames = pixels.shape[1]
            print(f"pixels shape after possible cropping: {pixels.shape}, block_frames: {block_frames}")

            print(f"📡 Queueing {block_frames} frames from block {idx+1} for sending...")
            queue_start = time.time()

            for frame_idx in range(block_frames):
                if not generation_active or stop_event.is_set():
                    break

                frame_tensor = pixels[0, frame_idx].cpu()
                # print(f"\t Queueing frame {frame_idx+1}/{block_frames} of block {idx+1}, shape: {frame_tensor.shape}, dtype: {frame_tensor.dtype}")

                # Queue frame data in non-blocking way
                frame_send_queue.put((frame_tensor, total_frames_sent, idx, job_id))
                total_frames_sent += 1

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
            else:
                block_time = "not profiling"

            queue_time = time.time() - queue_start
            print(f"✅ Block {idx+1} completed in {block_time:.2f} ms ({block_frames} frames queued in {queue_time:.3f} ms)")

            current_start_frame += current_num_frames

        generation_time = time.time() - generation_start_time
        print(f"🎉 Generation completed in {generation_time:.2f} ms! {total_frames_sent} frames queued for sending")

        if profile:
            _end.record()
            torch.cuda.synchronize()
            _time = _start.elapsed_time(_end)
            print(f"🎉 Generation completed in {_time:.2f} ms! (recorded with cuda event!)")


        # Wait for all frames to be sent before completing
        emit_progress('Waiting for all frames to be sent...', 97)
        print("⏳ Waiting for all frames to be sent...")
        frame_send_queue.join()  # Wait for all queued frames to be processed
        print("✅ All frames sent successfully!")

        generate_mp4_from_images("./images","./videos/"+anim_name+".mp4", frame_rate )
        # Final progress update
        emit_progress('Generation complete!', 100)

        try:
            socketio.emit('generation_complete', {
                'message': 'Video generation completed!',
                'total_frames': total_frames_sent,
                'generation_time': f"{generation_time:.2f}s",
                'job_id': job_id
            })
        except Exception as e:
            print(f"❌ Failed to emit generation complete: {e}")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        try:
            socketio.emit('error', {
                'message': f'Generation failed: {str(e)}',
                'job_id': job_id
            })
        except Exception as e:
            print(f"❌ Failed to emit error: {e}")
    finally:
        generation_active = False
        stop_event.set()

        # Clean up sender thread
        try:
            frame_send_queue.put(None)
        except Exception as e:
            print(f"❌ Failed to put None in frame_send_queue: {e}")


def generate_mp4_from_images(image_directory, output_video_path, fps=24):
    """
    Generate an MP4 video from a directory of images ordered alphabetically.

    :param image_directory: Path to the directory containing images.
    :param output_video_path: Path where the output MP4 will be saved.
    :param fps: Frames per second for the output video.
    """
    global anim_name
    # Construct the ffmpeg command
    cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', os.path.join(image_directory, anim_name+'/'+anim_name+'_%03d.jpg'),  # Adjust the pattern if necessary
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video_path
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Video saved to {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def calculate_sha256(data):
    # Convert data to bytes if it's not already
    if isinstance(data, str):
        data = data.encode()
    # Calculate SHA-256 hash
    sha256_hash = hashlib.sha256(data).hexdigest()
    return sha256_hash

# Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to frontend-buffered demo server'})


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


@socketio.on('start_generation')
def handle_start_generation(data):
    global generation_active, frame_number, anim_name, frame_rate

    frame_number = 0
    if generation_active:
        emit('error', {'message': 'Generation already in progress'})
        return

    prompt = data.get('prompt', '')

    seed = data.get('seed', -1)
    if seed==-1:
        seed = random.randint(0, 2**32)

    # Extract words up to the first punctuation or newline
    words_up_to_punctuation = re.split(r'[^\w\s]', prompt)[0].strip() if prompt else ''
    if not words_up_to_punctuation:
        words_up_to_punctuation = re.split(r'[\n\r]', prompt)[0].strip()

    # Calculate SHA-256 hash of the entire prompt
    sha256_hash = calculate_sha256(prompt)

    # Create anim_name with the extracted words and first 10 characters of the hash
    anim_name = f"{words_up_to_punctuation[:20]}_{str(seed)}_{sha256_hash[:10]}"

    generation_active = True
    generation_start_time = time.time()
    enable_torch_compile = data.get('enable_torch_compile', False)
    enable_fp8 = data.get('enable_fp8', False)
    use_taehv = data.get('use_taehv', False)
    frame_rate = data.get('fps', 6)

    if not prompt:
        emit('error', {'message': 'Prompt is required'})
        return

    # Start generation in background thread
    socketio.start_background_task(generate_video_stream___prof___, prompt, seed,
                                   enable_torch_compile, enable_fp8, use_taehv)
    emit('status', {'message': 'Generation started - frames will be sent immediately'})


@socketio.on('stop_generation')
def handle_stop_generation():
    global generation_active, stop_event, frame_send_queue
    generation_active = False
    stop_event.set()

    # Signal sender thread to stop (will be processed after current frames)
    try:
        frame_send_queue.put(None)
    except Exception as e:
        print(f"❌ Failed to put None in frame_send_queue: {e}")

    emit('status', {'message': 'Generation stopped'})

# Web routes


@app.route('/')
def index():
    return render_template('demo.html')


@app.route('/api/status')
def api_status():
    return jsonify({
        'generation_active': generation_active,
        'free_vram_gb': get_cuda_free_memory_gb(gpu),
        'fp8_applied': fp8_applied,
        'torch_compile_applied': torch_compile_applied,
        'current_use_taehv': current_use_taehv
    })


if __name__ == '__main__':
    print(f"🚀 Starting demo on http://{args.host}:{args.port}")
    socketio.run(app, host=args.host, port=args.port, debug=False)

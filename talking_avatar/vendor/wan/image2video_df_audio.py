# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from contextlib import contextmanager

import torch
from tqdm import tqdm

from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
import numpy as np
from einops import repeat, rearrange
import torch.nn.functional as F

# Audio-related constant
# WTVR = 640 * 2  # Audio-to-video frame conversion ratio #  for fps 12.5
WTVR = 640 # for fps 25


def calculate_chunk_count(video_frame_total_length, base_num_frames, overlap_history):
    if video_frame_total_length <= base_num_frames:
        return 1
    return math.ceil(
        (video_frame_total_length - base_num_frames)
        / (base_num_frames - overlap_history)
    ) + 1

class WanTI2V:

    def __init__(
        self,
        transformer,
        vae,
        device,
        weight_dtype,
        config,
        audio_encoder=None,
        audio_weighted_sum=None,
        audio_projection=None,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
            audio_encoder: Audio encoding model (optional)
            audio_weighted_sum: Weighted sum module for audio embeddings (optional)
            audio_projection: Projection layer for audio embeddings (optional)
        """
        self.vae = vae
        self.vae_stride = (4, 16, 16)
        self.patch_size = (1, 2, 2)
        self.model = transformer
        self.sp_size = 1
        self.num_train_timesteps = 1000
        self.param_dtype = weight_dtype
        self.device = device
        self.config = config

        # Audio-related components
        self.audio_encoder = audio_encoder
        self.audio_weighted_sum = audio_weighted_sum
        self.audio_projection = audio_projection

    def generate(self,
            img: torch.Tensor,
            prompt_embeds: torch.Tensor,
            negative_prompt_embeds: torch.Tensor,
            max_area=704 * 1280,
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=40,
            guide_scale=5.0,
            seed=-1,
            base_num_frames=45,
            overlap_history=9,
            addnoise_condition=20,
            ref_copy_motion_idx=0,
            ref_latent_attn=False,
            audio=None,
            ref_lmk=None,
            audio_guidance_scale=5.0,
            end_latent_input=False,
            end_frame_idx=None,
            empty_start_frame=False,
            half_end_latent_input=False,
            prefix_frame=1,
            audio_mask=None,
        ):
        r"""
        Generate video frames from an input image and text prompt using a diffusion process.

        Args:
            img (torch.Tensor):
                Input image tensor of shape [3, H, W].
            prompt_embeds (torch.Tensor):
                Prompt embedding tensor.
            negative_prompt_embeds (torch.Tensor):
                Negative prompt embedding tensor.
            max_area (int, optional):
                Maximum pixel area for latent space calculation. Controls video resolution scaling. Default: 704*1280.
            frame_num (int, optional):
                Number of frames to generate. Should be 4n+1. Default: 121.
            shift (float, optional):
                Noise schedule shift parameter affecting temporal dynamics. Default: 5.0.
            sample_solver (str, optional):
                Solver used to sample the video. Default: 'unipc'.
            sampling_steps (int, optional):
                Number of diffusion sampling steps. Higher values improve quality but slow generation. Default: 40.
            guide_scale (float, optional):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity. Default: 5.0.
            seed (int, optional):
                Random seed for noise generation. If -1, a random seed is used. Default: -1.
            base_num_frames (int, optional):
                Number of frames to generate. Should be 4n+1. Default: 45.
            overlap_history (int, optional):
                Number of frames to overlap. Default: 9.
            addnoise_condition (int, optional):
                Add noise condition. Default: 20.
            audio (torch.Tensor, optional):
                Audio input tensor of shape (B, T) where T is audio length. Default: None.
            ref_lmk (torch.Tensor, optional):
                Reference landmarks of shape (B, 159, 2) for facial landmarks. Default: None.
            audio_guidance_scale (float, optional):
                Classifier-free guidance scale for audio conditioning. Default: 5.0.

        Returns:
            torch.Tensor:
                Generated video frames tensor of shape (C, N, H, W), where:
                    - C: Number of color channels (3 for RGB)
                    - N: Number of frames (frame_num)
                    - H: Frame height (determined by input and scaling)
                    - W: Frame width (determined by input and scaling)
        """

        return self.i2v(
            img,
            prompt_embeds,
            negative_prompt_embeds,
            max_area,
            frame_num,
            shift,
            sample_solver,
            sampling_steps,
            guide_scale,
            seed,
            base_num_frames,
            overlap_history,
            addnoise_condition,
            ref_copy_motion_idx,
            ref_latent_attn,
            audio,
            ref_lmk,
            audio_guidance_scale,
            end_latent_input,
            end_frame_idx,
            empty_start_frame,
            half_end_latent_input,
            prefix_frame,
            audio_mask,
            )


    def i2v(self,
            img: torch.Tensor,
            prompt_embeds: torch.Tensor,
            negative_prompt_embeds: torch.Tensor,
            max_area=704 * 1280,
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=40,
            guide_scale=5.0,
            seed=-1,
            base_num_frames=45,
            overlap_history=9,
            addnoise_condition=20,
            ref_copy_motion_idx=0,
            ref_latent_attn=False,
            audio=None,
            ref_lmk=None,
            audio_guidance_scale=5.0,
            end_latent_input=False,
            end_frame_idx=None,
            empty_start_frame=False,
            half_end_latent_input=False,
            prefix_frame=1,
            audio_mask=None,
            ):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            img (torch.Tensor):
                Input image tensor. Shape: [3, H, W]
            prompt_embeds (torch.Tensor):
                Prompt embedding tensor. Shape: [1, 512, 4096]
            negative_prompt_embeds (torch.Tensor):
                Negative prompt embedding tensor. Shape: [1, 512, 4096]
            max_area (int):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (int):
                How many frames to sample from a video. The number should be 4n+1
            shift (float):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (str):
                Solver used to sample the video.
            sampling_steps (int):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (float):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            seed (int):
                Random seed for noise generation. If -1, use random seed
            base_num_frames (int):
                Number of frames to generate. Should be 4n+1. Default: 45.
            overlap_history (int):
                Number of frames to overlap. Default: 9.
            addnoise_condition (int):
                Add noise condition. Default: 20.
            ref_copy_motion_idx (int, optional):
                Number of frames to copy motion from the reference image. Default: 0.
            audio (torch.Tensor, optional):
                Audio input tensor of shape (B, T) where T is audio length. Default: None.
            ref_lmk (torch.Tensor, optional):
                Reference landmarks of shape (B, 159, 2) for facial landmarks. Default: None.
            audio_guidance_scale (float, optional):
                Classifier-free guidance scale for audio conditioning. Default: 5.0.
            Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (121)
                - H: Frame height (from max_area)
                - W: Frame width (from max_area)
        """
        context = prompt_embeds
        context_null = negative_prompt_embeds

        if len(img.shape) == 4:
            img = img.unsqueeze(2) # [B, 3, 1, H, W]
            assert img.shape[0] == 1
            img = img[0]  # [3, 1, H, W]
        else:
            assert img.shape[0] == 3
            img = img.unsqueeze(1) # [3, 1, H, W]

        # Repeat ref image along temporal axis for prefix latent (ref_end_mode="replace")
        if prefix_frame > 1:
            img_for_encode = img.repeat(1, prefix_frame, 1, 1)  # [3, prefix_frame, H, W]
        else:
            img_for_encode = img
        z = self.vae.encode([img_for_encode])  # [C, T_latent, H, W]

        if end_latent_input:
            end_latent = [z[0].clone()]
            assert end_frame_idx is not None, "end_frame_idx is required when end_latent_input is True"
            end_frame_idx = torch.tensor([end_frame_idx], device=self.device, dtype=torch.long)
        else:
            end_latent = None
            end_frame_idx = None

        prefix_latent = [z[0].clone()]
        oh = img.shape[-2]
        ow = img.shape[-1]

        # Calculate latent dimensions
        latent_height = oh // self.vae_stride[1]
        latent_width = ow // self.vae_stride[2]

        Fr = frame_num


        # video level frame length
        if audio is not None and self.audio_encoder is not None:
            audio = audio.to(prompt_embeds.device).to(self.param_dtype)
            video_frame_valid_length = math.ceil(audio.shape[1] / WTVR)
        else:
            video_frame_valid_length = Fr

        if (video_frame_valid_length - base_num_frames) % (base_num_frames - overlap_history) != 0:
            video_frame_padding_length = (base_num_frames - overlap_history) - (video_frame_valid_length - base_num_frames) % (base_num_frames - overlap_history)
        else:
            video_frame_padding_length = 0

        video_frame_total_length = video_frame_valid_length + video_frame_padding_length
        # assert ((video_frame_total_length-1) % 4 == 0), "video_frame_total_length should be divisible by 4 since I add the padding"

        n_iter = calculate_chunk_count(
            video_frame_total_length, base_num_frames, overlap_history
        )
        latent_frame_total_length = (video_frame_total_length - 1) // self.vae_stride[0] + 1
        print(f"n_iter:{n_iter}, video_frame _total_length:{video_frame_total_length}, latent_frame_total_length:{latent_frame_total_length}, video_frame_padding_length: {video_frame_padding_length}")

        if audio is not None:
            print(f"audio length:{audio.shape[1]}, video_frame_valid_length:{video_frame_valid_length}")

        output_video = None

        video_frame_start_index = 0
        prefix_latent_frame_length = z[0].shape[1] if not empty_start_frame else 0  # actual latent frames after VAE encode
        current_overlap_history = 0

        # Prepare the spatial audio-conditioning mask.
        full_face_mask = bool(self.config.get("full_face_mask", False))
        infer_mask_border_ratio = float(self.config.get("mask_border_ratio", 0.1))
        if audio_mask is not None:
            if audio_mask.ndim != 4 or audio_mask.shape[1] != 1:
                raise ValueError("audio_mask must have shape [B, 1, H, W]")
            target_size = (latent_height // 2, latent_width // 2)
            audio_attn_output_mask_ref = (
                F.interpolate(
                    audio_mask.to(prompt_embeds.device, dtype=torch.float32).clamp(0, 1),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                .unsqueeze(2)
                .to(dtype=self.param_dtype)
            )
        elif ref_lmk is not None and audio is not None and not full_face_mask:
            audio_attn_output_mask_ref = make_face_mask_from_lmk_4dim(
                ref_lmk.unsqueeze(1),
                torch.randn([img.shape[0] if len(img.shape) == 4 else 1, 16, 1, latent_height // 2 , latent_width // 2]),
                mask_border_ratio=infer_mask_border_ratio,
                image_size_height=oh,
                image_size_width=ow
            ).to(prompt_embeds.device, dtype=self.param_dtype)
        else:
            mask_bsz = ref_lmk.shape[0] if ref_lmk is not None else 1
            audio_attn_output_mask_ref = torch.ones(
                [mask_bsz, 1, 1, latent_height // 2, latent_width // 2],
                device=prompt_embeds.device, dtype=self.param_dtype)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):
            for chunk_idx in range(n_iter):
                right_padding_length = 0

                if output_video is not None:  # i !=0
                    prefix_video = output_video[:, -overlap_history:].to(prompt_embeds.device)
                    if ref_copy_motion_idx != 0:
                        prefix_video[:, :ref_copy_motion_idx] = img.clone().repeat(1, ref_copy_motion_idx, 1, 1)
                    prefix_latent = [self.vae.encode(prefix_video.unsqueeze(0))[0]]  # [(c, f, h, w)]
                    current_overlap_history = overlap_history
                    prefix_latent_frame_length = (current_overlap_history - 1) // self.vae_stride[0] + 1

                video_frame_end_index = min(video_frame_start_index + base_num_frames - current_overlap_history,
                                            video_frame_total_length)
                video_frame_gen_length = video_frame_end_index - video_frame_start_index + current_overlap_history

                latent_frame_cur_iter_length = (video_frame_gen_length - 1) // self.vae_stride[0] + 1
                latent_shape = [self.vae.model.z_dim, latent_frame_cur_iter_length, oh // self.vae_stride[1], ow // self.vae_stride[2]]

                noise = [torch.randn(
                    latent_shape, dtype=self.param_dtype, device=prompt_embeds.device)] # [1, 48, 10, 10]
                seq_len = latent_shape[1] * latent_shape[2] * latent_shape[3] // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])

                if chunk_idx > 0:
                    clean_latent_frame_num = (ref_copy_motion_idx - 1) // self.vae_stride[0] + 1 if ref_copy_motion_idx != 0 else 0
                    prefix_latent[0][:, clean_latent_frame_num:prefix_latent_frame_length, :, :] = (1 - 0.001*addnoise_condition) * prefix_latent[0][:, clean_latent_frame_num:prefix_latent_frame_length, :, :] + 0.001*addnoise_condition * noise[0][:, clean_latent_frame_num:prefix_latent_frame_length, :, :]

                if sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                elif sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                else:
                    raise NotImplementedError("Unsupported solver.")

                # sample videos

                latent = noise[0]
                if prefix_latent_frame_length > 0:
                    latent[:, :prefix_latent_frame_length, :, :] = prefix_latent[0]
                prefix_seq_len = prefix_latent[0].shape[1] * prefix_latent[0].shape[2] * prefix_latent[0].shape[3] // (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])

                is_end_latent_input = bool(chunk_idx % 2 == 1) if half_end_latent_input else True

                # Process audio for this chunk if available
                audio_embeds = None
                audio_attn_mask = None
                if self.config.network.get("audio_out_attn_mask", False):
                    audio_attn_output_mask = audio_attn_output_mask_ref
                    audio_attn_output_mask = repeat(audio_attn_output_mask, "b 1 1 H W -> b 1 Lv H W", Lv = latent_frame_cur_iter_length).squeeze(1).unsqueeze(-1)
                    audio_attn_output_mask = rearrange(audio_attn_output_mask, "b Lv H W 1  -> b (Lv H W) 1 ")
                else:
                    audio_attn_output_mask = None

                if audio is not None and self.audio_encoder is not None:
                    current_audio = audio[:, video_frame_start_index * WTVR: video_frame_end_index * WTVR]

                    # Left padding
                    if current_overlap_history > 0:
                        if video_frame_start_index == 0:
                            current_audio = F.pad(current_audio, (current_overlap_history * WTVR, 0))
                        else:
                            previous_audio = audio[:, (video_frame_start_index-current_overlap_history)*WTVR:
                                                  video_frame_start_index*WTVR].to(prompt_embeds.device).to(self.param_dtype)
                            current_audio = torch.cat([previous_audio, current_audio], dim=1)

                    # Right padding
                    if current_audio.shape[1] < video_frame_gen_length * WTVR:
                        right_padding_length = (video_frame_gen_length * WTVR - current_audio.shape[1]) // WTVR
                        current_audio = F.pad(current_audio, (0, video_frame_gen_length * WTVR - current_audio.shape[1]))

                    # Encode audio
                    if self.config.network.get("audio_last_layer", False):
                        audio_embeds = self.audio_encoder(current_audio, output_len=video_frame_gen_length * 2).last_hidden_state
                    else:
                        if WTVR == 640: ## fps 25
                            audio_embeds = self.audio_encoder(current_audio, output_len=video_frame_gen_length).hidden_states
                        elif WTVR == 640 * 2: ## fps 12.5
                            audio_embeds = self.audio_encoder(current_audio, output_len=video_frame_gen_length * 2).hidden_states
                        else:
                            raise ValueError(f"WTVR must be 640 or 640 * 2, but got {WTVR}")
                        audio_embeds = torch.stack(audio_embeds, dim=2)

                    # Apply weighted sum and projection if available
                    audio_embeds = self.audio_weighted_sum(audio_embeds) if self.audio_weighted_sum is not None else audio_embeds
                    audio_embeds = self.audio_projection(audio_embeds) if self.audio_projection is not None else audio_embeds
                    audio_proj_length = audio_embeds.shape[1]

                    # Split audio sequence to match latent frames
                    if self.config.network.get("audio_temp_interpolate"):
                        pos_idx_ranges = self.split_interpolated_audio_sequence(audio_proj_length, latent_num_frames=latent_frame_cur_iter_length)
                    else:
                        # Fallback to simple splitting
                        pos_idx_ranges = self.split_audio_sequence(audio_proj_length, latent_frame_cur_iter_length)

                    if audio_embeds.ndim == 4:
                        ln = audio_embeds.shape[2]  # num of audio layers
                        audio_embeds = rearrange(audio_embeds, 'b la ln d -> (b ln) la d')
                        audio_embeds, k_lens = self.split_tensor_with_padding(audio_embeds, pos_idx_ranges)
                        audio_embeds = rearrange(audio_embeds, '(b ln) lv la d -> b lv (ln la) d', ln=ln)
                    else:
                        audio_embeds, k_lens = self.split_tensor_with_padding(audio_embeds, pos_idx_ranges)


                arg_c = {
                    'context': [context[0]],
                    'seq_len': seq_len,
                    'ref_latents': z[0].unsqueeze(0).clone() if ref_latent_attn else None,
                    'cond_pure_norm': self.config.network.backbone.get("cond_pure_norm", False),
                    'end_latents': end_latent[0].unsqueeze(0).clone() if end_latent is not None and is_end_latent_input else None,
                    'end_frame_idx': end_frame_idx if end_frame_idx is not None and is_end_latent_input else None,
                }

                # Add audio embeddings if available
                if audio_embeds is not None:
                    arg_c['audio_embeds'] = audio_embeds
                    arg_c['audio_attn_mask'] = audio_attn_mask
                    arg_c['audio_attn_output_mask'] = audio_attn_output_mask

                arg_null = {
                    'context': context_null,
                    'seq_len': seq_len,
                    'ref_latents': z[0].unsqueeze(0).clone() if ref_latent_attn else None,
                    'cond_pure_norm': self.config.network.backbone.get("cond_pure_norm", False),
                    'end_latents': end_latent[0].unsqueeze(0).clone() if end_latent is not None and is_end_latent_input else None,
                    'end_frame_idx': end_frame_idx if end_frame_idx is not None and is_end_latent_input else None,
                }

                # # Add audio for uncond as well if doing audio guidance
                # if audio_embeds is not None and guide_scale > 1:
                #     arg_null['audio_embeds'] = audio_embeds
                #     arg_null['audio_attn_mask'] = audio_attn_mask
                #     arg_null['audio_attn_output_mask'] = audio_attn_output_mask

                for _, t in enumerate(tqdm(timesteps)):
                    latent_model_input = [latent.to(self.device)]
                    timestep = [t] # B

                    timestep = torch.stack(timestep).to(self.device)

                    # Refactored: combine repeat and clone for clarity and maintainability
                    timestep = repeat(
                        timestep,
                        'b -> b f h w',
                        f=latent_shape[1] // self.patch_size[0],
                        h=latent_shape[2] // self.patch_size[1],
                        w=latent_shape[3] // self.patch_size[2]
                    ).clone()  # Ensure new memory allocation to avoid shared memory issues

                    # 명시적으로 텐서를 복사하여 할당 문제 방지
                    if chunk_idx == 0:
                        if prefix_latent_frame_length > 0:
                            timestep[:, :prefix_latent_frame_length, :, :] = torch.zeros_like(timestep[:, :prefix_latent_frame_length, :, :])
                    else:
                        if clean_latent_frame_num > 0:
                            timestep[:, :clean_latent_frame_num, :, :] = torch.zeros_like(timestep[:, :clean_latent_frame_num, :, :])
                        timestep[:, clean_latent_frame_num:prefix_latent_frame_length, :, :] = torch.full_like(timestep[:, clean_latent_frame_num:prefix_latent_frame_length, :, :], addnoise_condition)



                    timestep = timestep.flatten(1)

                    # Handle classifier-free guidance with audio
                    if guide_scale > 1:
                        if audio_embeds is not None and audio_guidance_scale > 0:
                            # Three-way CFG: uncond, audio_uncond, full_cond

                            # Full conditional (text + audio)
                            noise_pred_cond = self.model(
                                latent_model_input, t=timestep, **arg_c)[0]

                            # Set audio scale to 0 for audio_uncond pass if model supports it
                            if hasattr(self.model, 'blocks'):
                                for block in self.model.blocks:
                                    if hasattr(block, 'cross_attn') and hasattr(block.cross_attn, 'audio_scale'):
                                        block.cross_attn.audio_scale = 0.0
                                    if hasattr(block, 'cross_attn_audio') and hasattr(block.cross_attn_audio, 'audio_scale'):
                                        block.cross_attn_audio.audio_scale = 0.0

                            # Unconditional (negative prompt)
                            noise_pred_uncond = self.model(
                                latent_model_input, t=timestep, **arg_null)[0]

                            # Audio unconditional (text only, no audio)
                            arg_audio_uncond = arg_c.copy()
                            noise_pred_audio_uncond = self.model(
                                latent_model_input, t=timestep, **arg_audio_uncond)[0]

                            # Reset audio scale
                            if hasattr(self.model, 'blocks'):
                                for block in self.model.blocks:
                                    if hasattr(block, 'cross_attn') and hasattr(block.cross_attn, 'audio_scale'):
                                        block.cross_attn.audio_scale = 1.0
                                    if hasattr(block, 'cross_attn_audio') and hasattr(block.cross_attn_audio, 'audio_scale'):
                                        block.cross_attn_audio.audio_scale = 1.0

                            # Three-way combination
                            noise_pred = noise_pred_uncond + \
                                        guide_scale * (noise_pred_audio_uncond - noise_pred_uncond) + \
                                        audio_guidance_scale * (noise_pred_cond - noise_pred_audio_uncond)
                        else:

                            if hasattr(self.model, 'blocks'):
                                for block in self.model.blocks:
                                    if hasattr(block, 'cross_attn') and hasattr(block.cross_attn, 'audio_scale'):
                                        block.cross_attn.audio_scale = 0.0
                                    if hasattr(block, 'cross_attn_audio') and hasattr(block.cross_attn_audio, 'audio_scale'):
                                        block.cross_attn_audio.audio_scale = 0.0

                            # Standard two-way CFG
                            noise_pred_cond = self.model(
                                latent_model_input, t=timestep, **arg_c)[0]

                            noise_pred_uncond = self.model(
                                latent_model_input, t=timestep, **arg_null)[0]

                            noise_pred = noise_pred_uncond + guide_scale * (
                                noise_pred_cond - noise_pred_uncond)
                    else:
                        # No CFG
                        noise_pred = self.model(
                            latent_model_input, t=timestep, **arg_c)[0]

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latent.unsqueeze(0),
                        return_dict=False,
                        )[0]
                    latent = temp_x0.squeeze(0)
                    if prefix_latent_frame_length > 0:
                        latent[:, :prefix_latent_frame_length, :, :] = prefix_latent[0]

                    x0 = [latent]
                    del latent_model_input, timestep



                videos = self.vae.decode(x0)
                if chunk_idx == 0:
                    gen_video = videos[0].clamp(-1, 1).cpu()
                    # gen_video[:, 0, :20, :20] = 0 # HACK!
                    output_video = gen_video
                else:
                    gen_video = videos[0][:, current_overlap_history:
                                             video_frame_gen_length-right_padding_length].clamp(-1, 1).cpu()
                    # gen_video[:, 0, :20, :20] = 0 # HACK!

                    output_video = torch.cat(
                        [output_video, gen_video], 1 # 끝에 padding 있다면 이 부분도 추가, video_frame_gen_length
                    )  # c, f, h, w
                video_frame_start_index = video_frame_end_index ## Don't forget to update this

        del noise, latent, x0
        del sample_scheduler
        output_video = [(output_video / 2 + 0.5).clamp(0, 1)]
        output_video = [video.permute(1, 2, 3, 0) * 255 for video in output_video] # [F, H, W, 3]
        output_video = [video.cpu().numpy().astype(np.uint8) for video in output_video]
        return output_video

    def split_interpolated_audio_sequence(self, audio_embeds_length, latent_num_frames=13, audio_token_per_latent_frame=8):
        """Split audio sequence into latent frame slices."""
        audio_token_per_latent_frame = self.config.network.get("audio_token_per_latent_frame", audio_token_per_latent_frame)
        s = audio_embeds_length - (latent_num_frames-1)*audio_token_per_latent_frame
        assert s > 0, f"Not enough audio! audio_len={audio_embeds_length}, latents={latent_num_frames}, stride={audio_token_per_latent_frame}, first_tokens={s}"
        return [[s-audio_token_per_latent_frame, s-1]] + [[s+(i-1)*audio_token_per_latent_frame, s+i*audio_token_per_latent_frame-1] for i in range(1, latent_num_frames)]

    def split_audio_sequence(self, audio_length, latent_frames):
        """Simple audio sequence splitting for non-interpolated case."""
        # Simple uniform splitting
        tokens_per_frame = max(1, audio_length // latent_frames)
        ranges = []
        for i in range(latent_frames):
            start = i * tokens_per_frame
            end = min((i + 1) * tokens_per_frame - 1, audio_length - 1)
            ranges.append([start, end])
        return ranges

    def split_tensor_with_padding(self, input_tensor, pos_idx_ranges, expand_length=0):
        """
        Split `input_tensor` (shape [1, L, D]) into padded subsequences.

        Returns:
        sub_sequences: [1, F, L_sub, D]  – padded subsequences
        k_lens      : [F]               – true (unpadded) lengths
        """
        seq_len = input_tensor.size(1)
        max_valid_idx = seq_len - 1
        sub_seqs, klen = [], []

        for start, end in ((s - expand_length, e + expand_length) for s, e in pos_idx_ranges):
            pad_front, pad_back = max(-start, 0), max(end - max_valid_idx, 0)

            valid = input_tensor[:, max(start, 0): min(end + 1, seq_len), :]  # slice in-bounds
            klen.append(valid.size(1))                                         # unpadded length

            # (0,0) ➜ embed-dim; (pad_front, pad_back) ➜ seq-dim
            sub_seqs.append(F.pad(valid, (0, 0, pad_front, pad_back)))

        return torch.stack(sub_seqs, dim=1), torch.tensor(klen, dtype=torch.long)


# Standalone helper function for face mask generation
def make_face_mask_from_lmk_4dim(lmk_4dim, target_latents, mask_border_ratio=0.1, image_size_height=256, image_size_width=256):
    """Create a face mask from facial landmarks for each batch.

    Args:
        lmk_4dim (torch.Tensor): Facial landmarks tensor of shape [B, T, 159, 2] where B is batch size,
            T is number of frames, and each landmark has 2D coordinates.
        target_latents (torch.Tensor): Target latent tensor of shape [B, C, T//4, H//8, W//8].
        mask_border_ratio (float, optional): Ratio to expand mask border. Defaults to 0.1.
        image_size_height (int, optional): Original image height. Defaults to 256.
        image_size_width (int, optional): Original image width. Defaults to 256.

    Returns:
        torch.Tensor: Face mask tensor of shape [B, 1, T_latents, H//8, W//8] where 1 indicates face region.
    """
    B, T, C, D = lmk_4dim.shape
    height_latents_ratio = image_size_height / target_latents.shape[-2]
    width_latents_ratio = image_size_width / target_latents.shape[-1]

    face_masks = []

    def f(x, y, mask_border_ratio=0.1):
        h, w = x.shape[-2:]
        height_border = (y[2] - y[0]) * mask_border_ratio
        width_border = (y[3] - y[1]) * mask_border_ratio
        y[0] = max(y[0] - height_border, 0)
        y[2] = min(y[2] + height_border, h)

        y[1] = max(y[1] - width_border, 0)
        y[3] = min(y[3] + width_border, w)

        x[:, y[0]:y[2], y[1]:y[3]] = 1
        return x

    for lmk in lmk_4dim:
        min_y = torch.min(lmk[..., 1]) / height_latents_ratio
        min_x = torch.min(lmk[..., 0]) / width_latents_ratio
        max_y = torch.max(lmk[..., 1]) / height_latents_ratio
        max_x = torch.max(lmk[..., 0]) / width_latents_ratio

        left_upper_box_coords = torch.tensor([min_y, min_x], device=lmk.device).to(torch.long)
        right_lower_box_coords = torch.tensor([max_y, max_x], device=lmk.device).to(torch.long)
        bbox = torch.cat([left_upper_box_coords, right_lower_box_coords], dim=-1)

        face_mask_shape = list(target_latents.shape[-2:])
        T_latents = target_latents.shape[2]
        face_mask = torch.zeros((1, face_mask_shape[0], face_mask_shape[1]), device=target_latents.device)
        face_mask = f(face_mask, bbox, mask_border_ratio).detach()

        face_mask = face_mask.unsqueeze(0).repeat(1, T_latents, 1, 1)

        face_masks.append(face_mask)

    face_mask = torch.stack(face_masks, dim=0)
    return face_mask

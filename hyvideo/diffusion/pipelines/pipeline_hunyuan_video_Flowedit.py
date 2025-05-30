# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
from einops import rearrange
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from packaging import version

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from ...constants import PRECISION_TO_TYPE
from ...vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from ...text_encoder import TextEncoder
from ...modules import HYVideoDiffusionTransformer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class HunyuanVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using HunyuanVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`TextEncoder`]):
            Frozen text-encoder.
        text_encoder_2 ([`TextEncoder`]):
            Frozen text-encoder_2.
        transformer ([`HYVideoDiffusionTransformer`]):
            A `HYVideoDiffusionTransformer` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["text_encoder_2"]
    _exclude_from_cpu_offload = ["transformer"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: HYVideoDiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        scheduler_reverse: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
        progress_bar_config: Dict[str, Any] = None,
        args=None,
    ):
        super().__init__()

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        self.args = args
        # ==========================================================================================

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if (
            hasattr(scheduler.config, "clip_sample")
            and scheduler.config.clip_sample is True
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)


        if (
            hasattr(scheduler_reverse.config, "steps_offset")
            and scheduler_reverse.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler_reverse: {scheduler_reverse} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler_reverse.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler_reverse/scheduler_reverse_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler_reverse.config)
            new_config["steps_offset"] = 1
            scheduler_reverse._internal_dict = FrozenDict(new_config)

        if (
            hasattr(scheduler_reverse.config, "clip_sample")
            and scheduler_reverse.config.clip_sample is True
        ):
            deprecation_message = (
                f"The configuration file of this scheduler_reverse: {scheduler_reverse} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler_reverse/scheduler_reverse_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler_reverse.config)
            new_config["clip_sample"] = False
            scheduler_reverse._internal_dict = FrozenDict(new_config)


        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            scheduler_reverse=scheduler_reverse,
            text_encoder_2=text_encoder_2,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        # return self._guidance_scale > 1 and self.transformer.config.time_cond_proj_dim is None
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`torch.Tensor`, *optional*):
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
            data_type (`str`, *optional*):
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(text_encoder.model, lora_scale)
            else:
                scale_lora_layers(text_encoder.model, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, text_encoder.tokenizer)

            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)
            

            if clip_skip is None:
                print(device)
                prompt_outputs = text_encoder.encode(
                    text_inputs, data_type=data_type, device=device
                )
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(
                    text_inputs,
                    output_hidden_states=True,
                    data_type=data_type,
                    device=device,
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                    prompt_embeds
                )

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(
                    uncond_tokens, text_encoder.tokenizer
                )

            # max_length = prompt_embeds.shape[1]
            uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type)

            negative_prompt_outputs = text_encoder.encode(
                uncond_input, data_type=data_type, device=device
            )
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(
                    1, num_videos_per_prompt
                )
                negative_attention_mask = negative_attention_mask.view(
                    batch_size * num_videos_per_prompt, seq_len
                )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, -1
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt, 1
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, seq_len, -1
                )

        if text_encoder is not None:
            if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(text_encoder.model, lora_scale)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )

    def decode_latents(self, latents, enable_tiling=True):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        if enable_tiling:
            self.vae.enable_tiling()
            image = self.vae.decode(latents, return_dict=False)[0]
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        if image.ndim == 4:
            image = image.cpu().permute(0, 2, 3, 1).float()
        else:
            image = image.cpu().float()
        return image

    def prepare_extra_func_kwargs(self, func, kwargs):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        video_length,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        vae_ver="88-4c-sd",
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if video_length is not None:
            if "884" in vae_ver:
                if video_length != 1 and (video_length - 1) % 4 != 0:
                    raise ValueError(
                        f"`video_length` has to be 1 or a multiple of 4 but is {video_length}."
                    )
            elif "888" in vae_ver:
                if video_length != 1 and (video_length - 1) % 8 != 0:
                    raise ValueError(
                        f"`video_length` has to be 1 or a multiple of 8 but is {video_length}."
                    )

        if callback_steps is not None and (
            not isinstance(callback_steps, int) or callback_steps <= 0
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )


    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            print("Using inversed latents")
            latents = latents.to(device)

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self,
        w: torch.Tensor,
        embedding_dim: int = 512,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb



    @torch.no_grad()

    def inverse(
        self,
        latents: torch.Tensor,
        source_prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        num_videos_per_prompt: Optional[int] = 1,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        **kwargs,
    ):
        
        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not self.args.disable_autocast
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not self.args.disable_autocast



        # 2. Prepare the source prompt embeddings

        device = torch.device(f"cuda:{dist.get_rank()}") if dist.is_initialized() else self._execution_device
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = self.encode_prompt(
            source_prompt,
            device,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_attention_mask=negative_attention_mask,
            lora_scale=self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None,
            clip_skip=self.clip_skip,
            data_type="video",
        )
        if self.text_encoder_2 is not None:
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_mask_2,
                negative_prompt_mask_2,
            ) = self.encode_prompt(
                source_prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                attention_mask=None,
                negative_prompt_embeds=None,
                negative_attention_mask=None,
                lora_scale=self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2,
                data_type="video",
            )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_mask_2 = None
            negative_prompt_mask_2 = None

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        # 3. Perform inverse denoising loop
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler_reverse,
            num_inference_steps,
            device,
            **self.prepare_extra_func_kwargs(self.scheduler_reverse.set_timesteps, {"n_tokens": n_tokens}),
        )
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=latents.shape[0],
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            video_length=video_length,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 4. Denoising loop
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler_reverse.step,
            {"generator": generator, "eta": eta},
        )



        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler_reverse.order
        self._num_timesteps = len(timesteps)
        
        # 反转 timesteps
        # timesteps = torch.flip(timesteps, dims=[0])

        for i, t in enumerate(timesteps):
            # 打印 i 和 t
            print(f"Step {i}: t = {t}")        

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                print(f"Step {i}: t = {t}")
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler_reverse.scale_model_input(
                    latent_model_input, t
                )

                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(target_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                with torch.autocast(
                    device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
                ):
                    noise_pred = self.transformer(
                        latent_model_input,
                        t_expand,
                        text_states=prompt_embeds,
                        text_mask=prompt_mask,
                        text_states_2=prompt_embeds_2,
                        freqs_cos=freqs_cis[0],
                        freqs_sin=freqs_cis[1],
                        guidance=guidance_expand,
                        return_dict=True,
                    )["x"]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                latents = self.scheduler_reverse.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler_reverse.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()

        #self.transformer.to("cpu")
        return latents

    @torch.no_grad()
    def reverse(
        self,
        latents: torch.Tensor,
        target_prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        num_videos_per_prompt: Optional[int] = 1,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        **kwargs,
    ):
        # 1. Prepare the target prompt embeddings
        device = torch.device(f"cuda:{dist.get_rank()}") if dist.is_initialized() else self._execution_device
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = self.encode_prompt(
            target_prompt,
            device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_attention_mask=negative_attention_mask,
            lora_scale=self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None,
            clip_skip=self.clip_skip,
            data_type="video",
        )
        if self.text_encoder_2 is not None:
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_mask_2,
                negative_prompt_mask_2,
            ) = self.encode_prompt(
                target_prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                attention_mask=None,
                negative_prompt_embeds=None,
                negative_attention_mask=None,
                lora_scale=self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2,
                data_type="video",
            )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_mask_2 = None
            negative_prompt_mask_2 = None

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])




        # 2. Perform reverse denoising loop
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            **self.prepare_extra_func_kwargs(self.scheduler.set_timesteps, {"n_tokens": n_tokens}),
        )
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=latents.shape[0],
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            video_length=video_length,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 3. Denoising loop
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": generator, "eta": eta},
        )
        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not self.args.disable_autocast
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not self.args.disable_autocast

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)


        for i, t in enumerate(timesteps):
            # 打印 i 和 t
            print(f"Step {i}: t = {t}")  


        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(target_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                with torch.autocast(
                    device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
                ):
                    noise_pred = self.transformer(
                        latent_model_input,
                        t_expand,
                        text_states=prompt_embeds,
                        text_mask=prompt_mask,
                        text_states_2=prompt_embeds_2,
                        freqs_cos=freqs_cis[0],
                        freqs_sin=freqs_cis[1],
                        guidance=guidance_expand,
                        return_dict=True,
                    )["x"]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()

        self.transformer.to("cpu")
    
        return latents


    @torch.no_grad()
    def FlowEdit(
        self,
        latents: torch.Tensor,
        target_prompt: Union[str, List[str]],  # 目标提示
        source_prompt: Union[str, List[str]],     # 新增：源提示
        height: int,
        width: int,
        video_length: int,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        num_videos_per_prompt: Optional[int] = 1,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        tar_embedded_guidance_scale: Optional[float] = 5.5,       # 目标提示引导强度
        src_embedded_guidance_scale: Optional[float] = 1.5,       # 新增：源提示引导强度
        # FlowEditFLUX 新增参数
        n_avg: int = 1,       # 速度场平均次数
        n_min: int = 0,       # 最小常规采样步数
        n_max: int = 24,      # 最大ODE编辑步数
        **kwargs,
    ):
        # 1. Prepare the target prompt embeddings
        device = torch.device(f"cuda:{dist.get_rank()}") if dist.is_initialized() else self._execution_device
        
        # ========== 1. 双提示编码 ==========
        # 编码源提示
        (
            src_prompt_embeds,
            src_negative_prompt_embeds,
            src_prompt_mask,
            src_negative_prompt_mask,
        ) = self.encode_prompt(
            source_prompt,
            device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_attention_mask=negative_attention_mask,
            lora_scale=self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None,
            clip_skip=self.clip_skip,
            data_type="video",
        )
        if self.text_encoder_2 is not None:
            (
                src_prompt_embeds_2,
                src_negative_prompt_embeds_2,
                src_prompt_mask_2,
                src_negative_prompt_mask_2,
            ) = self.encode_prompt(
                source_prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                attention_mask=None,
                negative_prompt_embeds=None,
                negative_attention_mask=None,
                lora_scale=self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2,
                data_type="video",
            )
        else:
            src_prompt_embeds_2 = None
            src_negative_prompt_embeds_2 = None
            src_prompt_mask_2 = None
            src_negative_prompt_mask_2 = None

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            src_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds])
            if src_prompt_mask is not None:
                src_prompt_mask = torch.cat([src_negative_prompt_mask, src_prompt_mask])
            if src_prompt_embeds_2 is not None:
                src_prompt_embeds_2 = torch.cat([src_negative_prompt_embeds_2, src_prompt_embeds_2])
            if src_prompt_mask_2 is not None:
                src_prompt_mask_2 = torch.cat([src_negative_prompt_mask_2, src_prompt_mask_2])      
        
           
        # 编码目标提示   
        (
            tar_prompt_embeds,
            tar_negative_prompt_embeds,
            tar_prompt_mask,
            tar_negative_prompt_mask,
        ) = self.encode_prompt(
            target_prompt,
            device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_attention_mask=negative_attention_mask,
            lora_scale=self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None,
            clip_skip=self.clip_skip,
            data_type="video",
        )
        if self.text_encoder_2 is not None:
            (
                tar_prompt_embeds_2,
                tar_negative_prompt_embeds_2,
                tar_prompt_mask_2,
                tar_negative_prompt_mask_2,
            ) = self.encode_prompt(
                target_prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                attention_mask=None,
                negative_prompt_embeds=None,
                negative_attention_mask=None,
                lora_scale=self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2,
                data_type="video",
            )
        else:
            tar_prompt_embeds_2 = None
            tar_negative_prompt_embeds_2 = None
            tar_prompt_mask_2 = None
            tar_negative_prompt_mask_2 = None

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            tar_prompt_embeds = torch.cat([tar_negative_prompt_embeds, tar_prompt_embeds])
            if tar_prompt_mask is not None:
                tar_prompt_mask = torch.cat([tar_negative_prompt_mask, tar_prompt_mask])
            if tar_prompt_embeds_2 is not None:
                tar_prompt_embeds_2 = torch.cat([tar_negative_prompt_embeds_2, tar_prompt_embeds_2])
            if tar_prompt_mask_2 is not None:
                tar_prompt_mask_2 = torch.cat([tar_negative_prompt_mask_2, tar_prompt_mask_2])




        # ========== 2. 时间步配置 ==========
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            **self.prepare_extra_func_kwargs(self.scheduler.set_timesteps, {"n_tokens": n_tokens}),
        )

        # ========== 3. 潜在变量初始化 ==========
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size=latents.shape[0],
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            video_length=video_length,
            dtype=tar_prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # ========== 4. 主循环 ==========
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": generator, "eta": eta},
        )
        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not self.args.disable_autocast
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not self.args.disable_autocast

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        edit_latents = latents.clone()  # 初始化编辑状态
        for i, t in enumerate(timesteps):
            # 打印 i 和 t
            print(f"Step {i}: t = {t}")  


        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # 跳过超出n_max范围的步骤
                if num_inference_steps - i > n_max:
                    continue

                # ODE编辑阶段
                if num_inference_steps - i > n_min:


                    latent_model_input = (
                        torch.cat([latents] * 2)
                        if self.do_classifier_free_guidance
                        else latents
                    )
                    latent_model_input = self.scheduler.scale_model_input(
                        latent_model_input, t
                    )

                    zt_edit  = (
                        torch.cat([edit_latents] * 2)
                        if self.do_classifier_free_guidance
                        else edit_latents
                    )
                    zt_edit  = self.scheduler.scale_model_input(
                        zt_edit , t
                    )


                    V_delta_avg = torch.zeros_like(latent_model_input)

                    t_expand = t.repeat(latent_model_input.shape[0])

                    # print("src_embedded_guidance_scale: ",src_embedded_guidance_scale)
                    # print("tar_embedded_guidance_scale: ",tar_embedded_guidance_scale)
                    # print("source_prompt: ",source_prompt)
                    # print("target_prompt: ",target_prompt)
                    src_guidance_expand = (
                        torch.tensor(
                            [src_embedded_guidance_scale] * latent_model_input.shape[0],
                            dtype=torch.float32,
                            device=device,
                        ).to(target_dtype)
                        * 1000.0
                        if src_embedded_guidance_scale is not None
                        else None
                    )

                    tar_guidance_expand = (
                        torch.tensor(
                            [tar_embedded_guidance_scale] * latent_model_input.shape[0],
                            dtype=torch.float32,
                            device=device,
                        ).to(target_dtype)
                        * 1000.0
                        if tar_embedded_guidance_scale is not None
                        else None
                    )

                    t_i=t/1000
                    print("t_i: ",t_i)
                    for k in range(n_avg):
                        # 生成前向噪声
                        fwd_noise = torch.randn_like(latent_model_input)
                        zt_src = (1 - t_i) * latent_model_input + t_i * fwd_noise
                        zt_tar = zt_edit  + zt_src - latent_model_input

                        # 计算源提示速度场
                        with torch.autocast(
                            device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
                        ):
                            src_noise_pred = self.transformer(
                                zt_src,
                                t_expand,
                                text_states=src_prompt_embeds,
                                text_mask=src_prompt_mask,
                                text_states_2=src_prompt_embeds_2,
                                freqs_cos=freqs_cis[0],
                                freqs_sin=freqs_cis[1],
                                guidance=src_guidance_expand,
                                return_dict=True,
                            )["x"]

                        if self.do_classifier_free_guidance:
                            src_noise_pred_uncond, src_noise_pred_text = src_noise_pred.chunk(2)
                            src_noise_pred = src_noise_pred_uncond + self.guidance_scale * (
                                src_noise_pred_text - src_noise_pred_uncond
                            )

                        if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                            src_noise_pred = rescale_noise_cfg(
                                src_noise_pred,
                                src_noise_pred_text,
                                guidance_rescale=self.guidance_rescale,
                            )

                        # 计算目标提示速度场
                        with torch.autocast(
                            device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
                        ):
                            tar_noise_pred = self.transformer(
                                zt_tar,
                                t_expand,
                                text_states=tar_prompt_embeds,
                                text_mask=tar_prompt_mask,
                                text_states_2=tar_prompt_embeds_2,
                                freqs_cos=freqs_cis[0],
                                freqs_sin=freqs_cis[1],
                                guidance=tar_guidance_expand,
                                return_dict=True,
                            )["x"]

                        if self.do_classifier_free_guidance:
                            tar_noise_pred_uncond, tar_noise_pred_text = tar_noise_pred.chunk(2)
                            tar_noise_pred = tar_noise_pred_uncond + self.guidance_scale * (
                                tar_noise_pred_text - tar_noise_pred_uncond
                            )

                        if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                            tar_noise_pred = rescale_noise_cfg(
                                tar_noise_pred,
                                tar_noise_pred_text,
                                guidance_rescale=self.guidance_rescale,
                            )

                        V_delta_avg += (1/n_avg) * (tar_noise_pred - src_noise_pred)

                    # 欧拉法更新
                    zt_edit  = self.scheduler.step(
                        V_delta_avg, t, zt_edit , **extra_step_kwargs, return_dict=False
                    )[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        tar_prompt_embeds = callback_outputs.pop("tar_prompt_embeds", tar_prompt_embeds)
                        tar_negative_prompt_embeds = callback_outputs.pop(
                            "tar_negative_prompt_embeds", tar_negative_prompt_embeds
                        )








                # 常规采样阶段
                else:
                    if i == num_inference_steps - n_min:  # 初始化采样噪声
                        
                        latent_model_input = (
                            torch.cat([latents] * 2)
                            if self.do_classifier_free_guidance
                            else latents
                        )
                        latent_model_input = self.scheduler.scale_model_input(
                            latent_model_input, t
                        )


                        zt_edit  = (
                            torch.cat([edit_latents] * 2)
                            if self.do_classifier_free_guidance
                            else edit_latents
                        )
                        zt_edit  = self.scheduler.scale_model_input(
                            zt_edit , t
                        )

                        t_i=t/1000
                        fwd_noise = torch.randn_like(latent_model_input)
                        xt_src = (1 - t_i) * latent_model_input + t_i * fwd_noise
                        xt_tar = zt_edit + xt_src - latent_model_input


                    t_expand = t.repeat(latent_model_input.shape[0])

                    tar_guidance_expand = (
                        torch.tensor(
                            [tar_embedded_guidance_scale] * latent_model_input.shape[0],
                            dtype=torch.float32,
                            device=device,
                        ).to(target_dtype)
                        * 1000.0
                        if tar_embedded_guidance_scale is not None
                        else None
                    )



                    with torch.autocast(
                        device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
                    ):
                        tar_noise_pred = self.transformer(
                            xt_tar,
                            t_expand,
                            text_states=tar_prompt_embeds,
                            text_mask=tar_prompt_mask,
                            text_states_2=tar_prompt_embeds_2,
                            freqs_cos=freqs_cis[0],
                            freqs_sin=freqs_cis[1],
                            guidance=tar_guidance_expand,
                            return_dict=True,
                        )["x"]

                    if self.do_classifier_free_guidance:
                        tar_noise_pred_uncond, tar_noise_pred_text = tar_noise_pred.chunk(2)
                        tar_noise_pred = tar_noise_pred_uncond + self.guidance_scale * (
                            tar_noise_pred_text - tar_noise_pred_uncond
                        )

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        tar_noise_pred = rescale_noise_cfg(
                            tar_noise_pred,
                            tar_noise_pred_text,
                            guidance_rescale=self.guidance_rescale,
                        )

                    xt_tar = self.scheduler.step(
                        tar_noise_pred, t, xt_tar, **extra_step_kwargs, return_dict=False
                    )[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        tar_prompt_embeds = callback_outputs.pop("tar_prompt_embeds", tar_prompt_embeds)
                        tar_negative_prompt_embeds = callback_outputs.pop(
                            "tar_negative_prompt_embeds", tar_negative_prompt_embeds
                        )


                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()

        # ========== 5. 返回结果 ==========
        final_latents = zt_edit if n_min == 0 else xt_tar
        self.transformer.to("cpu")
        return final_latents



    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        video_tensor: torch.Tensor,
        source_prompt: Union[str, List[str]],
        target_prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        **kwargs,
    ):
        
        """
        Generate video from text prompts.

        Args:
            video_tensor (torch.Tensor): Input video tensor.
            source_prompt (Union[str, List[str]]): Source text prompt.
            target_prompt (Union[str, List[str]]): Target text prompt.
            height (int): Height of the video.
            width (int): Width of the video.
            video_length (int): Length of the video.
            data_type (str, optional): Type of data. Defaults to "video".
            num_inference_steps (int, optional): Number of inference steps. Defaults to 50.
            timesteps (List[int], optional): Timesteps for inference. Defaults to None.
            sigmas (List[float], optional): Sigmas for inference. Defaults to None.
            guidance_scale (float, optional): Guidance scale. Defaults to 7.5.
            negative_prompt (Optional[Union[str, List[str]]], optional): Negative text prompt. Defaults to None.
            num_videos_per_prompt (Optional[int], optional): Number of videos per prompt. Defaults to 1.
            eta (float, optional): Eta for inference. Defaults to 0.0.
            generator (Optional[Union[torch.Generator, List[torch.Generator]]], optional): Generator for random numbers. Defaults to None.
            latents (Optional[torch.Tensor], optional): Latents for inference. Defaults to None.
            prompt_embeds (Optional[torch.Tensor], optional): Prompt embeddings. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            negative_prompt_embeds (Optional[torch.Tensor], optional): Negative prompt embeddings. Defaults to None.
            negative_attention_mask (Optional[torch.Tensor], optional): Negative attention mask. Defaults to None.
            output_type (Optional[str], optional): Output type. Defaults to "pil".
            return_dict (bool, optional): Whether to return a dictionary. Defaults to True.
            cross_attention_kwargs (Optional[Dict[str, Any]], optional): Cross attention kwargs. Defaults to None.
            guidance_rescale (float, optional): Guidance rescale. Defaults to 0.0.
            clip_skip (Optional[int], optional): Clip skip. Defaults to None.
            callback_on_step_end (Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]], optional): Callback on step end. Defaults to None.
            callback_on_step_end_tensor_inputs (List[str], optional): Callback on step end tensor inputs. Defaults to ["latents"].
            freqs_cis (Tuple[torch.Tensor, torch.Tensor], optional): Frequencies for CIS. Defaults to None.
            vae_ver (str, optional): VAE version. Defaults to "88-4c-sd".
            enable_tiling (bool, optional): Enable tiling. Defaults to False.
            n_tokens (Optional[int], optional): Number of tokens. Defaults to None.
            embedded_guidance_scale (Optional[float], optional): Embedded guidance scale. Defaults to None.

        Returns:
            Union[torch.Tensor, np.ndarray]: Generated video.
        
        Examples:
            None
        """



        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (vae_dtype != torch.float32) and not self.args.disable_autocast



        # 确保 num_videos_per_prompt 有默认值
        if num_videos_per_prompt is None:
            num_videos_per_prompt = 1
            
        # 1. Encode the video tensor to latents
        #video_tensor = (video_tensor + 1) / 2  # 将视频张量从 [-1, 1] 范围转换到 [0, 1] 范围
        video_tensor = video_tensor.unsqueeze(0)  # 添加batch维度
        # 首先用 VAE 编码视频张量
        with torch.autocast(
            device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
        ):
            if enable_tiling:
                self.vae.enable_tiling()
                latents = self.vae.encode(video_tensor).latent_dist.sample()
            else:
                latents = self.vae.encode(video_tensor).latent_dist.sample()

        # 然后应用缩放/平移因子转换
        if (
            hasattr(self.vae.config, "shift_factor") 
            and self.vae.config.shift_factor
        ):
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            latents = latents * self.vae.config.scaling_factor


        # # 1. Perform inverse to get latents
        # self.transformer.to("cuda")
        # latents = self.inverse(
        #     latents,
        #     source_prompt,
        #     height,
        #     width,
        #     video_length,
        #     num_inference_steps,
        #     guidance_scale,
        #     negative_prompt,
        #     eta,
        #     generator,
        #     prompt_embeds,
        #     num_videos_per_prompt,
        #     attention_mask,
        #     negative_prompt_embeds,
        #     negative_attention_mask,
        #     cross_attention_kwargs,
        #     guidance_rescale,
        #     clip_skip,
        #     callback_on_step_end,
        #     callback_on_step_end_tensor_inputs,
        #     freqs_cis,
        #     vae_ver,
        #     enable_tiling,
        #     n_tokens,
        #     embedded_guidance_scale,
        #     **kwargs,
        # )

        # # 2. Perform reverse to generate video
        # latents = self.reverse(
        #     latents,
        #     target_prompt,
        #     height,
        #     width,
        #     video_length,
        #     num_inference_steps,
        #     guidance_scale,
        #     negative_prompt,
        #     eta,
        #     generator,
        #     prompt_embeds,
        #     num_videos_per_prompt,
        #     attention_mask,
        #     negative_prompt_embeds,
        #     negative_attention_mask,
        #     cross_attention_kwargs,
        #     guidance_rescale,
        #     clip_skip,
        #     callback_on_step_end,
        #     callback_on_step_end_tensor_inputs,
        #     freqs_cis,
        #     vae_ver,
        #     enable_tiling,
        #     n_tokens,
        #     embedded_guidance_scale,
        #     **kwargs,
        # )


        latents = self.FlowEdit(
            latents,
            target_prompt,
            source_prompt,
            height,
            width,
            video_length,
            num_inference_steps,
            guidance_scale,
            negative_prompt,
            eta,
            generator,
            prompt_embeds,
            num_videos_per_prompt,
            attention_mask,
            negative_prompt_embeds,
            negative_attention_mask,
            cross_attention_kwargs,
            guidance_rescale,
            clip_skip,
            callback_on_step_end,
            callback_on_step_end_tensor_inputs,
            freqs_cis,
            vae_ver,
            enable_tiling,
            n_tokens,
            tar_embedded_guidance_scale=embedded_guidance_scale,
            src_embedded_guidance_scale=1.5,
            # FlowEditFLUX 新增参数
            n_avg=1,
            n_min=0,
            n_max=24,
            **kwargs,
        )

        # 3. Decode latents to video
        self.transformer.to("cpu")
        if not output_type == "latent":
            expand_temporal_dim = False
            if len(latents.shape) == 4:
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    latents = latents.unsqueeze(2)
                    expand_temporal_dim = True
            elif len(latents.shape) == 5:
                pass
            else:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            if (
                hasattr(self.vae.config, "shift_factor")
                and self.vae.config.shift_factor
            ):
                latents = (
                    latents / self.vae.config.scaling_factor
                    + self.vae.config.shift_factor
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            with torch.autocast(
                device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
            ):
                if enable_tiling:
                    self.vae.enable_tiling()
                    image = self.vae.decode(
                        latents, return_dict=False, generator=generator
                    )[0]
                else:
                    image = self.vae.decode(
                        latents, return_dict=False, generator=generator
                    )[0]

            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)

        else:
            image = latents

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().float()

        self.maybe_free_model_hooks()

        if not return_dict:
            return image

        return HunyuanVideoPipelineOutput(videos=image)
from typing import Any, Callable, Dict, List
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import *
from diffusers.schedulers import (
    FlowMatchEulerDiscreteScheduler,
    DDIMScheduler,
    DDPMScheduler,
    ScoreSdeVeScheduler,
)

from torch import FloatTensor, Generator
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers.models.transformers.transformer_flux import *
import math
from torch.nn import functional as F

from util.trace import *
from util.sparse import apply_topk_attention_pruning, record_sparse_profile

from util.stat import time_recorder

rcd = time_recorder()

stat = []


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(
                -1
            )  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(
                -2
            )  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(
                f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
            )

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class AlterFluxPipeline(FluxPipeline):
    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__(
            scheduler,
            vae,
            text_encoder,
            tokenizer,
            text_encoder_2,
            tokenizer_2,
            transformer,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # Alter
        enableTraceGen: bool = False,
        tracePath: str = None,
        enableDiffInfer: bool = False,
        enableSparseReuse: bool = False,
        sparseReuseThreshold: float = 0.01,
        enableAttnCache: bool = False,
        enableSparseAttn: bool = False,
        sparseAttnMask: list[list[dict]] = None,
        sparseProf: list[list[dict]] = None,
    ):

        self.text_encoder.to("cuda")
        self.text_encoder_2.to("cuda")

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        self.text_encoder.to("cpu")
        self.text_encoder_2.to("cpu")
        self.vae.to("cpu")
        torch.cuda.empty_cache()

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = (
            None
            if timesteps is not None
            else np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        )
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # Init cache

        b = batch_size
        h = self.transformer.config["num_attention_heads"]
        patchSize = self.transformer.config["patch_size"]
        n = latents.shape[-2]
        nPromopt = prompt_embeds.shape[1]
        k = self.transformer.config["attention_head_dim"]

        if enableDiffInfer or enableSparseReuse:
            cacheDiff = [
                {}
                for _ in range(
                    len(self.transformer.transformer_blocks)
                    + len(self.transformer.single_transformer_blocks)
                )
            ]
        else:
            cacheDiff = [
                None
                for _ in range(
                    len(self.transformer.transformer_blocks)
                    + len(self.transformer.single_transformer_blocks)
                )
            ]

        if enableAttnCache:
            cacheAttn = [
                {}
                for _ in range(
                    len(self.transformer.transformer_blocks)
                    + len(self.transformer.single_transformer_blocks)
                )
            ]
            for IndexBlock in range(
                len(self.transformer.transformer_blocks)
                + len(self.transformer.single_transformer_blocks)
            ):
                if IndexBlock < len(self.transformer.transformer_blocks):
                    cacheAttn[IndexBlock]["qCache"] = torch.empty(
                        b, h, n + nPromopt, k, device=device, dtype=latents.dtype
                    )
                    cacheAttn[IndexBlock]["kCache"] = torch.empty(
                        b, h, n + nPromopt, k, device=device, dtype=latents.dtype
                    )
                else:
                    cacheAttn[IndexBlock]["qCache"] = torch.empty(
                        b, h, n + nPromopt, k, device=device, dtype=latents.dtype
                    )
                    cacheAttn[IndexBlock]["kCache"] = torch.empty(
                        b, h, n + nPromopt, k, device=device, dtype=latents.dtype
                    )

        else:
            cacheAttn = [
                None
                for _ in range(
                    len(self.transformer.transformer_blocks)
                    + len(self.transformer.single_transformer_blocks)
                )
            ]

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    #
                    enableTraceGen=enableTraceGen,
                    tracePath=tracePath,
                    enableDiffInfer=enableDiffInfer,
                    enableSparseReuse=enableSparseReuse,
                    sparseReuseThreshold=sparseReuseThreshold,
                    forceFill=(i == 0),
                    cacheDiff=cacheDiff,
                    #
                    enableAttnCache=enableAttnCache,
                    flushAttnCache=(i % 2 == 0),
                    cacheAttn=cacheAttn,
                    #
                    enableSparseAttn=enableSparseAttn,
                    sparseAttnMask=sparseAttnMask[i],
                    idxTimestep=i,
                    sparseProf=sparseProf,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        print(rcd.time)

        if output_type == "latent":
            image = latents

        else:
            self.vae.to(device)
            latents = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


class AlterFluxTransformer2DModel(FluxTransformer2DModel):
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: F.Tuple[int] = ...,
    ):
        super().__init__(
            patch_size,
            in_channels,
            num_layers,
            num_single_layers,
            attention_head_dim,
            num_attention_heads,
            joint_attention_dim,
            pooled_projection_dim,
            guidance_embeds,
            axes_dims_rope,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        # Alter
        enableTraceGen: bool = False,
        tracePath: str = None,
        enableDiffInfer: bool = False,
        enableSparseReuse: bool = False,
        sparseReuseThreshold: float = 0.01,
        forceFill: bool = False,
        cacheDiff: list[dict] = None,
        enableAttnCache: bool = False,
        flushAttnCache: bool = False,
        cacheAttn: list[dict] = None,
        enableSparseAttn: bool = False,
        sparseAttnMask: list[dict] = None,
        idxTimestep: int = None,
        sparseProf: list[list[dict]] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                joint_attention_kwargs is not None
                and joint_attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                encoder_hidden_states, hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    enableTraceGen=enableTraceGen,
                    tracePath=tracePath,
                    enableDiffInfer=enableDiffInfer,
                    enableSparseReuse=enableSparseReuse,
                    sparseReuseThreshold=sparseReuseThreshold,
                    forceFill=forceFill,
                    cacheDiff=cacheDiff[index_block],
                    enableAttnCache=enableAttnCache,
                    flushAttnCache=flushAttnCache,
                    cacheAttn=cacheAttn[index_block],
                    enableSparseAttn=enableSparseAttn,
                    sparseAttnMask=sparseAttnMask[index_block],
                    idxTimestep=idxTimestep,
                    sparseProf=sparseProf,
                    scale=(32.0),
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(
                    controlnet_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[
                            index_block % len(controlnet_block_samples)
                        ]
                    )
                else:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[index_block // interval_control]
                    )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    enableTraceGen=enableTraceGen,
                    tracePath=tracePath,
                    enableDiffInfer=enableDiffInfer,
                    enableSparseReuse=enableSparseReuse,
                    sparseReuseThreshold=sparseReuseThreshold,
                    forceFill=forceFill,
                    cacheDiff=cacheDiff[index_block + len(self.transformer_blocks)],
                    enableAttnCache=enableAttnCache,
                    flushAttnCache=flushAttnCache,
                    cacheAttn=cacheAttn[index_block + len(self.transformer_blocks)],
                    enableSparseAttn=enableSparseAttn,
                    sparseAttnMask=sparseAttnMask[
                        index_block + len(self.transformer_blocks)
                    ],
                    idxTimestep=idxTimestep,
                    sparseProf=sparseProf,
                    scale=(32.0),
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(
                    controlnet_single_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class AlterFluxTransformerBlock(FluxTransformerBlock):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        qk_norm="rms_norm",
        eps=0.000001,
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, qk_norm, eps)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        # Alter
        enableTraceGen: bool = False,
        tracePath: str = None,
        enableDiffInfer: bool = False,
        enableSparseReuse: bool = False,
        sparseReuseThreshold: float = 0.01,
        forceFill: bool = False,
        cacheDiff: dict = None,
        enableAttnCache: bool = False,
        flushAttnCache: bool = False,
        cacheAttn: dict = None,
        enableSparseAttn: bool = False,
        sparseAttnMask: dict = None,
        idxTimestep: int = None,
        sparseProf: list[list[dict]] = None,
        scale: float = None,
    ):

        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )
        joint_attention_kwargs = joint_attention_kwargs or {}
        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            enableTraceGen=enableTraceGen,
            tracePath=tracePath,
            enableAttnCache=enableAttnCache,
            flushAttnCache=flushAttnCache,
            qCache=(cacheAttn["qCache"] if cacheAttn else None),
            kCache=(cacheAttn["kCache"] if cacheAttn else None),
            enableSparseAttn=enableSparseAttn,
            sparseAttnMask=(sparseAttnMask["attn_joint"] if sparseAttnMask else None),
            idxTimestep=idxTimestep,
            idxBlock=self.idx_block,
            nameAttnBlock="attn_joint",
            sparseProf=sparseProf,
            **joint_attention_kwargs,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )

        if enableDiffInfer:
            from diff.diff import diff_ffn

            ff_output = diff_ffn(
                norm_hidden_states,
                cacheDiff,
                "ffn",
                forceFill,
                ffn_layer=self.ff,
                scale=scale,
                enableTraceGen=enableTraceGen,
                tracePath=tracePath,
            )

        elif enableSparseReuse:
            from diff.diff import sparse_reuse_ffn

            ff_output = sparse_reuse_ffn(
                norm_hidden_states,
                cacheDiff,
                "ffn",
                forceFill,
                ffn_layer=self.ff,
                threshold=sparseReuseThreshold,
                enableTraceGen=enableTraceGen,
                tracePath=tracePath,
            )

        else:
            if enableTraceGen:
                dump_trace(
                    linear(
                        (
                            norm_hidden_states.shape[0] * norm_hidden_states.shape[1],
                            self.ff.net[0].proj.weight.shape[0],
                            norm_hidden_states.shape[2],
                        ),
                        gelu=True,
                    ),
                    tracePath,
                )

            rcd.start()
            ff_output = self.ff.net[0].proj(norm_hidden_states)
            rcd.end()

            rcd.start()
            ff_output = F.gelu(ff_output, approximate="tanh")
            rcd.end()

            ff_output = self.ff.net[1](ff_output)

            if enableTraceGen:
                dump_trace(
                    linear(
                        (
                            ff_output.shape[0] * ff_output.shape[1],
                            self.ff.net[2].weight.shape[0],
                            ff_output.shape[2],
                        )
                    ),
                    tracePath,
                )

            rcd.start()
            ff_output = self.ff.net[2](ff_output)
            rcd.end()

        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
            + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context.net[0].proj(norm_encoder_hidden_states)
        context_ff_output = F.gelu(context_ff_output, approximate="tanh")
        context_ff_output = self.ff_context.net[1](context_ff_output)
        context_ff_output = self.ff_context.net[2](context_ff_output)

        encoder_hidden_states = (
            encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        )
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class AlterFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4):
        super().__init__(dim, num_attention_heads, attention_head_dim, mlp_ratio)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        # Alter
        enableTraceGen: bool = False,
        tracePath: str = None,
        enableDiffInfer: bool = False,
        enableSparseReuse: bool = False,
        sparseReuseThreshold: float = 0.01,
        forceFill: bool = False,
        cacheDiff: dict = None,
        enableAttnCache: bool = False,
        flushAttnCache: bool = False,
        cacheAttn: dict = None,
        enableSparseAttn: bool = False,
        sparseAttnMask: dict = None,
        idxTimestep: int = None,
        sparseProf: list[list[dict]] = None,
        scale: float = None,
    ):

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        joint_attention_kwargs = joint_attention_kwargs or {}

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            enableTraceGen=enableTraceGen,
            tracePath=tracePath,
            enableAttnCache=enableAttnCache,
            flushAttnCache=flushAttnCache,
            qCache=(cacheAttn["qCache"] if cacheAttn else None),
            kCache=(cacheAttn["kCache"] if cacheAttn else None),
            enableSparseAttn=enableSparseAttn,
            sparseAttnMask=(sparseAttnMask["attn_single"] if sparseAttnMask else None),
            idxTimestep=idxTimestep,
            idxBlock=self.idx_block,
            nameAttnBlock="attn_single",
            sparseProf=sparseProf,
        )

        ff_weight1 = self.proj_out.weight[:, : attn_output.shape[-1]]
        ff_weight2 = self.proj_out.weight[:, attn_output.shape[-1] :]

        ff_output1 = torch.matmul(attn_output, ff_weight1.T) + self.proj_out.bias

        if enableDiffInfer:
            from diff.diff import diff_ffn

            ff_output2 = diff_ffn(
                norm_hidden_states,
                cacheDiff,
                "ffn",
                forceFill,
                weight0=self.proj_mlp.weight,
                bias0=self.proj_mlp.bias,
                weight1=ff_weight2,
                bias1=self.proj_out.bias,
                scale=scale,
                enableTraceGen=enableTraceGen,
                tracePath=tracePath,
            )
        elif enableSparseReuse:
            from diff.diff import sparse_reuse_ffn

            ff_output2 = sparse_reuse_ffn(
                norm_hidden_states,
                cacheDiff,
                "ffn",
                forceFill,
                weight0=self.proj_mlp.weight,
                bias0=self.proj_mlp.bias,
                weight1=ff_weight2,
                bias1=self.proj_out.bias,
                threshold=sparseReuseThreshold,
                enableTraceGen=enableTraceGen,
                tracePath=tracePath,
            )
        else:
            if enableTraceGen:
                dump_trace(
                    linear(
                        (
                            norm_hidden_states.shape[0] * norm_hidden_states.shape[1],
                            self.proj_mlp.weight.shape[0],
                            norm_hidden_states.shape[2],
                        ),
                        gelu=True,
                    ),
                    tracePath,
                )

            rcd.start()
            mlp_hidden_states = self.proj_mlp(norm_hidden_states)
            rcd.end()
            rcd.start()
            mlp_hidden_states = self.act_mlp(mlp_hidden_states)
            rcd.end()

            if enableTraceGen:
                dump_trace(
                    linear(
                        (
                            mlp_hidden_states.shape[0] * mlp_hidden_states.shape[1],
                            ff_weight2.shape[0],
                            mlp_hidden_states.shape[2],
                        )
                    ),
                    tracePath,
                )

            rcd.start()
            ff_output2 = (
                torch.matmul(mlp_hidden_states, ff_weight2.T) + self.proj_out.bias
            )
            rcd.end()

        ff_output = ff_output1 + ff_output2

        gate = gate.unsqueeze(1)
        hidden_states = gate * ff_output

        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class AlterAttention(Attention):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 8,
        kv_heads: int | None = None,
        dim_head: int = 64,
        dropout: float = 0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: str | None = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: str | None = None,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        norm_num_groups: int | None = None,
        spatial_norm_dim: int | None = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 0.00001,
        rescale_output_factor: float = 1,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: AttnProcessor | None = None,
        out_dim: int = None,
        context_pre_only=None,
        pre_only=False,
    ):
        super().__init__(
            query_dim,
            cross_attention_dim,
            heads,
            kv_heads,
            dim_head,
            dropout,
            bias,
            upcast_attention,
            upcast_softmax,
            cross_attention_norm,
            cross_attention_norm_num_groups,
            qk_norm,
            added_kv_proj_dim,
            added_proj_bias,
            norm_num_groups,
            spatial_norm_dim,
            out_bias,
            scale_qk,
            only_cross_attention,
            eps,
            rescale_output_factor,
            residual_connection,
            _from_deprecated_attn_block,
            processor,
            out_dim,
            context_pre_only,
            pre_only,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # Alter
        enableTraceGen: bool = False,
        tracePath: str = None,
        enableAttnCache: bool = False,
        flushAttnCache: bool = False,
        qCache: torch.Tensor = None,
        kCache: torch.Tensor = None,
        enableSparseAttn: bool = False,
        sparseAttnMask: torch.Tensor = None,
        idxTimestep: int = None,
        idxBlock: int = None,
        nameAttnBlock: str = None,
        sparseProf: list[list[dict]] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(
            inspect.signature(self.processor.__call__).parameters.keys()
        )
        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k
            for k, _ in cross_attention_kwargs.items()
            if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {
            k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters
        }

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            enableTraceGen=enableTraceGen,
            tracePath=tracePath,
            enableAttnCache=enableAttnCache,
            flushAttnCache=flushAttnCache,
            qCache=qCache,
            kCache=kCache,
            enableSparseAttn=enableSparseAttn,
            sparseAttnMask=sparseAttnMask,
            idxTimestep=idxTimestep,
            idxBlock=idxBlock,
            nameAttnBlock=nameAttnBlock,
            sparseProf=sparseProf,
            **cross_attention_kwargs,
        )


class AlterFluxAttnProcessor2_0(FluxAttnProcessor2_0):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        # Alter
        enableTraceGen: bool = False,
        tracePath: str = None,
        enableAttnCache: bool = False,
        flushAttnCache: bool = False,
        qCache: torch.Tensor = None,
        kCache: torch.Tensor = None,
        enableSparseAttn: bool = False,
        sparseAttnMask: torch.Tensor = None,
        idxTimestep: int = None,
        idxBlock: int = None,
        nameAttnBlock: str = None,
        sparseProf: list[list[dict]] = None,
    ) -> torch.FloatTensor:

        name_attn_block = (
            "attn_joint" if encoder_hidden_states is not None else "attn_single"
        )

        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        # `sample` projections.

        if enableTraceGen:
            if not (enableAttnCache and (not flushAttnCache)):
                dump_trace(
                    linear(
                        (
                            hidden_states.shape[0] * hidden_states.shape[1],
                            attn.to_q.weight.shape[0],
                            hidden_states.shape[2],
                        )
                    ),
                    tracePath,
                )
                dump_trace(
                    linear(
                        (
                            hidden_states.shape[0] * hidden_states.shape[1],
                            attn.to_q.weight.shape[0],
                            hidden_states.shape[2],
                        )
                    ),
                    tracePath,
                )
            dump_trace(
                linear(
                    (
                        hidden_states.shape[0] * hidden_states.shape[1],
                        attn.to_q.weight.shape[0],
                        hidden_states.shape[2],
                    )
                ),
                tracePath,
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.

            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            #     from .embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # rcd.start()
        # hidden_states_test = F.scaled_dot_product_attention(
        #     query, key, value, dropout_p=0.0, is_causal=False
        # )
        # rcd.end()

        if enableTraceGen:
            attn_shape_trace = [
                *query.shape[:2],
                query.shape[2] // 64,
                query.shape[2] // 64,
            ]
            attn_mask_trace = (
                torch.logical_not(sparseAttnMask)
                if enableSparseAttn
                else torch.full(attn_shape_trace, True)
            )
            dump_trace(
                attention(
                    query.shape,
                    (enableAttnCache and (not flushAttnCache)),
                    attn_mask_trace,
                ),
                tracePath,
            )

        if enableAttnCache:
            if flushAttnCache:
                h = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
                qCache.copy_(query)
                kCache.copy_(key)
            else:
                h = torch.matmul(qCache, kCache.transpose(-1, -2)) / math.sqrt(head_dim)
        else:
            h = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)

        h = apply_topk_attention_pruning(h)

        if enableSparseAttn:
            attnMask = torch.full_like(h, 0)
            sparseAttnMaskUnpacked = (
                (
                    sparseAttnMask.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(*sparseAttnMask.shape, 64, 64)
                )
                .transpose(-3, -2)
                .flatten(-2, -1)
                .flatten(-3, -2)
            )
            attnMask[..., -hidden_states.shape[1] :, -hidden_states.shape[1] :] = (
                torch.where(sparseAttnMaskUnpacked, -torch.inf, 0)
            )
            h = h + attnMask

        s = h.softmax(-1)

        record_sparse_profile(
            sparseProf,
            idxTimestep,
            idxBlock,
            nameAttnBlock or name_attn_block,
            s[..., -hidden_states.shape[1] :, -hidden_states.shape[1] :],
            s.shape[-1],
        )

        hidden_states = torch.matmul(s, value)

        # from kernel.block_spmm import block_sparse_spmm, block_mask_to_csr_fast

        # if enableAttnCache:
        #     if flushAttnCache:
        #         if enableSparseAttn:
        #             row_ptr, col_idx = block_mask_to_csr_fast(sparseAttnMask)
        #             rcd.start()
        #             s = h.softmax(dim=-1)
        #             attn_output = block_sparse_spmm(
        #                 s, value, row_ptr, col_idx, block_size=64
        #             )
        #             attn_output = block_sparse_spmm(
        #                 s, value, row_ptr, col_idx, block_size=64
        #             )
        #             rcd.end()
        #         else:
        #             rcd.start()
        #             h = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
        #             s = h.softmax(dim=-1)
        #             attn_output = torch.matmul(s, value)
        #             rcd.end()
        #     else:
        #         if enableSparseAttn:
        #             row_ptr, col_idx = block_mask_to_csr_fast(sparseAttnMask)
        #             rcd.start()
        #             attn_output = block_sparse_spmm(
        #                 s, value, row_ptr, col_idx, block_size=64
        #             )

        #             rcd.end()
        #         else:
        #             rcd.start()
        #             attn_output = torch.matmul(s, value)
        #             rcd.end()

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            if enableTraceGen:
                dump_trace(
                    linear(
                        (
                            hidden_states.shape[0] * hidden_states.shape[1],
                            attn.to_out[0].weight.shape[0],
                            hidden_states.shape[2],
                        )
                    ),
                    tracePath,
                )
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:

            return hidden_states


def apply_alter(
    pipeline: FluxPipeline,
):
    pipeline.__class__ = AlterFluxPipeline
    diffusion_model = pipeline.transformer
    diffusion_model.__class__ = AlterFluxTransformer2DModel

    processor = AlterFluxAttnProcessor2_0()
    block_cnt = 0
    for _, module in diffusion_model.named_modules():
        if module.__class__.__name__ == "FluxTransformerBlock":
            module.__class__ = AlterFluxTransformerBlock
            module.attn.__class__ = AlterAttention
            module.attn.set_processor(processor)
            module.idx_block = block_cnt
            block_cnt += 1
        if module.__class__.__name__ == "FluxSingleTransformerBlock":
            module.__class__ = AlterFluxSingleTransformerBlock
            module.attn.__class__ = AlterAttention
            module.attn.set_processor(processor)
            module.idx_block = block_cnt
            block_cnt += 1

    return pipeline


def preprocess(prompt):
    return prompt


def postprocess(output, path, name):
    output.images[0].save(f"{path}{name}")

from diffusers.models.attention_processor import AttnProcessor
import torch
from diffusers.models import AutoencoderKL, PixArtTransformer2DModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers import PixArtSigmaPipeline
from diffusers.pipelines.latte.pipeline_latte import *
from diffusers.models.transformers.latte_transformer_3d import *
from diffusers.models.attention import *
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.models.attention_processor import *

from util.trace import *
from util.sparse import apply_topk_attention_pruning, record_sparse_profile

from util.stat import time_recorder

rcd = time_recorder()

stat = []


class AlterLattePipeline(LattePipeline):
    def __init__(self, tokenizer, text_encoder, vae, transformer, scheduler):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        video_length: int = 16,
        height: int = 512,
        width: int = 512,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        clean_caption: bool = True,
        mask_feature: bool = True,
        enable_temporal_attentions: bool = True,
        decode_chunk_size: Optional[int] = None,
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
    ) -> Union[LattePipelineOutput, Tuple]:

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default
        decode_chunk_size = (
            decode_chunk_size if decode_chunk_size is not None else video_length
        )

        # 1. Check inputs. Raise error if not correct
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clean_caption=clean_caption,
            mask_feature=mask_feature,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            video_length,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        # Init cache
        b = batch_size * video_length * (2 if do_classifier_free_guidance else 1)
        h = self.transformer.config["num_attention_heads"]
        patchSize = self.transformer.config["patch_size"]
        n = (latents.shape[-2] // patchSize) * (latents.shape[-1] // patchSize)
        nPromopt = prompt_embeds.shape[1]
        k = self.transformer.config["attention_head_dim"]

        if enableDiffInfer or enableSparseReuse:
            cacheDiff = [{} for _ in range(len(self.transformer.transformer_blocks))]
        else:
            cacheDiff = [None for _ in range(len(self.transformer.transformer_blocks))]

        if enableAttnCache:
            cacheAttn = [{} for _ in range(len(self.transformer.transformer_blocks))]
            for IndexBlock in range(len(self.transformer.transformer_blocks)):
                cacheAttn[IndexBlock]["qCache"] = torch.empty(
                    b, h, n, k, device=device, dtype=latents.dtype
                )
                cacheAttn[IndexBlock]["kCache"] = torch.empty(
                    b, h, n, k, device=device, dtype=latents.dtype
                )

                cacheAttn[IndexBlock]["qCache2"] = torch.empty(
                    b, h, n, k, device=device, dtype=latents.dtype
                )
                cacheAttn[IndexBlock]["kCache2"] = torch.empty(
                    b, h, nPromopt, k, device=device, dtype=latents.dtype
                )
        else:
            cacheAttn = [None for _ in range(len(self.transformer.transformer_blocks))]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor(
                        [current_timestep],
                        dtype=dtype,
                        device=latent_model_input.device,
                    )
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(
                        latent_model_input.device
                    )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=current_timestep,
                    enable_temporal_attentions=enable_temporal_attentions,
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
                    enableSparseAttn=False,
                    sparseAttnMask=sparseAttnMask[i],
                    idxTimestep=i,
                    sparseProf=sparseProf,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # use learned sigma?
                if not (
                    hasattr(self.scheduler.config, "variance_type")
                    and self.scheduler.config.variance_type
                    in ["learned", "learned_range"]
                ):
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                # compute previous video: x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
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
                    progress_bar.update()

        print(rcd.time)

        if not output_type == "latents":
            video = self.decode_latents(latents, video_length, decode_chunk_size=14)
            video = self.video_processor.postprocess_video(
                video=video, output_type=output_type
            )
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return LattePipelineOutput(frames=video)


class AlterLatteTransformer3DModel(LatteTransformer3DModel):
    def __init__(
        self,
        num_attention_heads=16,
        attention_head_dim=88,
        in_channels=None,
        out_channels=None,
        num_layers=1,
        dropout=0,
        cross_attention_dim=None,
        attention_bias=False,
        sample_size=64,
        patch_size=None,
        activation_fn="geglu",
        num_embeds_ada_norm=None,
        norm_type="layer_norm",
        norm_elementwise_affine=True,
        norm_eps=0.00001,
        caption_channels=None,
        video_length=16,
    ):
        super().__init__(
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            num_layers,
            dropout,
            cross_attention_dim,
            attention_bias,
            sample_size,
            patch_size,
            activation_fn,
            num_embeds_ada_norm,
            norm_type,
            norm_elementwise_affine,
            norm_eps,
            caption_channels,
            video_length,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        enable_temporal_attentions: bool = True,
        return_dict: bool = True,
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
    ):
        # Reshape hidden states
        batch_size, channels, num_frame, height, width = hidden_states.shape
        # batch_size channels num_frame height width -> (batch_size * num_frame) channels height width
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            -1, channels, height, width
        )

        # Input
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )
        num_patches = height * width

        hidden_states = self.pos_embed(
            hidden_states
        )  # alrady add positional embeddings

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        timestep, embedded_timestep = self.adaln_single(
            timestep,
            added_cond_kwargs=added_cond_kwargs,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        # Prepare text embeddings for spatial block
        # batch_size num_tokens hidden_size -> (batch_size * num_frame) num_tokens hidden_size
        encoder_hidden_states = self.caption_projection(
            encoder_hidden_states
        )  # 3 120 1152
        encoder_hidden_states_spatial = encoder_hidden_states.repeat_interleave(
            num_frame, dim=0
        ).view(-1, encoder_hidden_states.shape[-2], encoder_hidden_states.shape[-1])

        # Prepare timesteps for spatial and temporal block
        timestep_spatial = timestep.repeat_interleave(num_frame, dim=0).view(
            -1, timestep.shape[-1]
        )
        timestep_temp = timestep.repeat_interleave(num_patches, dim=0).view(
            -1, timestep.shape[-1]
        )

        # Spatial and temporal transformer blocks
        for i, (spatial_block, temp_block) in enumerate(
            zip(self.transformer_blocks, self.temporal_transformer_blocks)
        ):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    spatial_block,
                    hidden_states,
                    None,  # attention_mask
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    None,  # cross_attention_kwargs
                    None,  # class_labels
                    use_reentrant=False,
                )
            else:
                hidden_states = spatial_block(
                    hidden_states,
                    None,  # attention_mask
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    None,  # cross_attention_kwargs
                    None,  # class_labels
                    enableTraceGen=enableTraceGen,
                    tracePath=tracePath,
                    enableDiffInfer=enableDiffInfer,
                    enableSparseReuse=enableSparseReuse,
                    sparseReuseThreshold=sparseReuseThreshold,
                    forceFill=forceFill,
                    cacheDiff=cacheDiff[i],
                    enableAttnCache=enableAttnCache,
                    flushAttnCache=flushAttnCache,
                    cacheAttn=cacheAttn[i],
                    enableSparseAttn=enableSparseAttn,
                    sparseAttnMask=sparseAttnMask[i],
                    idxTimestep=idxTimestep,
                    sparseProf=sparseProf,
                    scale=(128.0 if i < 3 else 64.0),
                )

            if enable_temporal_attentions:
                # (batch_size * num_frame) num_tokens hidden_size -> (batch_size * num_tokens) num_frame hidden_size
                hidden_states = hidden_states.reshape(
                    batch_size, -1, hidden_states.shape[-2], hidden_states.shape[-1]
                ).permute(0, 2, 1, 3)
                hidden_states = hidden_states.reshape(
                    -1, hidden_states.shape[-2], hidden_states.shape[-1]
                )

                if i == 0 and num_frame > 1:
                    hidden_states = hidden_states + self.temp_pos_embed

                if self.training and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        temp_block,
                        hidden_states,
                        None,  # attention_mask
                        None,  # encoder_hidden_states
                        None,  # encoder_attention_mask
                        timestep_temp,
                        None,  # cross_attention_kwargs
                        None,  # class_labels
                        use_reentrant=False,
                    )
                else:
                    hidden_states = temp_block(
                        hidden_states,
                        None,  # attention_mask
                        None,  # encoder_hidden_states
                        None,  # encoder_attention_mask
                        timestep_temp,
                        None,  # cross_attention_kwargs
                        None,  # class_labels
                    )

                # (batch_size * num_tokens) num_frame hidden_size -> (batch_size * num_frame) num_tokens hidden_size
                hidden_states = hidden_states.reshape(
                    batch_size, -1, hidden_states.shape[-2], hidden_states.shape[-1]
                ).permute(0, 2, 1, 3)
                hidden_states = hidden_states.reshape(
                    -1, hidden_states.shape[-2], hidden_states.shape[-1]
                )

        embedded_timestep = embedded_timestep.repeat_interleave(num_frame, dim=0).view(
            -1, embedded_timestep.shape[-1]
        )
        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None]
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(
                -1,
                height,
                width,
                self.config.patch_size,
                self.config.patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                -1,
                self.out_channels,
                height * self.config.patch_size,
                width * self.config.patch_size,
            )
        )
        output = output.reshape(
            batch_size, -1, output.shape[-3], output.shape[-2], output.shape[-1]
        ).permute(0, 2, 1, 3, 4)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class AlterBasicTransformerBlock(BasicTransformerBlock):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int | None = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 0.00001,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: str | None = None,
        num_positional_embeddings: int | None = None,
        ada_norm_continous_conditioning_embedding_dim: int | None = None,
        ada_norm_bias: int | None = None,
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__(
            dim,
            num_attention_heads,
            attention_head_dim,
            dropout,
            cross_attention_dim,
            activation_fn,
            num_embeds_ada_norm,
            attention_bias,
            only_cross_attention,
            double_self_attention,
            upcast_attention,
            norm_elementwise_affine,
            norm_type,
            norm_eps,
            final_dropout,
            attention_type,
            positional_embeddings,
            num_positional_embeddings,
            ada_norm_continous_conditioning_embedding_dim,
            ada_norm_bias,
            ff_inner_dim,
            ff_bias,
            attention_out_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
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
    ) -> torch.Tensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored."
                )
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = (
            cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        )
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            enableTraceGen=enableTraceGen,
            tracePath=tracePath,
            enableAttnCache=enableAttnCache,
            flushAttnCache=flushAttnCache,
            qCache=(cacheAttn["qCache"] if cacheAttn else None),
            kCache=(cacheAttn["kCache"] if cacheAttn else None),
            enableSparseAttn=enableSparseAttn,
            sparseAttnMask=(sparseAttnMask["attn_self"] if sparseAttnMask else None),
            idxTimestep=idxTimestep,
            idxBlock=self.idx_block,
            sparseProf=sparseProf,
            **cross_attention_kwargs,
        )

        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(
                    hidden_states, added_cond_kwargs["pooled_text_emb"]
                )
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                enableAttnCache=enableAttnCache,
                flushAttnCache=flushAttnCache,
                qCache=(cacheAttn["qCache2"] if cacheAttn else None),
                kCache=(cacheAttn["kCache2"] if cacheAttn else None),
                enableSparseAttn=False,
                sparseAttnMask=None,
                idxTimestep=idxTimestep,
                idxBlock=self.idx_block,
                sparseProf=sparseProf,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size
            )
        else:
            if enableDiffInfer:
                from diff.diff import diff_ffn

                ff_output = diff_ffn(
                    norm_hidden_states,
                    cacheDiff,
                    "ffn",
                    forceFill,
                    self.ff,
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
                    self.ff,
                    threshold=sparseReuseThreshold,
                    enableTraceGen=enableTraceGen,
                    tracePath=tracePath,
                )

            else:
                if enableTraceGen:
                    dump_trace(
                        linear(
                            (
                                norm_hidden_states.shape[0]
                                * norm_hidden_states.shape[1],
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

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

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
        elementwise_affine: bool = True,
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
            elementwise_affine,
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
            sparseProf=sparseProf,
            **cross_attention_kwargs,
        )


class AlterAttnProcessor2_0(AttnProcessor2_0):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
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
        sparseProf: list[list[dict]] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        name_attn_block = (
            "attn_cross" if encoder_hidden_states is not None else "attn_self"
        )
        type_transformer_block = (
            "spatial" if hidden_states.shape[0] == 32 else "temporal"
        )

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

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

        rcd.start()
        query = attn.to_q(hidden_states)
        rcd.end()

        # print(torch.any(query.isnan()))
        # if torch.any(query.isnan()):
        #     print(idx_timestep, idx_block, torch.any(hidden_states.isnan()))

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        rcd.start()
        key = attn.to_k(encoder_hidden_states)
        rcd.end()

        rcd.start()
        value = attn.to_v(encoder_hidden_states)
        rcd.end()

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # query, key = quantize_qk(query, key)

        # print(hidden_states.shape)

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
                h = torch.matmul(query / math.sqrt(head_dim), key.transpose(-1, -2))
                qCache.copy_(query)
                kCache.copy_(key)
            else:
                h = torch.matmul(qCache / math.sqrt(head_dim), kCache.transpose(-1, -2))
        else:
            h = torch.matmul(query / math.sqrt(head_dim), key.transpose(-1, -2))

        assert not ((attention_mask is not None) and (enableSparseAttn))
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
            attnMask[..., : hidden_states.shape[1], : hidden_states.shape[1]] = (
                torch.where(sparseAttnMaskUnpacked, -torch.inf, 0)
            )
            h = h + attnMask

        if attention_mask is not None:
            h = h + attention_mask

        h = apply_topk_attention_pruning(h)

        s = h.softmax(-1)

        if name_attn_block == "attn_self" and type_transformer_block == "spatial":
            record_sparse_profile(
                sparseProf,
                idxTimestep,
                idxBlock,
                name_attn_block,
                s[..., : hidden_states.shape[1], : hidden_states.shape[1]],
                s.shape[-1],
            )

        hidden_states = torch.matmul(s, value)

        from kernel.block_spmm import block_sparse_spmm, block_mask_to_csr_fast

        if enableAttnCache:
            if flushAttnCache:
                if enableSparseAttn:
                    row_ptr, col_idx = block_mask_to_csr_fast(sparseAttnMask)
                    rcd.start()
                    s = h.softmax(dim=-1)
                    attn_output = block_sparse_spmm(
                        s, value, row_ptr, col_idx, block_size=64
                    )
                    attn_output = block_sparse_spmm(
                        s, value, row_ptr, col_idx, block_size=64
                    )
                    rcd.end()
                else:
                    rcd.start()
                    h = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
                    s = h.softmax(dim=-1)
                    attn_output = torch.matmul(s, value)
                    rcd.end()
            else:
                if enableSparseAttn:
                    row_ptr, col_idx = block_mask_to_csr_fast(sparseAttnMask)
                    rcd.start()
                    attn_output = block_sparse_spmm(
                        s, value, row_ptr, col_idx, block_size=64
                    )

                    rcd.end()
                else:
                    rcd.start()
                    attn_output = torch.matmul(s, value)
                    rcd.end()

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

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

        rcd.start()
        hidden_states = attn.to_out[0](hidden_states)
        rcd.end()

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def apply_alter(
    pipeline: LattePipeline,
):
    pipeline.__class__ = AlterLattePipeline
    diffusion_model = pipeline.transformer
    diffusion_model.__class__ = AlterLatteTransformer3DModel

    processor = AlterAttnProcessor2_0()
    block_cnt = 0
    for _, module in diffusion_model.transformer_blocks.named_modules():
        if module.__class__.__name__ == "BasicTransformerBlock":
            module.__class__ = AlterBasicTransformerBlock
            module.attn1.__class__ = AlterAttention
            module.attn1.set_processor(processor)
            if module.attn2 is not None:
                module.attn2.__class__ = AlterAttention
                module.attn2.set_processor(processor)
            module.idx_block = block_cnt
            block_cnt += 1

    return pipeline


def preprocess(prompt):
    return prompt


def postprocess(output, path, name):
    import imageio

    imageio.mimwrite(
        f"{path}{name}.mp4",
        output.frames.cpu()[0].permute(0, 2, 3, 1),
        fps=8,
        quality=9,
    )

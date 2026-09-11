import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import *
from diffusers.models.transformers.transformer_sd3 import *
from diffusers.models.attention_processor import *

from util.trace import *
from util.sparse import apply_topk_attention_pruning, record_sparse_profile

from util.stat import time_recorder

rcd = time_recorder()
stat = []


class AlterStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
    ):
        super().__init__(
            transformer,
            scheduler,
            vae,
            text_encoder,
            tokenizer,
            text_encoder_2,
            tokenizer_2,
            text_encoder_3,
            tokenizer_3,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
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
        self.text_encoder_3.to("cuda")

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
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
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        self.text_encoder.to("cpu")
        self.text_encoder_2.to("cpu")
        self.text_encoder_3.to("cpu")
        self.vae.to("cpu")
        torch.cuda.empty_cache()

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Init cache

        b = batch_size * (2 if self.do_classifier_free_guidance else 1)
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
                    b, h, n + nPromopt, k, device=device, dtype=latents.dtype
                )
                cacheAttn[IndexBlock]["kCache"] = torch.empty(
                    b, h, n + nPromopt, k, device=device, dtype=latents.dtype
                )
                if IndexBlock in self.transformer.config["dual_attention_layers"]:
                    cacheAttn[IndexBlock]["qCache2"] = torch.empty(
                        b, h, n, k, device=device, dtype=latents.dtype
                    )
                    cacheAttn[IndexBlock]["kCache2"] = torch.empty(
                        b, h, n, k, device=device, dtype=latents.dtype
                    )
        else:
            cacheAttn = [None for _ in range(len(self.transformer.transformer_blocks))]

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # if i in flush_steps:
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
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

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

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
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

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
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)


class AlterSD3Transformer2DModel(SD3Transformer2DModel):
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: Tuple[int] = ...,
        qk_norm: str | None = None,
    ):
        super().__init__(
            sample_size,
            patch_size,
            in_channels,
            num_layers,
            attention_head_dim,
            num_attention_heads,
            joint_attention_dim,
            caption_projection_dim,
            pooled_projection_dim,
            out_channels,
            pos_embed_max_size,
            dual_attention_layers,
            qk_norm,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
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

        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(
            hidden_states
        )  # takes care of adding positional embeddings too.

        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

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
                        **ckpt_kwargs,
                    )
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
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
                    scale=(128.0 if index_block < 3 else 32.0),
                )

            # controlnet residual
            if (
                block_controlnet_hidden_states is not None
                and block.context_pre_only is False
            ):
                interval_control = len(self.transformer_blocks) // len(
                    block_controlnet_hidden_states
                )
                hidden_states = (
                    hidden_states
                    + block_controlnet_hidden_states[index_block // interval_control]
                )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
        )

        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self.out_channels,
                height * patch_size,
                width * patch_size,
            )
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class AlterJointTransformerBlock(JointTransformerBlock):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: str | None = None,
        use_dual_attention: bool = False,
    ):
        super().__init__(
            dim,
            num_attention_heads,
            attention_head_dim,
            context_pre_only,
            qk_norm,
            use_dual_attention,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
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

        if self.use_dual_attention:
            (
                norm_hidden_states,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                norm_hidden_states2,
                gate_msa2,
            ) = self.norm1(hidden_states, emb=temb)
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, emb=temb
            )

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            (
                norm_encoder_hidden_states,
                c_gate_msa,
                c_shift_mlp,
                c_scale_mlp,
                c_gate_mlp,
            ) = self.norm1_context(encoder_hidden_states, emb=temb)

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
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
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2(
                hidden_states=norm_hidden_states2,
                enableTraceGen=enableTraceGen,
                tracePath=tracePath,
                enableAttnCache=enableAttnCache,
                flushAttnCache=flushAttnCache,
                qCache=(cacheAttn["qCache2"] if cacheAttn else None),
                kCache=(cacheAttn["kCache2"] if cacheAttn else None),
                enableSparseAttn=enableSparseAttn,
                sparseAttnMask=(
                    sparseAttnMask["attn_self"] if sparseAttnMask else None
                ),
                idxTimestep=idxTimestep,
                idxBlock=self.idx_block,
                nameAttnBlock="attn_self",
                sparseProf=sparseProf,
            )
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
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

        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = (
                norm_encoder_hidden_states * (1 + c_scale_mlp[:, None])
                + c_shift_mlp[:, None]
            )
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context,
                    norm_encoder_hidden_states,
                    self._chunk_dim,
                    self._chunk_size,
                )
            else:
                context_ff_output = self.ff_context.net[0].proj(
                    norm_encoder_hidden_states
                )
                context_ff_output = F.gelu(context_ff_output, approximate="tanh")

                context_ff_output = self.ff_context.net[1](context_ff_output)
                context_ff_output = self.ff_context.net[2](context_ff_output)

            encoder_hidden_states = (
                encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            )

        return encoder_hidden_states, hidden_states


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


class AlterJointAttnProcessor2_0(JointAttnProcessor2_0):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
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
        *args,
        **kwargs,
    ) -> torch.FloatTensor:

        residual = hidden_states
        n_compute_token = hidden_states.shape[1]

        batch_size = hidden_states.shape[0]

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
        rcd.start()
        key = attn.to_k(hidden_states)
        rcd.end()
        rcd.start()
        value = attn.to_v(hidden_states)
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

        # `context` projections.
        if encoder_hidden_states is not None:

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

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # print((quantizeAdaptivePerTensorQuantize(query, 64) == query).all())

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
            # rcd.start()
            h = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
            # rcd.end()

        h = apply_topk_attention_pruning(h)

        if enableSparseAttn:
            # stat.append(sparseAttnMask.sum() / sparseAttnMask.numel())
            # print(sum(stat) / len(stat))
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
            attnMask[..., :n_compute_token, :n_compute_token] = torch.where(
                sparseAttnMaskUnpacked, -torch.inf, 0
            )
            h = h + attnMask

        s = h.softmax(dim=-1)

        record_sparse_profile(
            sparseProf,
            idxTimestep,
            idxBlock,
            nameAttnBlock or ("attn_joint" if encoder_hidden_states is not None else "attn_self"),
            s[..., :n_compute_token, :n_compute_token],
            s.shape[-1],
        )

        hidden_states = torch.matmul(s, value)

        # For evaluating SAR on GPU

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
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, :n_compute_token],
                hidden_states[:, n_compute_token:],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

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

        # hidden_states = token_unmerge(hidden_states, 64, unmerge_idx)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def apply_alter(
    pipeline: StableDiffusion3Pipeline,
):
    pipeline.__class__ = AlterStableDiffusion3Pipeline
    diffusion_model = pipeline.transformer
    diffusion_model.__class__ = AlterSD3Transformer2DModel

    processor = AlterJointAttnProcessor2_0()
    block_cnt = 0
    for _, module in diffusion_model.named_modules():
        if module.__class__.__name__ == "JointTransformerBlock":
            module.__class__ = AlterJointTransformerBlock
            module.attn.__class__ = AlterAttention
            module.attn.set_processor(processor)
            if module.attn2 is not None:
                module.attn2.__class__ = AlterAttention
                module.attn2.set_processor(processor)
            module.idx_block = block_cnt
            block_cnt += 1

    return pipeline


def preprocess(prompt):
    return prompt


def postprocess(output, path, name):
    output.images[0].save(f"{path}{name}")

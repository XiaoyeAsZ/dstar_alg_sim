from diffusers.models.attention_processor import AttnProcessor
import torch
from diffusers.models import AutoencoderKL, PixArtTransformer2DModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers import PixArtSigmaPipeline
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import *
from diffusers.models.transformers.pixart_transformer_2d import *
from diffusers.models.attention import *
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.models.attention_processor import *

from util.trace import *
from util.sparse import apply_topk_attention_pruning, record_sparse_profile

from util.stat import time_recorder

rcd = time_recorder()

stat = []


class AlterPixArtSigmaPipeline(PixArtSigmaPipeline):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKL,
        transformer: PixArtTransformer2DModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
        max_sequence_length: int = 300,
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
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:

        # 1. Check inputs. Raise error if not correct
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        if use_resolution_binning:
            if self.transformer.config.sample_size == 256:
                aspect_ratio_bin = ASPECT_RATIO_2048_BIN
            elif self.transformer.config.sample_size == 128:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            elif self.transformer.config.sample_size == 64:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            elif self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_256_BIN
            else:
                raise ValueError("Invalid sample size")
            orig_height, orig_width = height, width
            height, width = self.image_processor.classify_height_width_bin(
                height, width, ratios=aspect_ratio_bin
            )

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

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
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [negative_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        # 7. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )

        # Init cache
        b = batch_size * (2 if do_classifier_free_guidance else 1)
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
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=current_timestep,
                    added_cond_kwargs=added_cond_kwargs,
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
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                else:
                    noise_pred = noise_pred

                # compute previous image: x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                # image = self.vae.decode(
                #     latents / self.vae.config.scaling_factor, return_dict=False
                # )[0]
                # image = self.image_processor.postprocess(image, output_type=output_type)
                # image[0].save(f"/home/czhang/dit/pic/show/{i}.png")

        print(rcd.time)

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            if use_resolution_binning:
                image = self.image_processor.resize_and_crop_tensor(
                    image, orig_width, orig_height
                )
        else:
            image = latents

        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


class AlterPixArtTransformer2DModel(PixArtTransformer2DModel):
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 72,
        in_channels: int = 4,
        out_channels: int | None = 8,
        num_layers: int = 28,
        dropout: float = 0,
        norm_num_groups: int = 32,
        cross_attention_dim: int | None = 1152,
        attention_bias: bool = True,
        sample_size: int = 128,
        patch_size: int = 2,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: int | None = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm_single",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 0.000001,
        interpolation_scale: int | None = None,
        use_additional_conditions: bool | None = None,
        caption_channels: int | None = None,
        attention_type: str | None = "default",
    ):
        super().__init__(
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            num_layers,
            dropout,
            norm_num_groups,
            cross_attention_dim,
            attention_bias,
            sample_size,
            patch_size,
            activation_fn,
            num_embeds_ada_norm,
            upcast_attention,
            norm_type,
            norm_elementwise_affine,
            norm_eps,
            interpolation_scale,
            use_additional_conditions,
            caption_channels,
            attention_type,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
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

        if self.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError(
                "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
            )

        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.config.patch_size,
            hidden_states.shape[-1] // self.config.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)

        timestep, embedded_timestep = self.adaln_single(
            timestep,
            added_cond_kwargs,
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        # 2. Blocks
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
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    None,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=None,
                    block_idx=index_block,
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

        # 3. Output
        shift, scale = (
            self.scale_shift_table[None]
            + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(
            hidden_states.device
        )
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
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
        block_idx: int = None,
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
            block_idx=block_idx,
            enableTraceGen=enableTraceGen,
            tracePath=tracePath,
            enableAttnCache=enableAttnCache,
            flushAttnCache=flushAttnCache,
            qCache=(cacheAttn["qCache"] if cacheAttn else None),
            kCache=(cacheAttn["kCache"] if cacheAttn else None),
            enableSparseAttn=enableSparseAttn,
            sparseAttnMask=(sparseAttnMask["attn_self"] if sparseAttnMask else None),
            idxTimestep=idxTimestep,
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
        block_idx: int = None,
        enableTraceGen: bool = False,
        tracePath: str = None,
        enableAttnCache: bool = False,
        flushAttnCache: bool = False,
        qCache: torch.Tensor = None,
        kCache: torch.Tensor = None,
        enableSparseAttn: bool = False,
        sparseAttnMask: torch.Tensor = None,
        idxTimestep: int = None,
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
            block_idx=block_idx,
            enableTraceGen=enableTraceGen,
            tracePath=tracePath,
            enableAttnCache=enableAttnCache,
            flushAttnCache=flushAttnCache,
            qCache=qCache,
            kCache=kCache,
            enableSparseAttn=enableSparseAttn,
            sparseAttnMask=sparseAttnMask,
            idxTimestep=idxTimestep,
            sparseProf=sparseProf,
            **cross_attention_kwargs,
        )


class AlterPixartAttnProcessor(AttnProcessor2_0):
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
        block_idx: int = None,
        enableTraceGen: bool = False,
        tracePath: str = None,
        enableAttnCache: bool = False,
        flushAttnCache: bool = False,
        qCache: torch.Tensor = None,
        kCache: torch.Tensor = None,
        enableSparseAttn: bool = False,
        sparseAttnMask: torch.Tensor = None,
        idxTimestep: int = None,
        sparseProf: list[list[dict]] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        attn_type = "self" if encoder_hidden_states is None else "cross"

        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

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

        # h = torch.matmul(query / math.sqrt(head_dim), key.transpose(-1, -2))

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

        h = apply_topk_attention_pruning(h)

        assert not ((attention_mask is not None) and (enableSparseAttn))
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
            attnMask[..., : hidden_states.shape[1], : hidden_states.shape[1]] = (
                torch.where(sparseAttnMaskUnpacked, -torch.inf, 0)
            )
            h = h + attnMask

        if attention_mask is not None:
            h = h + attention_mask

        s = h.softmax(-1)

        if attn_type == "self":
            record_sparse_profile(
                sparseProf,
                idxTimestep,
                block_idx,
                "attn_self",
                s[..., : hidden_states.shape[1], : hidden_states.shape[1]],
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

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def apply_alter(
    pipeline: PixArtSigmaPipeline,
):
    pipeline.__class__ = AlterPixArtSigmaPipeline
    diffusion_model = pipeline.transformer
    diffusion_model.__class__ = AlterPixArtTransformer2DModel

    processor = AlterPixartAttnProcessor()
    block_cnt = 0
    for _, module in diffusion_model.named_modules():
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
    output.images[0].save(f"{path}/{name}")

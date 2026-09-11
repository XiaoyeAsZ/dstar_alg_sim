import torch

from diffusers.pipelines.dit.pipeline_dit import *
from diffusers.models.transformers.transformer_2d import *
from diffusers.models.attention_processor import *
from typing import Any, Callable, Dict, List

from util.trace import *
from util.sparse import apply_topk_attention_pruning, record_sparse_profile

from util.stat import time_recorder

rcd = time_recorder()

stat = []


class AlterDiTPipeline(DiTPipeline):
    def __init__(self, transformer, vae, scheduler, id2label=None):
        super().__init__(transformer, vae, scheduler, id2label)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
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
    ) -> Union[ImagePipelineOutput, Tuple]:

        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        class_labels = torch.tensor(
            class_labels, device=self._execution_device
        ).reshape(-1)
        class_null = torch.tensor([1000] * batch_size, device=self._execution_device)
        class_labels_input = (
            torch.cat([class_labels, class_null], 0)
            if guidance_scale > 1
            else class_labels
        )

        # Init cache
        b = batch_size * (2 if guidance_scale > 1 else 1)
        h = self.transformer.config["num_attention_heads"]
        patchSize = self.transformer.config["patch_size"]
        n = (latents.shape[-2] // patchSize) * (latents.shape[-1] // patchSize)
        k = self.transformer.config["attention_head_dim"]

        if enableDiffInfer or enableSparseReuse:
            cacheDiff = [{} for _ in range(len(self.transformer.transformer_blocks))]
        else:
            cacheDiff = [None for _ in range(len(self.transformer.transformer_blocks))]

        if enableAttnCache:
            cacheAttn = [{} for _ in range(len(self.transformer.transformer_blocks))]
            for IndexBlock in range(len(self.transformer.transformer_blocks)):
                cacheAttn[IndexBlock]["qCache"] = torch.empty(
                    b, h, n, k, device=self._execution_device, dtype=latents.dtype
                )
                cacheAttn[IndexBlock]["kCache"] = torch.empty(
                    b, h, n, k, device=self._execution_device, dtype=latents.dtype
                )
        else:
            cacheAttn = [None for _ in range(len(self.transformer.transformer_blocks))]

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        i = 0
        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = torch.cat([half, half], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            timesteps = t
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(timesteps, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor(
                    [timesteps], dtype=dtype, device=latent_model_input.device
                )
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(latent_model_input.shape[0])
            # predict noise model_output
            noise_pred = self.transformer(
                latent_model_input,
                timestep=timesteps,
                class_labels=class_labels_input,
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
            ).sample

            # perform guidance
            if guidance_scale > 1:
                eps, rest = (
                    noise_pred[:, :latent_channels],
                    noise_pred[:, latent_channels:],
                )
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                noise_pred = torch.cat([eps, rest], dim=1)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.scheduler.step(
                model_output, t, latent_model_input
            ).prev_sample

            i += 1

        print(rcd.time)

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents = latent_model_input

        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample

        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)


class AlterDiTTransformer2DModel(DiTTransformer2DModel):
    def __init__(
        self,
        num_attention_heads=16,
        attention_head_dim=72,
        in_channels=4,
        out_channels=None,
        num_layers=28,
        dropout=0,
        norm_num_groups=32,
        attention_bias=True,
        sample_size=32,
        patch_size=2,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        upcast_attention=False,
        norm_type="ada_norm_zero",
        norm_elementwise_affine=False,
        norm_eps=0.00001,
    ):
        super().__init__(
            num_attention_heads,
            attention_head_dim,
            in_channels,
            out_channels,
            num_layers,
            dropout,
            norm_num_groups,
            attention_bias,
            sample_size,
            patch_size,
            activation_fn,
            num_embeds_ada_norm,
            upcast_attention,
            norm_type,
            norm_elementwise_affine,
            norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
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
        # 1. Input
        height, width = (
            hidden_states.shape[-2] // self.patch_size,
            hidden_states.shape[-1] // self.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)

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
                    None,
                    None,
                    None,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
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
                    scale=(256.0 if index_block < 3 else 64.0),
                )

        # 3. Output
        conditioning = self.transformer_blocks[0].norm1.emb(
            timestep, class_labels, hidden_dtype=hidden_states.dtype
        )
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        hidden_states = (
            self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        )
        hidden_states = self.proj_out_2(hidden_states)

        # unpatchify
        height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            shape=(
                -1,
                height,
                width,
                self.patch_size,
                self.patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                -1,
                self.out_channels,
                height * self.patch_size,
                width * self.patch_size,
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
        # print(hidden_states.shape)

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

        if name_attn_block == "attn_self":
            record_sparse_profile(
                sparseProf,
                idxTimestep,
                idxBlock,
                name_attn_block,
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
    pipeline: DiTPipeline,
):
    pipeline.__class__ = AlterDiTPipeline
    diffusion_model = pipeline.transformer
    diffusion_model.__class__ = AlterDiTTransformer2DModel

    processor = AlterAttnProcessor2_0()
    block_cnt = 0
    for _, module in diffusion_model.named_modules():
        if module.__class__.__name__ == "BasicTransformerBlock":
            module.__class__ = AlterBasicTransformerBlock
            module.attn1.__class__ = AlterAttention
            module.attn1.set_processor(processor)
            module.idx_block = block_cnt
            block_cnt += 1

    return pipeline


def preprocess(prompt):
    if isinstance(prompt, str):
        return int(prompt)
    else:
        return prompt


def postprocess(output, path, name):
    output.images[0].save(f"{path}/{name}")

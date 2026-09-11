def load_pipe(models: str, config: dict | None = None):
    import torch
    config = config or {}
    local_files_only = config.get("localFilesOnly", config.get("local_files_only", False))

    match models:
        case "dit-xl-512":
            from diffusers import DiTPipeline

            pipe = DiTPipeline.from_pretrained(
                "facebook/DiT-XL-2-512",
                local_files_only=local_files_only,
            ).to("cuda")

            from models.dit_xl import apply_alter, preprocess, postprocess

            apply_func = apply_alter
            pre_func = preprocess
            post_func = postprocess

        case "pixart-sigma-1024":
            from diffusers import PixArtSigmaPipeline

            pipe = PixArtSigmaPipeline.from_pretrained(
                "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
                torch_dtype=torch.float16,
                use_safetensors=True,
                local_files_only=local_files_only,
            ).to("cuda")

            from models.pixart import apply_alter, preprocess, postprocess

            apply_func = apply_alter
            pre_func = preprocess
            post_func = postprocess

        case "flux.1-dev":
            from diffusers import FluxPipeline

            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16,
                local_files_only=local_files_only,
            ).to("cuda")

            from models.flux import apply_alter, preprocess, postprocess

            apply_func = apply_alter
            pre_func = preprocess
            post_func = postprocess

        case "flux.1-schnell":
            from diffusers import FluxPipeline

            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=torch.bfloat16,
                local_files_only=local_files_only,
            ).to("cuda")

            from models.flux import apply_alter, preprocess, postprocess

            apply_func = apply_alter
            pre_func = preprocess
            post_func = postprocess

        case "sd3.5-medium":
            from diffusers import StableDiffusion3Pipeline

            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-medium",
                torch_dtype=torch.bfloat16,
                local_files_only=local_files_only,
            ).to("cuda")

            from models.sd3 import apply_alter, preprocess, postprocess

            apply_func = apply_alter
            pre_func = preprocess
            post_func = postprocess

        case "sd3.5-large":
            from diffusers import StableDiffusion3Pipeline

            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3.5-large",
                torch_dtype=torch.bfloat16,
                local_files_only=local_files_only,
            ).to("cuda")

            from models.sd3 import apply_alter, preprocess, postprocess

            apply_func = apply_alter
            pre_func = preprocess
            post_func = postprocess

        case "latte":
            from diffusers import LattePipeline
            from diffusers.models import AutoencoderKLTemporalDecoder

            pipe = LattePipeline.from_pretrained(
                "maxin-cn/Latte-1",
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
            ).to("cuda")
            vae = AutoencoderKLTemporalDecoder.from_pretrained(
                "maxin-cn/Latte-1",
                subfolder="vae_temporal_decoder",
                torch_dtype=torch.float16,
                local_files_only=local_files_only,
            ).to("cuda")
            pipe.vae = vae
            from models.latte import apply_alter, preprocess, postprocess

            apply_func = apply_alter
            pre_func = preprocess
            post_func = postprocess

        case _:
            raise RuntimeError("No such models.")
    return pipe, apply_func, pre_func, post_func

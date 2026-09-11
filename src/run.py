import copy
import json
import os
import re
import warnings
from pathlib import Path


def _trace_path(config):
    value = config.get("tracePath")
    if value in (None, "", "None", "none"):
        return None
    return f"{str(value).rstrip(os.sep)}{os.sep}"


def _is_writable_dir(path):
    try:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_test"
        probe.write_text("")
        probe.unlink()
        return True
    except OSError:
        return False


def _flag(config, name, default=False):
    snake_name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return config.get(name, config.get(snake_name, default))


def _attention_prune_mode(config, profile=False):
    from util.sparse import normalize_attention_prune_mode

    mode = None
    if profile:
        mode = config.get(
            "profileAttentionPruneMode",
            config.get("profile_attention_prune_mode"),
        )
        if mode is None and _flag(config, "profileEnableSparseAttn", False):
            mode = "sar"
    else:
        mode = config.get("attentionPruneMode", config.get("attention_prune_mode"))
        if mode is None and _flag(config, "enableSparseAttn", False):
            mode = "sar"

    return normalize_attention_prune_mode(mode)


def _num_blocks(pipe):
    return len(pipe.transformer.transformer_blocks) + len(
        getattr(pipe.transformer, "single_transformer_blocks", [])
    )


def _num_steps(config):
    timesteps = config.get("timesteps", config.get("profileTimesteps"))
    if timesteps is not None:
        return len(timesteps)
    return config["num_inference_steps"]


def _empty_sparse_mask(pipe, config):
    return [[None for _ in range(_num_blocks(pipe))] for _ in range(_num_steps(config))]


def _size_spec(config):
    if (
        config.get("height") is not None
        and config["model"].startswith("flux.1-dev")
    ):
        return f"-{config['height']}"
    return ""


def _mask_stem(config):
    return config.get(
        "sparseProfileName",
        config.get("profileName", f"{config['model']}{_size_spec(config)}"),
    )


def _move_nested_tensors(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_move_nested_tensors(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_nested_tensors(item, device) for key, item in value.items()}
    return value


def _load_sparse_mask(pipe, config):
    import torch

    if _attention_prune_mode(config) != "sar":
        return _empty_sparse_mask(pipe, config)

    mask_dir = Path(config["sparseAttnMaskPath"])
    mask_path = mask_dir / f"{_mask_stem(config)}.pth"
    sparse_mask = torch.load(mask_path, map_location="cpu")
    return _move_nested_tensors(sparse_mask, "cuda")


def _read_prompts(config, pre):
    task = config.get("profileTask", config["task"])

    if config.get("prompts") is not None:
        raw_prompts = config["prompts"]
    elif config.get("prompt") is not None:
        raw_prompts = [config["prompt"]]
        if config.get("caption_path") is not None:
            warnings.warn(
                "Both prompt and caption_path are not None, caption_path is ignored."
            )
    elif config.get("caption_path") is not None:
        with open(config["caption_path"]) as cap:
            captions = json.load(cap)
        n_sample = config.get("n_sample")
        if isinstance(captions, dict) and "captions" in captions:
            raw_prompts = captions["captions"]
        elif isinstance(captions, dict):
            raw_prompts = list(captions.values())
        else:
            raw_prompts = captions
        raw_prompts = raw_prompts[:n_sample]
    elif task == "c2i":
        raw_prompts = list(range(config.get("n_sample", 100)))
    else:
        raise Exception("No prompt, prompts, or caption_path is specified.")

    return [pre(prompt) for prompt in raw_prompts]


def _seeds(config):
    if config.get("seeds") is not None:
        return [int(seed) for seed in config["seeds"]]
    if config.get("seedStart") is not None:
        n_seeds = int(config.get("nSeeds", 1))
        return [int(config["seedStart"]) + i for i in range(n_seeds)]
    return [int(config.get("seed", 0))]


def _pipe_kwargs(config, sparse_attn_mask, sparse_prof=None, profile=False):
    from util.sparse import set_attention_prune_config
    from diff.diff import set_diff_quant_config

    attention_prune_mode = _attention_prune_mode(config, profile=profile)
    set_attention_prune_config(
        attention_prune_mode,
        config.get("topkKeepRatio", config.get("topk_keep_ratio", 0.8)),
    )
    set_diff_quant_config(
        config.get("diffQuantMode", config.get("diff_quant_mode", "adapt")),
        config.get("diffQuantGroupSize", config.get("diff_quant_group_size", [1, 128])),
    )

    kwargs = {
        "num_inference_steps": config["num_inference_steps"],
        "enableTraceGen": _flag(config, "enableTraceGen", False),
        "tracePath": _trace_path(config),
        "enableDiffInfer": _flag(config, "enableDiffInfer", False),
        "enableSparseReuse": _flag(config, "enableSparseReuse", False),
        "sparseReuseThreshold": float(
            config.get("sparseReuseThreshold", config.get("sparse_reuse_threshold", 0.01))
        ),
        "enableAttnCache": _flag(config, "enableAttnCache", False),
        "enableSparseAttn": attention_prune_mode == "sar",
        "sparseAttnMask": sparse_attn_mask,
    }

    if profile:
        kwargs.update(
            {
                "enableTraceGen": _flag(config, "profileEnableTraceGen", False),
                "tracePath": _trace_path(config),
                "enableDiffInfer": _flag(config, "profileEnableDiffInfer", False),
                "enableSparseReuse": _flag(config, "profileEnableSparseReuse", False),
                "enableAttnCache": _flag(config, "profileEnableAttnCache", False),
                "enableSparseAttn": attention_prune_mode == "sar",
                "sparseProf": sparse_prof,
            }
        )

    timesteps = config.get("timesteps", config.get("profileTimesteps"))
    if timesteps is not None and config.get("profileTask", config["task"]) != "c2i":
        kwargs["timesteps"] = timesteps

    return kwargs


def _run_pipe(pipe, config, task, prompt, seed, sparse_attn_mask, sparse_prof=None):
    import torch

    profile = sparse_prof is not None
    kwargs = _pipe_kwargs(config, sparse_attn_mask, sparse_prof, profile=profile)
    kwargs["generator"] = torch.manual_seed(seed)

    if task == "c2i":
        return pipe([prompt], **kwargs)
    if task == "t2i":
        return pipe(
            prompt,
            height=config.get("height"),
            width=config.get("width"),
            output_type=config.get("profileOutputType", "latent") if profile else "pil",
            **kwargs,
        )
    if task == "t2v":
        return pipe(
            prompt,
            video_length=config.get("video_length", 16),
            output_type=config.get("profileOutputType", "latents") if profile else "pt",
            **kwargs,
        )
    raise Exception(f"Unsupported task <{task}>.")


def _safe_prompt_name(prompt):
    return " ".join(re.sub(r"[^a-zA-Z0-9]", " ", str(prompt)).rstrip().split(" "))


def _ensure_output_dir(config):
    out_path = config["out_path"]
    os.makedirs(out_path, exist_ok=True)
    return out_path if out_path.endswith("/") else f"{out_path}/"


def _resolve_config_paths(config):
    """Resolve artifact-relative paths independently of the caller's cwd."""
    root = Path(__file__).resolve().parents[1]
    path_keys = {
        "caption_path", "out_path", "sparseAttnMaskPath", "sparse_attn_mask_path",
        "tracePath", "trace_path", "hfHome", "hf_home",
    }
    for key in path_keys:
        value = config.get(key)
        if isinstance(value, str) and value not in {"", "None", "none"}:
            path = Path(value).expanduser()
            if not path.is_absolute():
                config[key] = str(root / path)
    return config


def _run_generate(pipe, config, pre, post):
    task = config["task"]
    prompts = _read_prompts(config, pre)
    if _flag(config, "enableTraceGen", False):
        trace_path = _trace_path(config)
        if trace_path:
            Path(trace_path).mkdir(parents=True, exist_ok=True)
    sparse_attn_mask = _load_sparse_mask(pipe, config)
    out_path = _ensure_output_dir(config)

    if task == "c2i":
        n_sample_per_prompt = 1
        for prompt in prompts:
            for sample_idx in range(n_sample_per_prompt):
                seed = config["seed"] if config.get("prompt") is not None else sample_idx
                output = _run_pipe(pipe, config, task, prompt, seed, sparse_attn_mask)
                post(output, out_path, f"{prompt}-{sample_idx}.png")
        return

    seed = int(config.get("seed", 0))
    suffix = ".png" if task == "t2i" else ".mp4"
    for prompt in prompts:
        output = _run_pipe(pipe, config, task, prompt, seed, sparse_attn_mask)
        post(output, out_path, f"{_safe_prompt_name(prompt)}_000000{suffix}")


def _build_sparse_mask(sparse_prof, threshold, selected_steps=None):
    import torch

    sparse_mask = [[{} for _ in step] for step in sparse_prof]
    summary_entries = []
    ratios = []
    selected_ratios = []
    selected_steps = set(selected_steps or [])

    for step_idx, step in enumerate(sparse_prof):
        for block_idx, block in enumerate(step):
            for attn_name, attn_prof in block.items():
                mask = (attn_prof["attn_map"] / attn_prof["n_token"] < threshold).bool()
                sparse_mask[step_idx][block_idx][attn_name] = mask.cpu()
                ratio = float(mask.float().mean().item())
                ratios.append(ratio)
                if not selected_steps or step_idx in selected_steps:
                    selected_ratios.append(ratio)
                    summary_entries.append(
                        {
                            "step": step_idx,
                            "block": block_idx,
                            "attention": attn_name,
                            "n_sample": int(attn_prof["n_sample"]),
                            "n_token": int(attn_prof["n_token"]),
                            "mask_shape": list(mask.shape),
                            "prune_ratio": ratio,
                        }
                    )

    return sparse_mask, {
        "mean_prune_ratio": float(sum(ratios) / len(ratios)) if ratios else 0.0,
        "selected_mean_prune_ratio": (
            float(sum(selected_ratios) / len(selected_ratios))
            if selected_ratios
            else 0.0
        ),
        "entries": summary_entries,
    }


def _run_sparse_profile(pipe, config, pre):
    import torch

    task = config.get("profileTask", "t2i")
    prompts = _read_prompts(config, pre)
    seeds = _seeds(config)
    n_blocks = _num_blocks(pipe)
    n_steps = _num_steps(config)
    sparse_prof = [[{} for _ in range(n_blocks)] for _ in range(n_steps)]
    empty_sparse_mask = _empty_sparse_mask(pipe, config)

    for prompt_idx, prompt in enumerate(prompts):
        for seed in seeds:
            print(
                f"profile prompt {prompt_idx + 1}/{len(prompts)}, seed {seed}, "
                f"steps {n_steps}"
            )
            _run_pipe(
                pipe,
                config,
                task,
                prompt,
                seed,
                empty_sparse_mask,
                sparse_prof=sparse_prof,
            )
            torch.cuda.empty_cache()

    threshold = float(config.get("sparseProfileThreshold", 1e-4))
    selected_steps = config.get("profileTimestepIndices")
    sparse_mask, summary = _build_sparse_mask(sparse_prof, threshold, selected_steps)

    out_dir = Path(
        config.get(
            "sparseAttnMaskPath",
            Path(__file__).resolve().parents[1] / "mask",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _mask_stem(config)
    mask_path = out_dir / f"{stem}.pth"
    summary_path = out_dir / f"{stem}.summary.json"
    torch.save(sparse_mask, mask_path)

    if config.get("sparseProfileSaveRaw", False):
        torch.save(sparse_prof, out_dir / f"{stem}.profile.pth")

    summary.update(
        {
            "model": config["model"],
            "profile_task": task,
            "num_prompts": len(prompts),
            "seeds": seeds,
            "num_inference_steps": config["num_inference_steps"],
            "timesteps": config.get("timesteps", config.get("profileTimesteps")),
            "profile_timestep_indices": selected_steps,
            "threshold": threshold,
            "mask_path": str(mask_path),
        }
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"saved sparse mask: {mask_path}")
    print(f"saved summary: {summary_path}")


def _expanded_profile_configs(config):
    sweeps = config.get("profileSweeps")
    if not sweeps:
        return [config]

    expanded = []
    for idx, sweep in enumerate(sweeps):
        sweep_config = copy.deepcopy(config)
        sweep_config.pop("profileSweeps", None)
        sweep_config.update(sweep)
        if "sparseProfileName" not in sweep:
            sweep_config["sparseProfileName"] = sweep.get(
                "profileName", f"{config['model']}-profile-{idx}"
            )
        expanded.append(sweep_config)
    return expanded


def _expanded_run_configs(config):
    sweeps = config.get("runSweeps")
    if not sweeps:
        return [config]

    expanded = []
    for sweep in sweeps:
        sweep_config = copy.deepcopy(config)
        sweep_config.pop("runSweeps", None)
        sweep_config.update(sweep)
        expanded.append(sweep_config)
    return expanded


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as cfg:
        config = json.load(cfg)
    config = _resolve_config_paths(config)

    os.environ["CUDA_VISIBLE_DEVICES"] = config["device"]
    hf_home = config.get(
        "hfHome",
        os.environ.get(
            "DSTAR_HF_CACHE",
            "/state/partition/czhang/hf_cache",
        ),
    )
    if not _is_writable_dir(hf_home):
        raise OSError(f"Configured hfHome is not writable: {hf_home}")
    os.environ["HUGGINGFACE_HUB_CACHE"] = hf_home
    os.environ["HF_HUB_CACHE"] = hf_home
    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = hf_home

    from models.load_pipe import load_pipe

    pipe, apply, pre, post = load_pipe(config["model"], config)
    pipe = apply(pipe)

    match config["task"]:
        case "c2i" | "t2i" | "t2v":
            for run_config in _expanded_run_configs(config):
                _run_generate(pipe, run_config, pre, post)
        case "sparse_profile" | "profile_sparse" | "prof":
            for profile_config in _expanded_profile_configs(config):
                _run_sparse_profile(pipe, profile_config, pre)
        case _:
            raise Exception("No task is specified.")

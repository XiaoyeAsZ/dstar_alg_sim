"""Public orchestration API for the combined DSTAR artifact."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

try:
    from .evaluation.quality import evaluate_generated_quality
except ImportError:  # Allow execution with the artifact directory on PYTHONPATH.
    from evaluation.quality import evaluate_generated_quality


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_config(
    config: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(config, Mapping):
        return dict(config), None
    path = Path(config).expanduser().resolve()
    with path.open() as handle:
        return json.load(handle), path


def _artifact_path(value: Any, base: Path) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith("/home/czhang/adapt/"):
        return str(ROOT / value.removeprefix("/home/czhang/adapt/"))
    if value.startswith("~/"):
        return str(Path(value).expanduser())
    if value in {"None", "none", ""}:
        return value
    path = Path(value)
    return str((base / path).resolve()) if not path.is_absolute() else value


def _prepare_config(
    config: str | Path | Mapping[str, Any], updates: Mapping[str, Any]
) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    data, source = _load_config(config)
    base = (
        ROOT
        if source and source.is_relative_to(ROOT)
        else source.parent if source else ROOT
    )
    path_keys = {
        "caption_path",
        "out_path",
        "sparseAttnMaskPath",
        "sparse_attn_mask_path",
        "tracePath",
        "trace_path",
        "hfHome",
        "hf_home",
        "HUGGINGFACE_HUB_CACHE",
    }
    data = {
        key: _artifact_path(value, base) if key in path_keys else value
        for key, value in data.items()
    }
    data.update(updates)
    data = {
        key: _artifact_path(value, ROOT) if key in path_keys else value
        for key, value in data.items()
    }
    temp_dir = tempfile.TemporaryDirectory(prefix="dstar-config-", dir=ROOT / "outputs")
    path = Path(temp_dir.name) / "config.json"
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path, temp_dir


def _run_engine(
    config: str | Path | Mapping[str, Any], updates: Mapping[str, Any] | None = None
) -> subprocess.CompletedProcess[str]:
    path, temp_dir = _prepare_config(config, updates or {})
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), env.get("PYTHONPATH", "")])
    env["HF_HUB_CACHE"] = env.get(
        "DSTAR_HF_CACHE", "/state/partition/czhang/hf_cache"
    )
    try:
        return subprocess.run(
            [sys.executable, str(SRC / "run.py"), "--config", str(path)],
            cwd=ROOT,
            env=env,
            text=True,
            check=True,
        )
    finally:
        temp_dir.cleanup()


def profile_mask(
    config: str | Path | Mapping[str, Any], **overrides: Any
) -> subprocess.CompletedProcess[str]:
    """Run the existing sparse-attention profiler and save a ``.pth`` mask."""

    updates = {
        "task": "sparse_profile",
        "sparseAttnMaskPath": str(ROOT / "mask"),
        "enableTraceGen": False,
        "tracePath": "None",
    }
    updates.update(overrides)
    return _run_engine(config, updates)


def run_inference(
    config: str | Path | Mapping[str, Any],
    *,
    mask_path: str | Path | None = None,
    quant_mode: str = "adapt",
    use_mask: bool = True,
    **overrides: Any,
) -> subprocess.CompletedProcess[str]:
    """Run existing inference with adaptive quantization and optional SAR mask."""

    updates: dict[str, Any] = {
        "enableDiffInfer": True,
        "diffQuantMode": quant_mode,
        "enableSparseAttn": bool(use_mask),
    }
    if use_mask:
        updates["attentionPruneMode"] = "sar"
        updates["sparseAttnMaskPath"] = str(
            Path(mask_path).expanduser().parent if mask_path else ROOT / "mask"
        )
        if mask_path:
            updates["sparseProfileName"] = Path(mask_path).stem
    updates.update(overrides)
    return _run_engine(config, updates)


def simulate(
    trace_path: str | Path, log_path: str | Path | None = None
) -> dict[str, Any]:
    """Run the copied SimPy DSTAR simulator over all trace files."""

    import torch
    from src.simulator.dstar.simulator import Simulator, dump_log

    trace_dir = Path(trace_path).expanduser()
    files = sorted(trace_dir.glob("*.trace"))
    simulator = Simulator()
    log_destination = Path(log_path).expanduser() if log_path else None
    for trace_file in files:
        trace = torch.load(trace_file, map_location="cpu")
        if trace["layer"] == "attention":
            log = simulator.attention(
                trace["shape"], trace["attn_mask"], trace["reuse"]
            )
        elif trace["layer"] == "linear":
            log = simulator.linear(
                trace["shape"], trace["nbit_map"], trace["diff"], trace["gelu"]
            )
        else:
            raise ValueError(f"Unsupported trace layer: {trace['layer']}")
        if log_destination:
            log_destination.parent.mkdir(parents=True, exist_ok=True)
            dump_log(str(log_destination), log)
    return {
        "trace_count": len(files),
        "cycles": simulator.env.now,
        "ops_giga": simulator.ops,
        "dram_access_giga": simulator.dram_access,
        "scratchpad_access_giga": simulator.scratchpad_access,
        "local_sram_access_giga": simulator.local_sram_access,
        "cycles_diffq": simulator.cycles_diffq,
        "cycles_mpmma": simulator.cycles_mpmma,
        "cycles_rdiff": simulator.cycles_rdiff,
    }


__all__ = ["profile_mask", "run_inference", "evaluate_generated_quality", "simulate"]

"""Self-contained generated-quality evaluation for the DSTAR artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _video_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)


def _image_stats(files: list[Path]) -> dict[str, Any]:
    if not files:
        return {"count": 0, "mean_rgb": None, "std_rgb": None, "mean_sharpness": None}

    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    sharpness: list[float] = []
    sizes: dict[str, int] = {}
    skipped = 0
    for path in files:
        try:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                size_key = f"{image.width}x{image.height}"
        except (OSError, ValueError):
            skipped += 1
            continue
        sizes[size_key] = sizes.get(size_key, 0) + 1
        means.append(array.mean(axis=(0, 1)))
        stds.append(array.std(axis=(0, 1)))
        gray = array.mean(axis=-1)
        # Variance of the discrete Laplacian is a useful sharpness proxy and
        # does not require OpenCV or a pretrained model.
        laplacian = (
            -4.0 * gray
            + np.roll(gray, 1, axis=0)
            + np.roll(gray, -1, axis=0)
            + np.roll(gray, 1, axis=1)
            + np.roll(gray, -1, axis=1)
        )
        sharpness.append(float(laplacian.var()))

    return {
        "count": len(files) - skipped,
        "skipped": skipped,
        "mean_rgb": np.mean(means, axis=0).round(6).tolist(),
        "std_rgb": np.mean(stds, axis=0).round(6).tolist(),
        "mean_sharpness": float(np.mean(sharpness)),
        "image_sizes": sizes,
    }


def _video_stats(files: list[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {"count": len(files), "total_bytes": 0, "files": []}
    for path in files:
        size = path.stat().st_size
        result["total_bytes"] += size
        result["files"].append({"path": str(path), "bytes": size})
    return result


def _reference_psnr(generated: list[Path], reference: Path) -> float | None:
    scores: list[float] = []
    for generated_path in generated:
        reference_path = reference / generated_path.name
        if not reference_path.exists():
            continue
        with Image.open(generated_path) as generated_image, Image.open(reference_path) as reference_image:
            lhs = np.asarray(generated_image.convert("RGB").resize(reference_image.size), dtype=np.float32)
            rhs = np.asarray(reference_image.convert("RGB"), dtype=np.float32)
        mse = float(np.mean((lhs - rhs) ** 2))
        scores.append(float("inf") if mse == 0 else 10.0 * math.log10((255.0**2) / mse))
    return float(np.mean(scores)) if scores else None


def evaluate_generated_quality(
    output_dir: str | Path,
    reference_dir: str | Path | None = None,
    json_path: str | Path | None = None,
    metrics: list[str] | None = None,
    batch_size: int = 16,
    splits: int = 10,
) -> dict[str, Any]:
    """Evaluate generated files with CLIP/FID/FVD/IS and descriptive statistics."""

    output = Path(output_dir).expanduser().resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"Generated output directory does not exist: {output}")
    images = _image_files(output)
    videos = _video_files(output)
    reference = Path(reference_dir).expanduser().resolve() if reference_dir else None
    report: dict[str, Any] = {
        "output_dir": str(output),
        "images": _image_stats(images),
        "videos": _video_stats(videos),
        "metrics": {},
    }
    with tempfile.TemporaryDirectory(prefix="dstar-eval-") as temp_dir:
        benchmark_json = Path(temp_dir) / "metrics.json"
        command = [
            sys.executable,
            str(Path(__file__).with_name("benchmarks.py")),
            str(output),
            "--metrics",
            *(metrics or ["auto"]),
            "--batch-size",
            str(batch_size),
            "--splits",
            str(splits),
            "--json",
            str(benchmark_json),
        ]
        if reference:
            command.extend(["--reference-dir", str(reference)])
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(
                "Benchmark evaluation failed:\n"
                + (completed.stdout + completed.stderr).strip()
            )
        benchmark = json.loads(benchmark_json.read_text())
    report["benchmark_device"] = benchmark["device"]
    report["metrics"].update(benchmark["metrics"])
    if reference and images:
        report["metrics"]["psnr"] = _reference_psnr(images, reference)
    if json_path is not None:
        destination = Path(json_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir")
    parser.add_argument("--reference-dir")
    parser.add_argument("--json")
    parser.add_argument("--metrics", nargs="+", default=["auto"], choices=["auto", "clip", "fid", "fvd", "is"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--splits", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(evaluate_generated_quality(args.output_dir, args.reference_dir, args.json, args.metrics, args.batch_size, args.splits), indent=2))


if __name__ == "__main__":
    main()

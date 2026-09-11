"""Self-contained CLIP, FID, FVD, and IS benchmark implementations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import entropy
from torch.nn import functional as F
from torchvision import models, transforms


EVALUATION_ROOT = Path(__file__).resolve().parent
ASSETS = EVALUATION_ROOT / "assets"
VENDOR = EVALUATION_ROOT / "vendor"
sys.path.insert(0, str(VENDOR))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _files(directory: Path, extensions: set[str]) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in extensions)


def _prompt_from_name(path: Path) -> str:
    return re.sub(r"_\d+$", "", path.stem).replace("_", " ").strip()


def clip_score(files: list[Path], device: str, batch_size: int) -> dict:
    import clip

    model, preprocess = clip.load(str(ASSETS / "ViT-B-32.pt"), device=device)
    scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(files), batch_size):
            batch_files = files[start : start + batch_size]
            images = torch.stack([preprocess(Image.open(path).convert("RGB")) for path in batch_files]).to(device)
            text = clip.tokenize([_prompt_from_name(path) for path in batch_files], truncate=True).to(device)
            image_features = model.encode_image(images)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            scores.extend((image_features * text_features).sum(dim=-1).float().cpu().tolist())
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "count": len(scores)}


def _inception_score(probabilities: np.ndarray, splits: int) -> dict:
    effective_splits = min(splits, max(1, len(probabilities) // 2))
    parts = np.array_split(probabilities, effective_splits)
    scores = []
    for part in parts:
        marginal = part.mean(axis=0)
        scores.append(float(np.exp(np.mean([entropy(probability, marginal) for probability in part]))))
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "splits": len(parts)}


def image_inception_score(files: list[Path], device: str, batch_size: int, splits: int) -> dict:
    transform = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    model = models.inception_v3(weights=None, transform_input=False, init_weights=False)
    model.load_state_dict(torch.load(ASSETS / "inception_v3_google.pth", map_location="cpu"))
    model = model.to(device).eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(files), batch_size):
            tensors = []
            for path in files[start : start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            probabilities.append(F.softmax(model(torch.stack(tensors).to(device)), dim=1).cpu().numpy())
    result = _inception_score(np.concatenate(probabilities), splits)
    result["count"] = len(files)
    return result


def coco_fid(output_dir: Path, device: str, batch_size: int) -> dict:
    from T2IBenchmark.feature_extractors.inceptionV3_feature_extractor import InceptionV3FE
    from T2IBenchmark.metrics.fid import FIDStats, frechet_distance

    extractor = InceptionV3FE(torch.device(device))
    preprocess = extractor.get_preprocess_fn()
    files = _files(output_dir, IMAGE_EXTENSIONS)
    features = []
    with torch.no_grad():
        for start in range(0, len(files), batch_size):
            tensors = []
            for path in files[start : start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(preprocess(image))
            features.append(extractor.forward(torch.stack(tensors)).numpy())
    generated_stats = FIDStats.from_features(np.concatenate(features))
    coco_stats = FIDStats.from_npz(str(ASSETS / "MS-COCO_val2014_fid_stats.npz"))
    score = frechet_distance(generated_stats, coco_stats)
    return {"value": float(score), "reference": "MS-COCO val2014"}


def _read_video(path: Path, frames: int = 16, size: int = 224) -> torch.Tensor:
    import cv2

    capture = cv2.VideoCapture(str(path))
    decoded = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        decoded.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not decoded:
        raise RuntimeError(f"Could not decode video: {path}")
    indices = np.linspace(0, len(decoded) - 1, frames).astype(int)
    sampled = [cv2.resize(decoded[index], (size, size), interpolation=cv2.INTER_AREA) for index in indices]
    return torch.from_numpy(np.stack(sampled)).permute(0, 3, 1, 2).float() / 255.0


def video_inception_score(files: list[Path], device: str, batch_size: int, splits: int) -> dict:
    model = models.video.r3d_18(weights=None)
    model.load_state_dict(torch.load(ASSETS / "r3d_18.pth", map_location="cpu"))
    model = model.to(device).eval()
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(files), batch_size):
            batch = torch.stack([_read_video(path, size=112) for path in files[start : start + batch_size]])
            batch = (batch.permute(0, 2, 1, 3, 4) - mean) / std
            probabilities.append(F.softmax(model(batch.to(device)), dim=1).cpu().numpy())
    result = _inception_score(np.concatenate(probabilities), splits)
    result["count"] = len(files)
    return result


def fvd_score(generated: list[Path], reference_dir: Path, device: str) -> dict:
    upstream = Path(__file__).resolve().parent / "upstream/video_metrics"
    sys.path.insert(0, str(upstream))
    from calculate_fvd import calculate_fvd

    reference = _files(reference_dir, VIDEO_EXTENSIONS)
    if not reference:
        raise ValueError(f"No reference videos found in {reference_dir}")
    count = min(len(generated), len(reference))
    generated_tensor = torch.stack([_read_video(path) for path in generated[:count]])
    reference_tensor = torch.stack([_read_video(path) for path in reference[:count]])
    result = calculate_fvd(reference_tensor, generated_tensor, torch.device(device), method="styleganv", only_final=True)
    return {"value": float(result["value"][-1]), "count": count, "reference_dir": str(reference_dir)}


def run(output_dir: Path, metrics: list[str], reference_dir: Path | None, batch_size: int, splits: int) -> dict:
    images = _files(output_dir, IMAGE_EXTENSIONS)
    videos = _files(output_dir, VIDEO_EXTENSIONS)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = set(metrics)
    if "auto" in selected:
        selected = {"clip", "is", "fid"} if images else {"is", "fvd"}
    results = {"device": device, "metrics": {}}
    for metric in sorted(selected):
        try:
            if metric == "clip":
                if not images:
                    raise ValueError("CLIP requires generated images")
                results["metrics"][metric] = clip_score(images, device, batch_size)
            elif metric == "is":
                files = images or videos
                if not files:
                    raise ValueError("IS requires generated images or videos")
                results["metrics"][metric] = (
                    image_inception_score(images, device, batch_size, splits)
                    if images
                    else video_inception_score(videos, device, batch_size, splits)
                )
            elif metric == "fid":
                if not images:
                    raise ValueError("FID requires generated images")
                results["metrics"][metric] = coco_fid(output_dir, device, batch_size)
            elif metric == "fvd":
                if not videos:
                    raise ValueError("FVD requires generated videos")
                if reference_dir is None:
                    raise ValueError("FVD requires --reference-dir")
                results["metrics"][metric] = fvd_score(videos, reference_dir, device)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        except Exception as error:
            results["metrics"][metric] = {"error": f"{type(error).__name__}: {error}"}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--metrics", nargs="+", default=["auto"], choices=["auto", "clip", "fid", "fvd", "is"])
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output_dir.resolve(), args.metrics, args.reference_dir.resolve() if args.reference_dir else None, args.batch_size, args.splits)
    args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

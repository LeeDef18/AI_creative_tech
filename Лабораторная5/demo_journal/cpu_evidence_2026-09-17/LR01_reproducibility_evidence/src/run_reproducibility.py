from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata as md
import json
import os
import platform
import time

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "run_config.json"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    names = [
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface-hub",
        "Pillow",
    ]
    return {name: md.version(name) for name in names}


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    load_args = {
        "revision": config["revision"],
        "use_safetensors": True,
        "torch_dtype": dtype,
    }
    if device == "cuda":
        load_args["variant"] = "fp16"

    model_started = time.perf_counter()
    pipeline = AutoPipelineForText2Image.from_pretrained(
        config["model_id"], **load_args
    ).to(device)
    model_load_seconds = time.perf_counter() - model_started

    common = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": config["model_id"],
        "revision_requested": config["revision"],
        "prompt": config["prompt"],
        "seed": config["seed"],
        "num_inference_steps": config["num_inference_steps"],
        "guidance_scale": config["guidance_scale"],
        "height": config["height"],
        "width": config["width"],
        "device": device,
        "dtype": str(dtype),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "packages": package_versions(),
        "deterministic_algorithms": True,
        "torch_num_threads": torch.get_num_threads(),
        "model_load_seconds_measured": model_load_seconds,
        "hf_home": os.environ.get("HF_HOME"),
    }

    manifests = []
    for run_number in range(1, int(config["runs"]) + 1):
        run_name = f"run_{run_number:03d}"
        run_dir = ARTIFACTS / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device="cpu").manual_seed(config["seed"])
        started = time.perf_counter()
        image = pipeline(
            prompt=config["prompt"],
            num_inference_steps=config["num_inference_steps"],
            guidance_scale=config["guidance_scale"],
            height=config["height"],
            width=config["width"],
            generator=generator,
        ).images[0]
        elapsed = time.perf_counter() - started
        image_path = run_dir / "result.png"
        image.save(image_path)
        with Image.open(image_path) as check:
            observed_size = list(check.size)
            observed_mode = check.mode
        manifest = {
            **common,
            "run": run_number,
            "run_name": run_name,
            "generation_seconds_measured": elapsed,
            "artifact": str(image_path.relative_to(ROOT)),
            "artifact_size_bytes": image_path.stat().st_size,
            "image_size_observed": observed_size,
            "image_mode_observed": observed_mode,
            "sha256": sha256(image_path),
        }
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (REPORTS / f"{run_name}.log").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifests.append(manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)

    same_hash = manifests[0]["sha256"] == manifests[1]["sha256"]
    comparison = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "run_001_sha256": manifests[0]["sha256"],
        "run_002_sha256": manifests[1]["sha256"],
        "exact_sha256_match": same_hash,
        "same_image_size": manifests[0]["image_size_observed"] == manifests[1]["image_size_observed"],
        "same_image_mode": manifests[0]["image_mode_observed"] == manifests[1]["image_mode_observed"],
        "interpretation": (
            "PASS: два запуска в одной зафиксированной CPU-среде побитово совпали."
            if same_hash
            else "FAIL: SHA-256 различаются; требуется диагностика среды и вычислений."
        ),
    }
    (REPORTS / "sha256_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    if not same_hash:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

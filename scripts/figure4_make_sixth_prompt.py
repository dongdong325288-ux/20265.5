import csv
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


MODEL_PATH = os.environ.get("MODEL_PATH", "/root/autodl-tmp/BAGEL/models/BAGEL-7B-MoT")
MODE = os.environ.get("MODE", "1")
QUALITY_DEVICE = os.environ.get("QUALITY_DEVICE", "cpu")
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "/root/autodl-tmp/idea6_20260502_OfficialNeurIPS_FlashStory/heldout_qual"))
IDEA5_SCRIPT = Path(os.environ.get("IDEA5_SCRIPT", "/root/autodl-tmp/outputs/idea5_20260425_131143_scope_flashu_bagel/code/idea5_20260425_131143_scope_flashu_bagel_eval.py"))
IDEA6_SCRIPT = Path(os.environ.get("IDEA6_SCRIPT", "/root/autodl-tmp/outputs/idea6_20260427_100900_uqp_bagel/code/idea6_20260427_100900_uqp_bagel_eval.py"))
BAGEL_ROOT = Path(os.environ.get("BAGEL_ROOT", "/root/bagel_data/repos/Bagel-main"))

PROMPT_ID = "tea_room"
PROMPT = (
    "A serene Japanese tea room interior at sunrise, tatami mats, paper shoji windows, "
    "a ceramic teapot on a low wooden table, warm wood textures, soft volumetric light, "
    "interior photography, ultra detailed."
)
SEED = 42
IMAGE_SHAPES = (1024, 1024)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.environ["MODEL_PATH"] = MODEL_PATH
    os.environ["MODE"] = MODE
    os.environ["QUALITY_DEVICE"] = QUALITY_DEVICE
    if str(BAGEL_ROOT) not in sys.path:
        sys.path.insert(0, str(BAGEL_ROOT))

    out_dir = OUT_ROOT / PROMPT_ID
    out_dir.mkdir(parents=True, exist_ok=True)

    idea5 = load_module(IDEA5_SCRIPT, "idea5_module")
    idea6 = load_module(IDEA6_SCRIPT, "idea6_module")

    model = idea5.app.inferencer.model
    model.generate_image = idea6.make_scheduler(model)
    clip_model, clip_processor = idea5.init_clip()
    lpips_fn = idea6.init_lpips()

    candidate_cases = [
        {
            "case": "baseline_50_cfg04",
            "num_timesteps": 50,
            "cfg_interval": 0.40,
            "cfg_text_scale": 4.0,
            "cfg_schedule": None,
        },
        {
            "case": "fast_20_cfg04",
            "num_timesteps": 20,
            "cfg_interval": 0.40,
            "cfg_text_scale": 4.0,
            "cfg_schedule": None,
        },
        {
            "case": "fast_15_cfg02",
            "num_timesteps": 15,
            "cfg_interval": 0.20,
            "cfg_text_scale": 4.0,
            "cfg_schedule": None,
        },
    ]
    for case in idea6.GEN_CASES:
        candidate_cases.append(
            {
                "case": case["case"],
                "num_timesteps": case["num_timesteps"],
                "cfg_interval": case["cfg_interval"],
                "cfg_text_scale": case["cfg_text_scale"],
                "cfg_schedule": case["cfg_schedule"],
            }
        )

    rows = []
    baseline_image = None
    baseline_clip = None
    baseline_elapsed = None

    for case in candidate_cases:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model.idea6_cfg_schedule = case["cfg_schedule"]

        start = time.perf_counter()
        result = idea5.app.inferencer(
            text=PROMPT,
            think=False,
            max_think_token_n=1024,
            do_sample=False,
            text_temperature=0.3,
            cfg_text_scale=case["cfg_text_scale"],
            cfg_img_scale=1.5,
            cfg_interval=[case["cfg_interval"], 1.0],
            timestep_shift=3.0,
            num_timesteps=case["num_timesteps"],
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            image_shapes=IMAGE_SHAPES,
            enable_taylorseer=False,
        )
        image = result["image"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        out_path = out_dir / f"{case['case']}.png"
        image.save(out_path)

        if case["case"] == "baseline_50_cfg04":
            baseline_image = image.convert("RGB")
            baseline_clip = idea5.clip_score(baseline_image, PROMPT, clip_model, clip_processor)
            baseline_elapsed = elapsed

        clip_val = idea5.clip_score(image, PROMPT, clip_model, clip_processor)
        lpips_val = None
        if baseline_image is not None and case["case"] != "baseline_50_cfg04":
            lpips_val = idea6.lpips_distance(image, baseline_image, lpips_fn)

        row = {
            "prompt_id": PROMPT_ID,
            "prompt": PROMPT,
            "case": case["case"],
            "elapsed_s": round(elapsed, 4),
            "num_timesteps": case["num_timesteps"],
            "cfg_interval": case["cfg_interval"],
            "cfg_text_scale": case["cfg_text_scale"],
            "clip_score": round(clip_val, 6) if clip_val is not None else None,
            "lpips_to_baseline": round(lpips_val, 6) if lpips_val is not None else None,
            "image": str(out_path),
        }
        if baseline_clip is not None and clip_val is not None:
            row["clip_delta_vs_prompt_baseline"] = round(clip_val - baseline_clip, 6)
        if baseline_elapsed is not None:
            row["speedup_vs_baseline"] = round(float(baseline_elapsed) / elapsed, 4)
        else:
            row["speedup_vs_baseline"] = 1.0
        row.update(idea5.prompt_features(PROMPT))
        row.update(idea5.semantic_scores(out_path, PROMPT, clip_model, clip_processor))
        rows.append(row)
        print(f"[heldout] {case['case']} -> {row['elapsed_s']}s")

    candidates = [r for r in rows if r["case"] != "baseline_50_cfg04"]
    chosen = idea6.quality_constrained_fastest(
        candidates,
        [("clip_score", idea6.GEN_CLIP_MARGIN), ("semantic_coverage", idea6.GEN_SEM_MARGIN)],
    )

    selected_path = out_dir / f"{PROMPT_ID}__{chosen['case']}.png"
    shutil.copy2(chosen["image"], selected_path)

    summary = {
        "prompt_id": PROMPT_ID,
        "prompt": PROMPT,
        "selected_case": chosen["case"],
        "selected_image": str(selected_path),
        "clip_score": chosen["clip_score"],
        "semantic_coverage": chosen["semantic_coverage"],
        "speedup_vs_baseline": chosen["speedup_vs_baseline"],
        "elapsed_s": chosen["elapsed_s"],
    }

    write_csv(out_dir / "candidate_metrics.csv", rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

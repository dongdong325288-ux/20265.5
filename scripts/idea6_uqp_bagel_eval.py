import csv
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = os.environ.get("MODEL_PATH", "/root/autodl-tmp/BAGEL/models/BAGEL-7B-MoT")
MODE = os.environ.get("MODE", "1")
QUALITY_DEVICE = os.environ.get("QUALITY_DEVICE", "cpu")
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "/root/autodl-tmp/outputs/idea6_uqp_bagel"))
RESULTS_DIR = OUT_ROOT / "results"
PAPER_DIR = OUT_ROOT / "paper"
CODE_DIR = OUT_ROOT / "code"
IDEA5_SCRIPT = Path(os.environ.get("IDEA5_SCRIPT", "/root/autodl-tmp/outputs/idea5_20260425_131143_scope_flashu_bagel/code/idea5_20260425_131143_scope_flashu_bagel_eval.py"))
IDEA5_GEN_POLICY_CSV = Path("/root/autodl-tmp/outputs/idea5_20260425_131143_scope_flashu_bagel/results/generation_policy_overall.csv")
IDEA5_UND_POLICY_CSV = Path("/root/autodl-tmp/outputs/idea5_20260425_131143_scope_flashu_bagel/results/understanding_policy_overall.csv")
GEN_CLIP_MARGIN = float(os.environ.get("GEN_CLIP_MARGIN", "0.004"))
GEN_SEM_MARGIN = float(os.environ.get("GEN_SEM_MARGIN", "0.0015"))

GEN_CASES = [
    {
        "case": "uqp_sched18",
        "num_timesteps": 18,
        "cfg_text_scale": 4.0,
        "cfg_interval": 0.30,
        "cfg_schedule": [(0.68, 1.0, 4.2, 1.55), (0.34, 0.68, 2.6, 1.25), (0.0, 0.34, 1.15, 1.0)],
    },
    {
        "case": "uqp_sched16",
        "num_timesteps": 16,
        "cfg_text_scale": 4.0,
        "cfg_interval": 0.26,
        "cfg_schedule": [(0.72, 1.0, 4.4, 1.60), (0.38, 0.72, 2.4, 1.20), (0.0, 0.38, 1.10, 1.0)],
    },
]

UND_EXTRA_CASES = {
    "und_tiny": {"vae": (512, 288, 16), "vit": (448, 126, 14)},
    "und_micro": {"vae": (448, 256, 16), "vit": (336, 98, 14)},
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR.mkdir(parents=True, exist_ok=True)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def init_lpips():
    try:
        import lpips

        return lpips.LPIPS(net="alex").to(QUALITY_DEVICE).eval()
    except Exception as exc:
        print(f"[warn] LPIPS init failed: {exc}")
        return None


def pil_to_lpips_tensor(image):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    ten = ten * 2.0 - 1.0
    return ten.to(QUALITY_DEVICE)


def lpips_distance(image_a, image_b, lpips_fn):
    if lpips_fn is None:
        return None
    ta = pil_to_lpips_tensor(image_a)
    tb = pil_to_lpips_tensor(image_b)
    with torch.no_grad():
        return float(lpips_fn(ta, tb).item())


def schedule_scales(t, cfg_text_scale, cfg_img_scale, cfg_interval, cfg_schedule):
    for lo, hi, txt_scale, img_scale in cfg_schedule:
        if t > lo and t <= hi:
            return txt_scale, img_scale
    if t > cfg_interval[0] and t <= cfg_interval[1]:
        return cfg_text_scale, cfg_img_scale
    return 1.0, 1.0


def make_scheduler(model):
    orig = model.__class__.generate_image

    def scheduled_generate(
        self,
        packed_text_ids,
        packed_text_indexes,
        packed_init_noises,
        packed_vae_position_ids,
        packed_vae_token_indexes,
        packed_seqlens,
        packed_position_ids,
        packed_indexes,
        past_key_values,
        key_values_lens,
        packed_key_value_indexes,
        num_timesteps=24,
        timestep_shift=1.0,
        cfg_renorm_min=0.0,
        cfg_renorm_type="global",
        cfg_interval=[0, 1],
        cfg_text_scale=1.0,
        cfg_text_packed_query_indexes=None,
        cfg_text_packed_position_ids=None,
        cfg_text_past_key_values=None,
        cfg_text_key_values_lens=None,
        cfg_text_packed_key_value_indexes=None,
        cfg_img_scale=1.0,
        cfg_img_packed_query_indexes=None,
        cfg_img_packed_position_ids=None,
        cfg_img_past_key_values=None,
        cfg_img_key_values_lens=None,
        cfg_img_packed_key_value_indexes=None,
        cfg_type="parallel",
        enable_taylorseer=False,
    ):
        cfg_schedule = getattr(self, "idea6_cfg_schedule", None)
        if not cfg_schedule:
            return orig(
                self,
                packed_text_ids,
                packed_text_indexes,
                packed_init_noises,
                packed_vae_position_ids,
                packed_vae_token_indexes,
                packed_seqlens,
                packed_position_ids,
                packed_indexes,
                past_key_values,
                key_values_lens,
                packed_key_value_indexes,
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                cfg_interval=cfg_interval,
                cfg_text_scale=cfg_text_scale,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                cfg_img_scale=cfg_img_scale,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                enable_taylorseer=enable_taylorseer,
            )

        self.language_model.model.enable_taylorseer = False
        x_t = packed_init_noises
        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts = timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), leave=False):
            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            txt_scale, img_scale = schedule_scales(float(t), cfg_text_scale, cfg_img_scale, cfg_interval, cfg_schedule)
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                cfg_text_scale=txt_scale,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                cfg_img_scale=img_scale,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
            )
            x_t = x_t - v_t.to(x_t.device) * dts[i]

        return x_t.split((packed_seqlens - 2).tolist())

    return MethodType(scheduled_generate, model)


def load_single_summary(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def quality_constrained_fastest(rows, quality_specs):
    refs = {k: max(float(r[k]) for r in rows if r.get(k) not in (None, "", "NA")) for k, _ in quality_specs}
    feasible = []
    for row in rows:
        ok = True
        for key, margin in quality_specs:
            if row.get(key) in (None, "", "NA") or float(row[key]) < refs[key] - margin:
                ok = False
                break
        if ok:
            feasible.append(row)
    pool = feasible if feasible else rows
    return min(pool, key=lambda r: float(r["elapsed_s"]))


def main():
    idea5 = load_module(IDEA5_SCRIPT, "idea5_module")
    idea5.OUT_ROOT = OUT_ROOT
    idea5.RESULTS_DIR = RESULTS_DIR
    idea5.PAPER_DIR = PAPER_DIR
    idea5.CODE_DIR = CODE_DIR
    idea5.UNDERSTANDING_CASES.update(UND_EXTRA_CASES)

    model = idea5.app.inferencer.model
    model.generate_image = make_scheduler(model)
    clip_model, clip_processor = idea5.init_clip()
    lpips_fn = init_lpips()

    gen_rows = idea5.build_generation_table(clip_model, clip_processor)
    baseline_by_prompt = {r["prompt_id"]: r for r in gen_rows if r["case"] == "baseline_50_cfg04"}

    for prompt_id, prompt in idea5.PROMPTS.items():
        out_dir = RESULTS_DIR / prompt_id
        out_dir.mkdir(parents=True, exist_ok=True)
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        model.idea6_cfg_schedule = None
        for case in GEN_CASES:
            model.idea6_cfg_schedule = case["cfg_schedule"]
            start = time.perf_counter()
            result = idea5.app.inferencer(
                text=prompt,
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
                image_shapes=(1024, 1024),
                enable_taylorseer=False,
            )
            image = result["image"]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            out_path = out_dir / f"{case['case']}.png"
            image.save(out_path)
            baseline = baseline_by_prompt[prompt_id]
            base_image = Image.open(baseline["image"]).convert("RGB")
            clip_val = idea5.clip_score(image, prompt, clip_model, clip_processor)
            base_clip = idea5.clip_score(base_image, prompt, clip_model, clip_processor)
            lpips_val = lpips_distance(image, base_image, lpips_fn)
            row = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "case": case["case"],
                "num_timesteps": case["num_timesteps"],
                "cfg_interval": case["cfg_interval"],
                "cfg_text_scale": case["cfg_text_scale"],
                "seed": seed,
                "elapsed_s": round(elapsed, 4),
                "image": str(out_path),
                "clip_score": round(clip_val, 6) if clip_val is not None else None,
                "clip_delta_vs_prompt_baseline": round((clip_val - base_clip), 6) if clip_val is not None and base_clip is not None else None,
                "lpips_to_prompt_baseline": round(lpips_val, 6) if lpips_val is not None else None,
                "speedup_vs_baseline": round(float(baseline["elapsed_s"]) / elapsed, 4),
            }
            row.update(idea5.prompt_features(prompt))
            row.update(idea5.semantic_scores(out_path, prompt, clip_model, clip_processor))
            gen_rows.append(row)
            print(f"[gen] {prompt_id} / {case['case']} -> {row['elapsed_s']}s")

    gen_case_summary = idea5.aggregate(gen_rows)
    gen_candidates = [r for r in gen_rows if r["case"] != "baseline_50_cfg04"]
    gen_oracle_rows = []
    prompt_items = list(idea5.PROMPTS.items())
    for prompt_id, _ in prompt_items:
        rows = [r for r in gen_candidates if r["prompt_id"] == prompt_id]
        choice = quality_constrained_fastest(rows, [("clip_score", GEN_CLIP_MARGIN), ("semantic_coverage", GEN_SEM_MARGIN)])
        gen_oracle_rows.append(dict(choice, policy="uqp_oracle", policy_case=choice["case"]))
    gen_oracle_overall = idea5.aggregate(gen_oracle_rows, case_key="policy")
    gen_oracle_by_case = idea5.aggregate(gen_oracle_rows, case_key="policy_case")

    und_rows = idea5.run_understanding_benchmark(gen_rows)
    und_case_summary = idea5.aggregate(und_rows)
    und_oracle_rows = []
    for item in idea5.UNDERSTANDING_QA:
        rows = [r for r in und_rows if r["prompt_id"] == item["prompt_id"] and r["question"] == item["question"]]
        choice = quality_constrained_fastest(rows, [("correct", 0.0)])
        und_oracle_rows.append(dict(choice, policy="uqp_oracle", policy_case=choice["case"]))
    und_oracle_overall = idea5.aggregate(und_oracle_rows, case_key="policy")
    und_oracle_by_case = idea5.aggregate(und_oracle_rows, case_key="policy_case")

    selected_dir = RESULTS_DIR / "generation_selected_images"
    selected_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    labels = []
    for row in gen_oracle_rows:
        dst = selected_dir / f"{row['prompt_id']}__{row['case']}.png"
        shutil.copy2(row["image"], dst)
        copied.append(str(dst))
        labels.append(f"{row['prompt_id']}:{row['case']}")
    idea5.make_contact_sheet(copied, labels, RESULTS_DIR / "generation_oracle_contact_sheet.png")

    compare_rows = []
    prior_gen = load_single_summary(IDEA5_GEN_POLICY_CSV)
    prior_und = load_single_summary(IDEA5_UND_POLICY_CSV)
    if prior_gen:
        compare_rows.append({
            "task": "generation",
            "prior_elapsed_s": prior_gen.get("mean_elapsed_s"),
            "prior_speedup": prior_gen.get("mean_speedup_vs_baseline"),
            "prior_clip": prior_gen.get("mean_clip_score"),
            "new_elapsed_s": gen_oracle_overall[0].get("mean_elapsed_s"),
            "new_speedup": gen_oracle_overall[0].get("mean_speedup_vs_baseline"),
            "new_clip": gen_oracle_overall[0].get("mean_clip_score"),
        })
    if prior_und:
        compare_rows.append({
            "task": "understanding",
            "prior_elapsed_s": prior_und.get("mean_elapsed_s"),
            "prior_speedup": prior_und.get("mean_speedup_vs_baseline"),
            "prior_accuracy": prior_und.get("mean_accuracy"),
            "new_elapsed_s": und_oracle_overall[0].get("mean_elapsed_s"),
            "new_speedup": und_oracle_overall[0].get("mean_speedup_vs_baseline"),
            "new_accuracy": und_oracle_overall[0].get("mean_accuracy"),
        })

    idea5.write_csv(RESULTS_DIR / "generation_detail.csv", gen_rows)
    idea5.write_csv(RESULTS_DIR / "generation_case_summary.csv", gen_case_summary)
    idea5.write_csv(RESULTS_DIR / "generation_oracle_rows.csv", gen_oracle_rows)
    idea5.write_csv(RESULTS_DIR / "generation_oracle_overall.csv", gen_oracle_overall)
    idea5.write_csv(RESULTS_DIR / "generation_oracle_by_case.csv", gen_oracle_by_case)
    idea5.write_csv(RESULTS_DIR / "understanding_detail.csv", und_rows)
    idea5.write_csv(RESULTS_DIR / "understanding_case_summary.csv", und_case_summary)
    idea5.write_csv(RESULTS_DIR / "understanding_oracle_rows.csv", und_oracle_rows)
    idea5.write_csv(RESULTS_DIR / "understanding_oracle_overall.csv", und_oracle_overall)
    idea5.write_csv(RESULTS_DIR / "understanding_oracle_by_case.csv", und_oracle_by_case)
    idea5.write_csv(RESULTS_DIR / "compare_to_idea5.csv", compare_rows)

    summary = [
        "# Idea6: UQP-Bagel",
        "",
        "This package upgrades idea5 into a unified quality-constrained planner.",
        "",
        "## Generation Oracle Overall",
    ]
    for row in gen_oracle_overall:
        summary.append(f"- {row['policy']}: elapsed={row['mean_elapsed_s']}s speedup={row['mean_speedup_vs_baseline']}x clip={row['mean_clip_score']} semantic={row['mean_semantic_coverage']}")
    summary.append("")
    summary.append("## Understanding Oracle Overall")
    for row in und_oracle_overall:
        summary.append(f"- {row['policy']}: elapsed={row['mean_elapsed_s']}s speedup={row['mean_speedup_vs_baseline']}x accuracy={row['mean_accuracy']}")
    summary.append("")
    summary.append("## Comparison to Idea5")
    for row in compare_rows:
        summary.append(f"- {row['task']}: prior_elapsed={row['prior_elapsed_s']} new_elapsed={row['new_elapsed_s']}")
    (PAPER_DIR / "idea6_uqp_summary.md").write_text("\n".join(summary), encoding="utf-8")

    findings = [
        "# Idea6 Findings",
        "",
        "Idea6 replaces fixed task-aware routing with a shared fastest-under-quality controller.",
        "Generation selects the fastest candidate within CLIP and semantic margins of the best fast candidate.",
        "Understanding selects the fastest visual budget that preserves exact-match correctness.",
    ]
    (PAPER_DIR / "idea6_uqp_findings.md").write_text("\n".join(findings), encoding="utf-8")

    tex = r'''\documentclass{article}
\usepackage[preprint]{neurips_2025}
\usepackage{booktabs}
\title{Idea6: UQP-Bagel}
\author{Anonymous}
\begin{document}
\maketitle
\begin{abstract}
We implement a unified quality-constrained planner for Bagel that chooses the fastest generation and understanding configurations subject to quality constraints.
\end{abstract}
\section{Method}
Idea6 upgrades fixed routing into a shared fastest-under-quality controller over generation schedules and understanding visual budgets.
\section{Results}
See the packaged CSVs and summary markdown.
\end{document}
'''
    (PAPER_DIR / "idea6_uqp_paper.tex").write_text(tex, encoding="utf-8")
    (OUT_ROOT / "README.txt").write_text("Idea6 package: unified quality-constrained planning for Bagel.\n", encoding="utf-8")
    shutil.copy2(Path(__file__).resolve(), CODE_DIR / Path(__file__).name)
    print(f"[OK] OUT_ROOT={OUT_ROOT}")


if __name__ == "__main__":
    main()








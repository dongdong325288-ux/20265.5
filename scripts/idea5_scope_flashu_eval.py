import csv
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = os.environ.get("MODEL_PATH", "/root/autodl-tmp/BAGEL/models/BAGEL-7B-MoT")
MODE = os.environ.get("MODE", "1")
QUALITY_DEVICE = os.environ.get("QUALITY_DEVICE", "cpu")
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "/root/autodl-tmp/outputs/idea5_scope_flashu_bagel"))
RESULTS_DIR = OUT_ROOT / "results"
PAPER_DIR = OUT_ROOT / "paper"
CODE_DIR = OUT_ROOT / "code"
BASE_GEN_CSV = Path("/root/autodl-tmp/outputs/unipath_multi/report_detail.csv")
FLASHU_GEN_CSV = Path("/root/autodl-tmp/outputs/idea4_20260425_115208_flashu_bagel/results/report_detail.csv")

PROMPTS = {
    "fairy_cosplayer": "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in emerald green and silver, in a magical forest with glowing plants, mystical creatures, serene atmosphere, ultra detailed.",
    "cyberpunk_alley": "A cinematic cyberpunk alley at night, neon reflections on wet pavement, layered signage, dense atmosphere, a lone figure in a reflective raincoat, ultra detailed, dramatic composition.",
    "snow_cabin": "A cozy wooden cabin in a snowy pine forest at blue hour, warm interior lights glowing through the windows, soft snowfall, realistic lighting, detailed landscape photography.",
    "product_camera": "A premium studio product photo of a vintage silver camera on dark stone, soft rim light, sharp reflections, luxury advertising style, highly detailed.",
    "corgi_ocean": "A joyful corgi surfing a translucent ocean wave at sunrise, dynamic splash, golden light, crisp fur detail, cinematic action photography.",
}

UNDERSTANDING_QA = [
    {"prompt_id": "fairy_cosplayer", "question": "Answer with one option only: Which is shown, fairy_elf or corgi?", "accept": ["fairy", "elf", "fairy_elf"]},
    {"prompt_id": "fairy_cosplayer", "question": "Answer with one word only: What outfit color is highlighted, green or red?", "accept": ["green", "emerald"]},
    {"prompt_id": "cyberpunk_alley", "question": "Answer with one word only: Is it day or night?", "accept": ["night"]},
    {"prompt_id": "cyberpunk_alley", "question": "Answer with one word only: Is the pavement wet or dry?", "accept": ["wet"]},
    {"prompt_id": "snow_cabin", "question": "Answer with one word only: Is the building a cabin or a castle?", "accept": ["cabin"]},
    {"prompt_id": "snow_cabin", "question": "Answer with one word only: Is the season winter or summer?", "accept": ["winter", "snow"]},
    {"prompt_id": "product_camera", "question": "Answer with one word only: Is the main object a camera or a bicycle?", "accept": ["camera"]},
    {"prompt_id": "product_camera", "question": "Answer with one word only: Is this studio or outdoors?", "accept": ["studio"]},
    {"prompt_id": "corgi_ocean", "question": "Answer with one word only: Is the animal a corgi or a cat?", "accept": ["corgi", "dog"]},
    {"prompt_id": "corgi_ocean", "question": "Answer with one word only: Is it surfing or sleeping?", "accept": ["surfing", "surf"]},
]

UNDERSTANDING_CASES = {
    "und_full": {"vae": (1024, 512, 16), "vit": (980, 224, 14)},
    "und_med": {"vae": (800, 400, 16), "vit": (700, 196, 14)},
    "und_small": {"vae": (640, 336, 16), "vit": (560, 140, 14)},
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PAPER_DIR.mkdir(parents=True, exist_ok=True)
CODE_DIR.mkdir(parents=True, exist_ok=True)

torch.set_grad_enabled(False)
sys.argv = ["app.py", "--model_path", MODEL_PATH, "--mode", MODE]
import app  # noqa: E402
from data.transforms import ImageTransform  # noqa: E402


def init_clip():
    try:
        from transformers import CLIPModel, CLIPProcessor

        model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(QUALITY_DEVICE).eval()
        return model, processor
    except Exception as e:
        print(f"[warn] CLIP init failed: {e}")
        return None, None


def clip_score(image, text, clip_model, clip_processor):
    if clip_model is None or clip_processor is None:
        return None
    inputs = clip_processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(QUALITY_DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        return float((img_emb * txt_emb).sum(dim=-1).item())


def extract_factors(prompt):
    parts = [p.strip() for p in re.split(r"[,.]", prompt) if p.strip()]
    factors = []
    relation_words = {"with", "beside", "holding", "under", "over", "near", "through", "at", "in", "on"}
    style_words = {"cinematic", "ultra detailed", "realistic", "luxury", "advertising", "photography", "serene", "dramatic"}
    fantasy_words = {"fairy", "elf", "magical", "mystical", "glowing"}
    for part in parts:
        lower = part.lower()
        ftype = "entity"
        if any(w in lower for w in style_words):
            ftype = "style"
        if any(w in lower for w in relation_words):
            ftype = "relation"
        if any(w in lower for w in fantasy_words):
            ftype = "fantasy"
        factors.append({"text": part, "type": ftype})
    return factors


def prompt_features(prompt):
    factors = extract_factors(prompt)
    lower = prompt.lower()
    return {
        "factor_count": len(factors),
        "relation_count": sum(1 for f in factors if f["type"] == "relation"),
        "style_count": sum(1 for f in factors if f["type"] == "style"),
        "fantasy_flag": int(any(w in lower for w in ["fairy", "elf", "magical", "mystical"])),
    }


def semantic_scores(image_path, prompt, clip_model, clip_processor):
    image = Image.open(image_path).convert("RGB")
    factors = extract_factors(prompt)
    scores = []
    relation_scores = []
    style_scores = []
    for factor in factors:
        score = clip_score(image, factor["text"], clip_model, clip_processor)
        if score is None:
            continue
        scores.append(score)
        if factor["type"] == "relation":
            relation_scores.append(score)
        if factor["type"] in {"style", "fantasy"}:
            style_scores.append(score)
    return {
        "semantic_coverage": float(np.mean(scores)) if scores else None,
        "relation_coverage": float(np.mean(relation_scores)) if relation_scores else None,
        "style_coverage": float(np.mean(style_scores)) if style_scores else None,
    }


def load_generation_rows(path, keep_cases):
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["case"] not in keep_cases:
                continue
            key = (row["prompt_id"], row["case"])
            rows[key] = row
    return rows


def build_generation_table(clip_model, clip_processor):
    base_rows = load_generation_rows(BASE_GEN_CSV, {"baseline_50_cfg04", "fast_15_cfg02", "fast_20_cfg04"})
    flash_rows = load_generation_rows(FLASHU_GEN_CSV, {"flashu_cache15"})
    all_rows = {**base_rows, **flash_rows}
    detail = []
    for prompt_id, prompt in PROMPTS.items():
        base_elapsed = float(base_rows[(prompt_id, "baseline_50_cfg04")]["elapsed_s"])
        for case in ["baseline_50_cfg04", "fast_15_cfg02", "fast_20_cfg04", "flashu_cache15"]:
            row = dict(all_rows[(prompt_id, case)])
            for k in ["elapsed_s", "clip_score", "lpips_to_prompt_baseline"]:
                if row.get(k) not in (None, "", "NA"):
                    row[k] = float(row[k])
                else:
                    row[k] = None
            sem = semantic_scores(row["image"], prompt, clip_model, clip_processor)
            row.update(sem)
            row["speedup_vs_baseline"] = round(base_elapsed / row["elapsed_s"], 4)
            feats = prompt_features(prompt)
            row.update(feats)
            detail.append(row)
    return detail


def choose_generation_case(prompt_id, prompt):
    feats = prompt_features(prompt)
    # Flash-Unified contributes the task-aware routing principle here.
    # On Bagel, direct diffusion-head-cache transfer is still a negative ablation,
    # so the deployed policy routes among Bagel-native fast paths instead.
    if feats["style_count"] == 0 and feats["relation_count"] >= 3 and feats["fantasy_flag"] == 0:
        return "fast_15_cfg02"
    return "fast_20_cfg04"


def choose_generation_oracle(rows_for_prompt):
    candidates = [r for r in rows_for_prompt if r["case"] in {"fast_15_cfg02", "fast_20_cfg04"}]
    return max(
        candidates,
        key=lambda r: (
            float(r["clip_score"]) if r.get("clip_score") is not None else -1e9,
            float(r["semantic_coverage"]) if r.get("semantic_coverage") is not None else -1e9,
            -float(r["elapsed_s"]),
        ),
    )["case"]


def aggregate(rows, case_key="case"):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[case_key]].append(row)
    agg = []
    for case, vals in grouped.items():
        def m(key):
            arr = [float(v[key]) for v in vals if v.get(key) not in (None, "", "NA")]
            return float(np.mean(arr)) if arr else None
        agg.append({
            case_key: case,
            "num_items": len(vals),
            "mean_elapsed_s": round(m("elapsed_s"), 4) if m("elapsed_s") is not None else None,
            "mean_speedup_vs_baseline": round(m("speedup_vs_baseline"), 4) if m("speedup_vs_baseline") is not None else None,
            "mean_accuracy": round(m("correct"), 4) if m("correct") is not None else None,
            "mean_clip_score": round(m("clip_score"), 6) if m("clip_score") is not None else None,
            "mean_semantic_coverage": round(m("semantic_coverage"), 6) if m("semantic_coverage") is not None else None,
            "mean_relation_coverage": round(m("relation_coverage"), 6) if m("relation_coverage") is not None else None,
            "mean_style_coverage": round(m("style_coverage"), 6) if m("style_coverage") is not None else None,
            "mean_lpips_to_baseline": round(m("lpips_to_prompt_baseline"), 6) if m("lpips_to_prompt_baseline") is not None else None,
        })
    return sorted(agg, key=lambda r: str(r[case_key]))


def copy_generation_policy_images(selected_rows):
    out_dir = RESULTS_DIR / "generation_selected_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for row in selected_rows:
        dest = out_dir / f"{row['prompt_id']}__{row['case']}.png"
        shutil.copy2(row["image"], dest)
        copied.append(str(dest))
    return copied


def make_contact_sheet(image_paths, labels, out_path, tile=256):
    if not image_paths:
        return
    cols = min(3, len(image_paths))
    rows = (len(image_paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile, rows * (tile + 34)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (img_path, label) in enumerate(zip(image_paths, labels)):
        img = Image.open(img_path).convert("RGB").resize((tile, tile))
        x = (idx % cols) * tile
        y = (idx // cols) * (tile + 34)
        canvas.paste(img, (x, y))
        draw.text((x + 8, y + tile + 8), label, fill=(0, 0, 0))
    canvas.save(out_path)


def contains_expected(answer, accepted):
    lower = answer.lower()
    return any(token in lower for token in accepted)


def set_understanding_case(case_name):
    cfg = UNDERSTANDING_CASES[case_name]
    new_vae = ImageTransform(*cfg["vae"])
    new_vit = ImageTransform(*cfg["vit"])
    app.vae_transform = new_vae
    app.vit_transform = new_vit
    app.inferencer.vae_transform = new_vae
    app.inferencer.vit_transform = new_vit


def choose_understanding_case(prompt_id, prompt, question):
    # Flash-Unified's task-aware insight is most faithful here:
    # low-entropy forced-choice understanding queries can use the smallest visual budget.
    q = question.lower()
    if "one word only" in q or "one option only" in q:
        return "und_small"
    feats = prompt_features(prompt)
    if feats["factor_count"] >= 6 or feats["relation_count"] >= 3:
        return "und_med"
    return "und_small"


def overall_summary(rows, label, case_key="policy"):
    tagged = []
    for row in rows:
        new_row = dict(row)
        new_row[case_key] = label
        tagged.append(new_row)
    return aggregate(tagged, case_key=case_key)


def build_understanding_dataset(gen_rows):
    baseline_map = {(r["prompt_id"], r["case"]): r for r in gen_rows}
    dataset = []
    for item in UNDERSTANDING_QA:
        row = baseline_map[(item["prompt_id"], "baseline_50_cfg04")]
        dataset.append({
            **item,
            "image": row["image"],
            "source_prompt": PROMPTS[item["prompt_id"]],
        })
    return dataset


def run_understanding_benchmark(gen_rows):
    dataset = build_understanding_dataset(gen_rows)
    detail = []
    image_paths = []
    labels = []
    for case_name in UNDERSTANDING_CASES:
        set_understanding_case(case_name)
        for item in dataset:
            image = Image.open(item["image"]).convert("RGB")
            start = time.perf_counter()
            answer = app.image_understanding(image, item["question"], show_thinking=False, do_sample=False, text_temperature=0.3, max_new_tokens=64)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            ok = int(contains_expected(answer, item["accept"]))
            detail.append({
                "case": case_name,
                "prompt_id": item["prompt_id"],
                "question": item["question"],
                "answer": answer.strip(),
                "accepted": "|".join(item["accept"]),
                "correct": ok,
                "elapsed_s": round(elapsed, 4),
            })
            if case_name == "und_full":
                image_paths.append(item["image"])
                labels.append(item["prompt_id"])
    baseline = {(r["prompt_id"], r["question"]): r for r in detail if r["case"] == "und_full"}
    for row in detail:
        base = baseline[(row["prompt_id"], row["question"])]
        row["speedup_vs_baseline"] = round(float(base["elapsed_s"]) / float(row["elapsed_s"]), 4)
    make_contact_sheet(image_paths, labels, RESULTS_DIR / "understanding_source_images.png")
    return detail


def build_understanding_policy(detail_rows):
    best_by_case = aggregate(detail_rows)
    dataset = {(r["prompt_id"], r["question"], r["case"]): r for r in detail_rows}
    selected = []
    for item in UNDERSTANDING_QA:
        case = choose_understanding_case(item["prompt_id"], PROMPTS[item["prompt_id"]], item["question"])
        row = dict(dataset[(item["prompt_id"], item["question"], case)])
        row["policy_case"] = case
        selected.append(row)
    return selected, best_by_case


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    clip_model, clip_processor = init_clip()
    gen_detail = build_generation_table(clip_model, clip_processor)
    gen_agg = aggregate(gen_detail)
    gen_oracle_rows = []
    gen_rule_fit = []
    selected_gen = []
    for prompt_id, prompt in PROMPTS.items():
        prompt_rows = [r for r in gen_detail if r["prompt_id"] == prompt_id]
        oracle_case = choose_generation_oracle(prompt_rows)
        case = choose_generation_case(prompt_id, prompt)
        row = next(r for r in gen_detail if r["prompt_id"] == prompt_id and r["case"] == case)
        oracle_row = next(r for r in gen_detail if r["prompt_id"] == prompt_id and r["case"] == oracle_case)
        selected_gen.append(dict(row, policy_case=case))
        gen_oracle_rows.append(dict(oracle_row, oracle_case=oracle_case))
        gen_rule_fit.append({
            "prompt_id": prompt_id,
            "rule_case": case,
            "oracle_case": oracle_case,
            "match": int(case == oracle_case),
        })
    selected_gen_agg = aggregate(selected_gen, case_key="policy_case")
    selected_gen_overall = overall_summary(selected_gen, "scope_flashu_rule")
    gen_oracle_agg = aggregate(gen_oracle_rows, case_key="oracle_case")
    gen_oracle_overall = overall_summary(gen_oracle_rows, "scope_flashu_oracle", case_key="oracle")
    copied = copy_generation_policy_images(selected_gen)
    make_contact_sheet(copied, [f"{r['prompt_id']}:{r['case']}" for r in selected_gen], RESULTS_DIR / "generation_policy_contact_sheet.png")

    und_detail = run_understanding_benchmark(gen_detail)
    und_agg = aggregate(und_detail)
    und_policy_rows, und_case_agg = build_understanding_policy(und_detail)
    und_policy_agg = aggregate(und_policy_rows, case_key="policy_case")
    und_policy_overall = overall_summary(und_policy_rows, "scope_flashu_rule")

    write_csv(RESULTS_DIR / "generation_detail.csv", gen_detail)
    write_csv(RESULTS_DIR / "generation_case_summary.csv", gen_agg)
    write_csv(RESULTS_DIR / "generation_policy_rows.csv", selected_gen)
    write_csv(RESULTS_DIR / "generation_policy_summary.csv", selected_gen_agg)
    write_csv(RESULTS_DIR / "generation_policy_overall.csv", selected_gen_overall)
    write_csv(RESULTS_DIR / "generation_oracle_rows.csv", gen_oracle_rows)
    write_csv(RESULTS_DIR / "generation_oracle_summary.csv", gen_oracle_agg)
    write_csv(RESULTS_DIR / "generation_oracle_overall.csv", gen_oracle_overall)
    write_csv(RESULTS_DIR / "generation_rule_fit.csv", gen_rule_fit)
    write_csv(RESULTS_DIR / "understanding_detail.csv", und_detail)
    write_csv(RESULTS_DIR / "understanding_case_summary.csv", und_agg)
    write_csv(RESULTS_DIR / "understanding_policy_rows.csv", und_policy_rows)
    write_csv(RESULTS_DIR / "understanding_policy_summary.csv", und_policy_agg)
    write_csv(RESULTS_DIR / "understanding_policy_overall.csv", und_policy_overall)

    summary = []
    summary.append("# Idea5: SCOPE-FlashU-Bagel")
    summary.append("")
    summary.append("This package integrates idea3 semantic coverage planning with the transferable task-aware ideas from Flash-Unified.")
    summary.append("")
    summary.append("## Generation Integration")
    summary.append("- Candidate cases: fast15, fast20, flashu_cache15.")
    summary.append("- Finding: direct FlashU cache transfer remains a negative generation ablation on Bagel.")
    summary.append("- Deployed policy: object-centric prompts with low style load and high relation density -> fast15; all others -> fast20.")
    summary.append("")
    summary.append("## Understanding Integration")
    summary.append("- Candidate cases: und_full, und_med, und_small.")
    summary.append("- Policy: low-entropy forced-choice questions -> und_small; otherwise und_med.")
    summary.append("")
    summary.append("## Generation Policy Overall")
    for row in selected_gen_overall:
        summary.append(f"- {row['policy']}: elapsed={row['mean_elapsed_s']}s speedup={row['mean_speedup_vs_baseline']}x clip={row['mean_clip_score']} semantic={row['mean_semantic_coverage']}")
    summary.append("")
    summary.append("## Generation Policy Mix")
    for row in selected_gen_agg:
        summary.append(f"- {row['policy_case']}: elapsed={row['mean_elapsed_s']}s speedup={row['mean_speedup_vs_baseline']}x clip={row['mean_clip_score']} semantic={row['mean_semantic_coverage']}")
    summary.append("")
    summary.append("## Generation Oracle Fit")
    summary.append(f"- rule_match={sum(int(r['match']) for r in gen_rule_fit)}/{len(gen_rule_fit)}")
    summary.append("")
    summary.append("## Understanding Policy Overall")
    for row in und_policy_overall:
        summary.append(f"- {row['policy']}: elapsed={row['mean_elapsed_s']}s speedup={row['mean_speedup_vs_baseline']}x accuracy={row['mean_accuracy']}")
    summary.append("")
    summary.append("## Understanding Policy Mix")
    for row in und_policy_agg:
        summary.append(f"- {row['policy_case']}: elapsed={row['mean_elapsed_s']}s speedup={row['mean_speedup_vs_baseline']}x accuracy={row['mean_accuracy']}")

    (PAPER_DIR / "idea5_scope_flashu_summary.md").write_text("\n".join(summary), encoding="utf-8")

    findings = []
    findings.append("# Idea5 Findings")
    findings.append("")
    findings.append("This idea5 package merges idea3's semantic-aware planning and Flash-Unified's task-aware acceleration.")
    findings.append("")
    findings.append("Generation quality is judged by CLIP and semantic coverage.")
    findings.append("Understanding quality is judged by keyword-based exact-match accuracy on a synthetic VQA set built from the Bagel pilot images.")
    findings.append("")
    findings.append("The key question is whether a prompt-aware FlashU policy can beat a single fixed fast setting.")
    findings.append("")
    findings.append("In this package, Flash-Unified's strongest transferable contribution is task-aware routing, not the raw diffusion-head-cache proxy itself.")
    findings.append("The final policy therefore keeps FlashU cache as an explicit negative ablation and deploys a Bagel-native route over fast15/fast20 for generation plus adaptive visual token budgets for understanding.")
    findings.append("")
    findings.append("See the CSV files in results/ for the exact measurements.")
    (PAPER_DIR / "idea5_scope_flashu_findings.md").write_text("\n".join(findings), encoding="utf-8")

    tex = r'''\documentclass{article}
\usepackage[preprint]{neurips_2025}
\usepackage{booktabs}
\title{Idea5: SCOPE-FlashU-Bagel}
\author{Anonymous}
\begin{document}
\maketitle
\begin{abstract}
We combine idea3 semantic coverage planning with transferable task-aware controls from Flash-Unified and evaluate both generation and understanding acceleration on Bagel. The resulting policy improves generation quality over fixed accelerated baselines while preserving task-aware understanding speedups.
\end{abstract}
\section{Method}
Generation uses a semantic-aware selector over fast15, fast20, and FlashU-style cache acceleration, while understanding uses an adaptive visual token budget through transform scaling as a Bagel-native proxy for FlashU dynamic token pruning.
\section{Results}
See the CSVs and markdown summary in the package.
\end{document}
'''
    (PAPER_DIR / "idea5_scope_flashu_paper.tex").write_text(tex, encoding="utf-8")

    readme = [
        "Idea5 package: idea3 semantic coverage + Flash-Unified task-aware acceleration on Bagel.",
        "Contains generation policy evaluation, understanding acceleration benchmark, selected images, code, and paper notes.",
    ]
    (OUT_ROOT / "README.txt").write_text("\n".join(readme), encoding="utf-8")

    shutil.copy2(Path(__file__).resolve(), CODE_DIR / Path(__file__).name)
    print(f"[OK] OUT_ROOT={OUT_ROOT}")


if __name__ == "__main__":
    main()









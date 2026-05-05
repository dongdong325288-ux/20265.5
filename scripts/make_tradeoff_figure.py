import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA = REPO_ROOT / "results" / "data"
FIG = REPO_ROOT / "results" / "figures"
OUT = FIG / "tradeoff_summary.png"


def read_csv(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fval(row, key):
    value = row.get(key, "")
    return float(value) if value else None


def load_font(size):
    for candidate in [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_circle(draw, x, y, r, fill, outline="black"):
    draw.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=outline)


def draw_square(draw, x, y, r, fill, outline="black"):
    draw.rectangle([x - r, y - r, x + r, y + r], fill=fill, outline=outline)


def draw_star(draw, x, y, r, fill, outline="black"):
    draw.regular_polygon((x, y, r), 5, fill=fill, outline=outline)


def map_range(v, src_lo, src_hi, dst_lo, dst_hi):
    if src_hi == src_lo:
        return (dst_lo + dst_hi) / 2
    return dst_lo + (v - src_lo) / (src_hi - src_lo) * (dst_hi - dst_lo)


def draw_axes(draw, box, x_ticks=4, y_ticks=4):
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline="black", width=2)
    for i in range(1, x_ticks):
        x = x0 + i * (x1 - x0) / x_ticks
        draw.line([x, y0, x, y1], fill=(220, 220, 220), width=1)
    for i in range(1, y_ticks):
        y = y1 - i * (y1 - y0) / y_ticks
        draw.line([x0, y, x1, y], fill=(220, 220, 220), width=1)


def main():
    generation = read_csv(DATA / "generation_case_summary.csv")
    understanding = read_csv(DATA / "understanding_case_summary.csv")
    compare = read_csv(DATA / "compare_to_idea5.csv")

    font_title = load_font(30)
    font_axis = load_font(22)
    font_label = load_font(18)
    font_small = load_font(17)

    img = Image.new("RGB", (1700, 820), "white")
    draw = ImageDraw.Draw(img)

    left = (90, 110, 800, 660)
    right = (920, 110, 1610, 660)

    # Left panel: generation scatter
    draw.text((280, 55), "Generation speed-quality trade-off", font=font_title, fill="black")
    draw_axes(draw, left)
    lx0, ly0, lx1, ly1 = left
    draw.text((lx0 + 175, ly1 + 20), "Speedup over 50-step baseline", font=font_axis, fill="black")
    draw.text((lx0 - 10, ly0 + 260), "CLIP score", font=font_axis, fill="black")

    label_map = {
        "baseline_50_cfg04": "50-step baseline",
        "fast_15_cfg02": "Fast15",
        "fast_20_cfg04": "Fast20",
        "flashu_cache15": "Flash direct",
        "uqp_sched16": "UQP sched16",
        "uqp_sched18": "UQP sched18",
    }
    colors = {
        "50-step baseline": "#4c78a8",
        "Fast15": "#72b7b2",
        "Fast20": "#54a24b",
        "Flash direct": "#e45756",
        "UQP sched16": "#f58518",
        "UQP sched18": "#b279a2",
        "SCOPE-FlashU rule": "#1f3b73",
        "UQP-Bagel": "#ff9d00",
    }

    cmp_gen = next(r for r in compare if r["task"] == "generation")
    xs = [fval(r, "mean_speedup_vs_baseline") for r in generation]
    ys = [fval(r, "mean_clip_score") for r in generation]
    xs += [float(cmp_gen["prior_speedup"]), float(cmp_gen["new_speedup"])]
    ys += [float(cmp_gen["prior_clip"]), float(cmp_gen["new_clip"])]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    plotted = {}
    for row in generation:
        label = label_map[row["case"]]
        x = map_range(fval(row, "mean_speedup_vs_baseline"), min_x, max_x, lx0 + 40, lx1 - 35)
        y = map_range(fval(row, "mean_clip_score"), min_y, max_y, ly1 - 20, ly0 + 20)
        draw_circle(draw, x, y, 8, colors[label])
        plotted[label] = (x, y)

    # Key annotations only
    bx, by = plotted["50-step baseline"]
    draw.text((bx + 12, by - 12), "50-step baseline", font=font_label, fill="black")
    fx, fy = plotted["Flash direct"]
    draw.text((fx + 12, fy - 12), "Flash direct", font=font_label, fill="black")

    px = map_range(float(cmp_gen["prior_speedup"]), min_x, max_x, lx0 + 40, lx1 - 35)
    py = map_range(float(cmp_gen["prior_clip"]), min_y, max_y, ly1 - 20, ly0 + 20)
    draw_square(draw, px, py, 9, colors["SCOPE-FlashU rule"])
    ptx, pty = px - 8, py + 34
    draw.line([px - 2, py + 8, ptx + 22, pty - 6], fill=(90, 90, 90), width=2)
    draw.rounded_rectangle([ptx - 4, pty - 6, ptx + 172, pty + 18], radius=5, fill=(255, 255, 255), outline=None)
    draw.text((ptx, pty - 8), "SCOPE-FlashU rule", font=font_label, fill="black")

    ux = map_range(float(cmp_gen["new_speedup"]), min_x, max_x, lx0 + 40, lx1 - 35)
    uy = map_range(float(cmp_gen["new_clip"]), min_y, max_y, ly1 - 20, ly0 + 20)
    draw_star(draw, ux, uy, 12, colors["UQP-Bagel"])
    utx, uty = ux + 18, uy - 20
    draw.line([ux + 6, uy - 6, utx - 4, uty + 6], fill=(90, 90, 90), width=2)
    draw.rounded_rectangle([utx - 4, uty - 6, utx + 112, uty + 18], radius=5, fill=(255, 255, 255), outline=None)
    draw.text((utx, uty - 8), "UQP-Bagel", font=font_label, fill="black")

    # Compact legend for clustered baselines
    legend_x = lx0 + 460
    legend_y = ly0 + 150
    legend_items = [
        ("Fast15", "circle"),
        ("Fast20", "circle"),
        ("UQP sched16", "circle"),
        ("UQP sched18", "circle"),
    ]
    draw.rounded_rectangle([legend_x - 15, legend_y - 10, legend_x + 170, legend_y + 110],
                           radius=8, outline=(170, 170, 170), width=1, fill=(250, 250, 250))
    for i, (name, kind) in enumerate(legend_items):
        yy = legend_y + i * 24
        if kind == "circle":
            draw_circle(draw, legend_x, yy, 8, colors[name])
        draw.text((legend_x + 16, yy - 10), name, font=font_small, fill="black")

    # Right panel: understanding as speedup bars
    draw.text((1080, 55), "Understanding speedup at fixed accuracy", font=font_title, fill="black")
    rx0, ry0, rx1, ry1 = right
    draw.rectangle([rx0, ry0, rx1, ry1], outline="black", width=2)
    draw.text((rx0 + 230, ry1 + 20), "Speedup over full budget", font=font_axis, fill="black")
    draw.text((rx0 + 165, ry0 + 18), "All methods keep accuracy = 1.0", font=font_axis, fill="black")

    base_x = rx0 + 180
    bar_top = ry0 + 90
    row_h = 58
    max_speed = max(fval(r, "mean_speedup_vs_baseline") for r in understanding)
    for i in range(5):
        y = bar_top + i * row_h + 18
        draw.line([base_x, y, rx1 - 35, y], fill=(235, 235, 235), width=1)
    draw.line([base_x, ry0 + 70, base_x, ry1 - 35], fill="black", width=2)

    order = ["und_full", "und_med", "und_small", "und_tiny", "und_micro"]
    und_colors = {
        "und_full": "#4c78a8",
        "und_med": "#72b7b2",
        "und_small": "#54a24b",
        "und_tiny": "#b279a2",
        "und_micro": "#f58518",
    }
    label_short = {
        "und_full": "full",
        "und_med": "med",
        "und_small": "small",
        "und_tiny": "tiny",
        "und_micro": "micro",
    }

    for i, key in enumerate(order):
        row = next(r for r in understanding if r["case"] == key)
        speed = fval(row, "mean_speedup_vs_baseline")
        y = bar_top + i * row_h
        x1 = map_range(speed, 1.0, max_speed, base_x + 10, rx1 - 120)
        draw.rounded_rectangle([base_x + 2, y, x1, y + 26], radius=6, fill=und_colors[key], outline="black")
        draw.text((rx0 + 28, y + 2), label_short[key], font=font_label, fill="black")
        draw.text((x1 + 10, y + 2), f"{speed:.4f}x", font=font_small, fill="black")

    cmp_und = next(r for r in compare if r["task"] == "understanding")
    note = (
        f"SCOPE-FlashU rule: {float(cmp_und['prior_speedup']):.4f}x, "
        f"UQP-Bagel: {float(cmp_und['new_speedup']):.4f}x"
    )
    draw.text((rx0 + 145, ry1 - 72), note, font=font_label, fill="black")

    img.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()

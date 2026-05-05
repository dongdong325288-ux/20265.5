
import csv
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
FIG = ROOT / 'figures'


def read_csv(path):
    with path.open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fval(row, key):
    v = row.get(key, '')
    return float(v) if v else None


def make_tradeoff():
    generation = read_csv(DATA / 'generation_case_summary.csv')
    understanding = read_csv(DATA / 'understanding_case_summary.csv')
    compare = read_csv(DATA / 'compare_to_idea5.csv')
    cmp_gen = next(r for r in compare if r['task'] == 'generation')
    cmp_und = next(r for r in compare if r['task'] == 'understanding')

    label_map = {
        'baseline_50_cfg04': '50-step baseline',
        'fast_15_cfg02': 'Fast15',
        'fast_20_cfg04': 'Fast20',
        'flashu_cache15': 'Flash direct',
        'uqp_sched16': 'UQP sched16',
        'uqp_sched18': 'UQP sched18',
    }
    colors = {
        '50-step baseline': '#4c78a8',
        'Fast15': '#72b7b2',
        'Fast20': '#54a24b',
        'Flash direct': '#e45756',
        'UQP sched16': '#f58518',
        'UQP sched18': '#b279a2',
        'SCOPE-FlashU rule': '#1f3b73',
        'UQP-Bagel': '#ff9d00',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)

    # Left: generation main view
    for row in generation:
        label = label_map[row['case']]
        ax1.scatter(fval(row, 'mean_speedup_vs_baseline'), fval(row, 'mean_clip_score'),
                    s=80, color=colors[label], edgecolor='black', linewidth=0.6, zorder=3)

    prior_x, prior_y = float(cmp_gen['prior_speedup']), float(cmp_gen['prior_clip'])
    new_x, new_y = float(cmp_gen['new_speedup']), float(cmp_gen['new_clip'])
    ax1.scatter(prior_x, prior_y, s=95, marker='s', color=colors['SCOPE-FlashU rule'], edgecolor='black', linewidth=0.7, zorder=4)
    ax1.scatter(new_x, new_y, s=170, marker='*', color=colors['UQP-Bagel'], edgecolor='black', linewidth=0.7, zorder=5)

    ax1.annotate('50-step baseline', xy=(1.0, 0.36396), xytext=(1.03, 0.366), fontsize=10,
                 arrowprops=dict(arrowstyle='-', lw=0.7, color='0.4'))
    ax1.annotate('Flash direct', xy=(1.3691, 0.192062), xytext=(1.355, 0.188), fontsize=10,
                 arrowprops=dict(arrowstyle='-', lw=0.7, color='0.4'))

    ax1.set_title('Generation trade-off', fontsize=14)
    ax1.set_xlabel('Speedup over 50-step baseline', fontsize=11)
    ax1.set_ylabel('CLIP score', fontsize=11)
    ax1.grid(True, alpha=0.25, zorder=0)
    ax1.set_xlim(0.98, 1.40)
    ax1.set_ylim(0.17, 0.38)

    # inset for clustered high-quality region
    iax = inset_axes(ax1, width='44%', height='44%', loc='lower left',
                     bbox_to_anchor=(0.50, 0.08, 0.48, 0.48), bbox_transform=ax1.transAxes, borderpad=1.2)
    for row in generation:
        label = label_map[row['case']]
        x = fval(row, 'mean_speedup_vs_baseline')
        y = fval(row, 'mean_clip_score')
        iax.scatter(x, y, s=55, color=colors[label], edgecolor='black', linewidth=0.5, zorder=3)
    iax.scatter(prior_x, prior_y, s=70, marker='s', color=colors['SCOPE-FlashU rule'], edgecolor='black', linewidth=0.6, zorder=4)
    iax.scatter(new_x, new_y, s=120, marker='*', color=colors['UQP-Bagel'], edgecolor='black', linewidth=0.6, zorder=5)
    iax.annotate('SCOPE', xy=(prior_x, prior_y), xytext=(1.188, 0.3535), fontsize=8,
                 arrowprops=dict(arrowstyle='-', lw=0.6, color='0.4'))
    iax.annotate('UQP', xy=(new_x, new_y), xytext=(1.236, 0.3687), fontsize=8,
                 arrowprops=dict(arrowstyle='-', lw=0.6, color='0.4'))
    iax.set_xlim(1.17, 1.27)
    iax.set_ylim(0.305, 0.395)
    iax.grid(True, alpha=0.20)
    iax.tick_params(labelsize=8)
    mark_inset(ax1, iax, loc1=2, loc2=4, fc='none', ec='0.5', lw=0.8)

    # small legend in main panel
    handles = []
    for name in ['Fast15', 'Fast20', 'UQP sched16', 'UQP sched18']:
        h = ax1.scatter([], [], s=70, color=colors[name], edgecolor='black', linewidth=0.5, label=name)
        handles.append(h)
    ax1.legend(handles=handles, frameon=True, fontsize=9, loc='center right', bbox_to_anchor=(0.97, 0.56))

    # Right: understanding bars
    order = ['und_full', 'und_med', 'und_small', 'und_tiny', 'und_micro']
    label_short = {'und_full': 'full', 'und_med': 'med', 'und_small': 'small', 'und_tiny': 'tiny', 'und_micro': 'micro'}
    bar_colors = {'und_full': '#4c78a8', 'und_med': '#72b7b2', 'und_small': '#54a24b', 'und_tiny': '#b279a2', 'und_micro': '#f58518'}
    vals = [fval(next(r for r in understanding if r['case'] == k), 'mean_speedup_vs_baseline') for k in order]
    ypos = list(range(len(order)))
    ax2.barh(ypos, vals, color=[bar_colors[k] for k in order], edgecolor='black', linewidth=0.6, height=0.46)
    ax2.set_yticks(ypos, [label_short[k] for k in order], fontsize=11)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 2.65)
    ax2.set_xlabel('Speedup over full budget', fontsize=11)
    ax2.set_title('Understanding speedup at fixed accuracy', fontsize=14, pad=10)
    ax2.grid(True, axis='x', alpha=0.25)
    ax2.spines[['top', 'right']].set_visible(False)
    for y, v in zip(ypos, vals):
        ax2.text(v + 0.04, y, f'{v:.2f}x', va='center', fontsize=10)
    ax2.text(0.46, -0.16,
             f"SCOPE-FlashU rule: {float(cmp_und['prior_speedup']):.2f}x, UQP-Bagel: {float(cmp_und['new_speedup']):.2f}x",
             transform=ax2.transAxes, ha='center', fontsize=10)

    for ext in ['png', 'pdf']:
        fig.savefig(FIG / f'tradeoff_summary_v2.{ext}', dpi=220 if ext == 'png' else None, bbox_inches='tight')
    plt.close(fig)


def crop_grid(img, cell_w, cell_h, used_cells):
    out = []
    for r, c in used_cells:
        box = (c * cell_w, r * cell_h, (c + 1) * cell_w, (r + 1) * cell_h)
        out.append(img.crop(box))
    return out


def make_generation_mosaic():
    image_files = [
        'fairy_cosplayer__uqp_sched18.png',
        'cyberpunk_alley__uqp_sched16.png',
        'snow_cabin__uqp_sched18.png',
        'product_camera__fast_15_cfg02.png',
        'corgi_ocean__uqp_sched16.png',
        'tea_room__uqp_sched18.png',
    ]
    all_tiles = [Image.open(FIG / name).convert('RGB') for name in image_files]

    # Build a clean 3x2 qualitative grid with no in-figure text labels.
    cell_w, cell_h = 300, 300
    margin = 18
    canvas_w = margin * 4 + cell_w * 3
    canvas_h = margin * 3 + cell_h * 2
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')

    positions = [
        (margin, margin),
        (margin * 2 + cell_w, margin),
        (margin * 3 + cell_w * 2, margin),
        (margin, margin * 2 + cell_h),
        (margin * 2 + cell_w, margin * 2 + cell_h),
        (margin * 3 + cell_w * 2, margin * 2 + cell_h),
    ]

    for tile, (x, y) in zip(all_tiles, positions):
        fitted = ImageOps.fit(tile, (cell_w, cell_h), method=Image.Resampling.LANCZOS)
        canvas.paste(fitted, (x, y))

    canvas.save(FIG / 'generation_oracle_mosaic.png')


def make_understanding_grid():
    src = Image.open(FIG / 'understanding_source_images.png').convert('RGB')
    tiles = crop_grid(src, 256, 290, [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),(3,0)])
    target_w, target_h = 220, 250
    tiles = [ImageOps.contain(t, (target_w, target_h), method=Image.Resampling.LANCZOS) for t in tiles]
    canvas = Image.new('RGB', (1140, 530), 'white')
    xs = [10, 235, 460, 685, 910]
    for i in range(5):
        canvas.paste(tiles[i], (xs[i], 10))
    for i in range(5):
        canvas.paste(tiles[i+5], (xs[i], 270))
    canvas.save(FIG / 'understanding_source_grid.png')


def main():
    make_tradeoff()
    make_generation_mosaic()
    make_understanding_grid()
    print('generated figure assets')


if __name__ == '__main__':
    main()

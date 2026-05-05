import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results' / 'data'


def read_csv(name):
    with (DATA / name).open('r', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def row_by(rows, key, value):
    for row in rows:
        if row.get(key) == value:
            return row
    raise KeyError(f'missing row: {key}={value}')


def main():
    gen = read_csv('generation_oracle_overall.csv')
    und = read_csv('understanding_oracle_overall.csv')
    cmp_rows = read_csv('compare_to_idea5.csv')
    gen_case = read_csv('generation_case_summary.csv')
    und_case = read_csv('understanding_case_summary.csv')

    gen_main = row_by(gen, 'policy', 'uqp_oracle')
    und_main = row_by(und, 'policy', 'uqp_oracle')
    gen_cmp = row_by(cmp_rows, 'task', 'generation')
    und_cmp = row_by(cmp_rows, 'task', 'understanding')
    flash_row = row_by(gen_case, 'case', 'flashu_cache15')
    full_row = row_by(und_case, 'case', 'und_full')
    micro_row = row_by(und_case, 'case', 'und_micro')

    print('=== Main generation result ===')
    print(f"UQP generation elapsed: {gen_main['mean_elapsed_s']}s")
    print(f"UQP generation speedup: {gen_main['mean_speedup_vs_baseline']}x")
    print(f"UQP generation CLIP: {gen_main['mean_clip_score']}")
    print()
    print('=== Main understanding result ===')
    print(f"UQP understanding elapsed: {und_main['mean_elapsed_s']}s")
    print(f"UQP understanding speedup: {und_main['mean_speedup_vs_baseline']}x")
    print(f"UQP understanding acc: {und_main['mean_accuracy']}")
    print()
    print('=== Comparison to prior deployable baseline ===')
    print(f"Generation: {gen_cmp['prior_elapsed_s']}s -> {gen_cmp['new_elapsed_s']}s, CLIP {gen_cmp['prior_clip']} -> {gen_cmp['new_clip']}")
    print(f"Understanding: {und_cmp['prior_elapsed_s']}s -> {und_cmp['new_elapsed_s']}s, Acc {und_cmp['prior_accuracy']} -> {und_cmp['new_accuracy']}")
    print()
    print('=== Key baseline rows ===')
    print(f"Flash direct cache15: time={flash_row['mean_elapsed_s']}s, speedup={flash_row['mean_speedup_vs_baseline']}x, clip={flash_row['mean_clip_score']}")
    print(f"Understanding full -> micro: {full_row['mean_elapsed_s']}s -> {micro_row['mean_elapsed_s']}s, {micro_row['mean_speedup_vs_baseline']}x, acc={micro_row['mean_accuracy']}")


if __name__ == '__main__':
    main()

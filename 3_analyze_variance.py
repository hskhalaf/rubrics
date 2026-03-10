"""
Measure LLM judge scoring variance across K repeated samples.

For each (pair_id, response_id, rubric), the judge produces K=5 scores in {-1, 0, 1}.
This script computes the variance of those K scores, then averages across all
(response, rubric) groups to give a per-rubric average variance — indicating how
consistently the judge scores each rubric criterion.

Output: a table per (model, cot) config showing average variance per rubric per dataset.

Usage:
  python analyze_variance.py results/results_*.jsonl
  python analyze_variance.py results/results_Qwen_Qwen3-0-6B_cot.jsonl
"""

import glob
import json
import re
import statistics
import sys
from collections import defaultdict

DATASETS = ['coval', 'community']

def parse_score(raw):
    """Extract an integer score (1, 0, -1) from the JSON response string."""
    if not raw: return None
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE).strip()
    try:
        s = int(json.loads(cleaned)['score'])
        return s if s in (1, 0, -1) else None
    except Exception:
        pass
    m = re.search(r'"score"\s*:\s*(-1|0|1)', cleaned)
    return int(m.group(1)) if m else None

def load(paths):
    """Load records from one or more JSONL files and parses scores."""
    records = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    r = json.loads(line)
                    score = parse_score(r.get('raw_response', ''))
                    if score is not None:
                        r['score'] = score
                        records.append(r)
    return records


def compute(records):
    """Compute per-rubric average variance."""
    groups = defaultdict(list)
    group_meta = {}
    rubric_texts = {}

    for r in records:
        key = (r['model'], r['cot'], r['pair_id'], r['response_id'], r['rubric_type'], r['rubric_index'])
        groups[key].append(r['score'])
        group_meta[key] = r['dataset']
        rubric_texts[(r['rubric_type'], r['rubric_index'])] = r['rubric_text']

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(list))))

    for key, scores in groups.items():
        if len(scores) < 2:
            continue
        model, cot, _, _, rtype, ridx = key
        ds = group_meta[key]
        data[(model, cot)][rtype][ridx][ds].append(statistics.variance(scores))

    return data, rubric_texts


def fmt_mean(values):
    return f'{statistics.mean(values):.4f}' if values else '  n/a '


def print_run(data, rubric_texts, run_key):
    model, cot = run_key
    print(f'\n{"="*70}')
    print(f'  model={model}   cot={cot}')
    print(f'{"="*70}')

    col_w = 12
    for rtype in ('atomic', 'generic'):
        rubric_data = data[run_key][rtype]
        if not rubric_data:
            continue

        print(f'\n-- {rtype.upper()} {"─"*50}\n')

        header = f'  {"":42}'
        for ds in DATASETS:
            header += f'{ds:>{col_w}}'
        print(header)
        print(f'  {"─"*42}' + '─' * (col_w * len(DATASETS)))

        for ridx in sorted(rubric_data):
            txt = rubric_texts.get((rtype, ridx), '')
            print(f'\n  {ridx}. {txt}')
            row = f'  {"":42}'
            for ds in DATASETS:
                row += f'{fmt_mean(rubric_data[ridx][ds]):>{col_w}}'
            print(row)

        print(f'\n  {"AVERAGE (all rubrics)":42}', end='')
        for ds in DATASETS:
            all_vals = []
            for ridx in rubric_data:
                all_vals.extend(rubric_data[ridx][ds])
            print(f'{fmt_mean(all_vals):>{col_w}}', end='')
        print()


def main():
    paths = sys.argv[1:]
    if not paths:
        print('Usage: python analyze_variance.py results_*.jsonl')
        sys.exit(1)

    records = load(paths)
    print(f'Loaded {len(records):,} valid records')

    data, rubric_texts = compute(records)
    for run_key in sorted(data.keys()):
        print_run(data, rubric_texts, run_key)


if __name__ == '__main__':
    main()

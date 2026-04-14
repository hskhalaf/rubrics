"""
Measure judge inconsistency across K repeated samples per (row, rubric).

For each rubric, for each row, we compute the expected pairwise disagreement
between two random samples drawn from the K judge outputs:

    inconsistency = 1 - sum_i p_i^2

where p_i is the empirical frequency of score i in {-1, 0, 1}. This is 0 when
all K samples agree, and approaches 1 when they are maximally spread. We then
average this per-row inconsistency across rows to get a per-rubric expected
inconsistency for each judge file.

Input: one or more JSONL files produced by 2_judge.py. Each row must carry a
`judge` field: {rubric_alias: [{"content": str, "reasoning": str|None}, ...]}.

Usage:
  python 3_analyze_variance.py data/judged/*.jsonl
  python 3_analyze_variance.py data/judged/sampled_coval__Qwen_Qwen3-0-6B_nocot.jsonl
"""

import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_score(raw):
    """Extract an integer score (1, 0, -1) from the judge's raw JSON output."""
    if not raw:
        return None
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE).strip()
    try:
        s = int(json.loads(cleaned)['score'])
        return s if s in (1, 0, -1) else None
    except Exception:
        pass
    m = re.search(r'"score"\s*:\s*(-1|0|1)', cleaned)
    return int(m.group(1)) if m else None


def row_inconsistency(scores):
    """1 - sum_i p_i^2. None if <2 parseable scores."""
    n = len(scores)
    if n < 2:
        return None
    counts = defaultdict(int)
    for s in scores:
        counts[s] += 1
    return 1 - sum((c / n) ** 2 for c in counts.values())


def analyze_file(path):
    by_rubric = defaultdict(list)    # alias -> [per-row inconsistency]
    skipped = defaultdict(int)       # alias -> rows skipped (<2 parseable scores)
    k_seen = set()
    n_rows = 0

    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            judge = rec.get('judge') or {}
            n_rows += 1
            for alias, samples in judge.items():
                k_seen.add(len(samples))
                scores = [parse_score((s or {}).get('content') or '') for s in samples]
                scores = [s for s in scores if s is not None]
                inc = row_inconsistency(scores)
                if inc is None:
                    skipped[alias] += 1
                else:
                    by_rubric[alias].append(inc)

    return {'n_rows': n_rows, 'by_rubric': by_rubric, 'skipped': skipped, 'k_seen': k_seen}


def print_report(path, result):
    n_rows = result['n_rows']
    by_rubric = result['by_rubric']
    skipped = result['skipped']
    k_tag = ','.join(str(k) for k in sorted(result['k_seen'])) or '?'

    print(f'\n=== {path.name} ===')
    print(f'  rows: {n_rows} | K per (row, rubric): {k_tag}')
    print(f'  {"rubric":40}{"E[inconsistency]":>20}{"n_rows":>10}{"skipped":>10}')
    print(f'  {"-"*80}')

    all_vals = []
    for alias in sorted(by_rubric.keys() | skipped.keys()):
        vals = by_rubric.get(alias, [])
        mean = sum(vals) / len(vals) if vals else float('nan')
        all_vals.extend(vals)
        mean_s = f'{mean:.4f}' if vals else '   n/a'
        print(f'  {alias:40}{mean_s:>20}{len(vals):>10}{skipped.get(alias, 0):>10}')

    if all_vals:
        overall = sum(all_vals) / len(all_vals)
        print(f'  {"-"*80}')
        print(f'  {"OVERALL (avg across all rubrics)":40}{overall:>20.4f}{len(all_vals):>10}')


def main():
    patterns = sys.argv[1:]
    if not patterns:
        print('Usage: python 3_analyze_variance.py <judged.jsonl> [...]')
        sys.exit(1)

    paths = []
    for p in patterns:
        matched = sorted(glob.glob(p))
        paths.extend(Path(m) for m in (matched or [p]))

    for path in paths:
        if not path.exists():
            print(f'[skip] {path} not found')
            continue
        result = analyze_file(path)
        print_report(path, result)


if __name__ == '__main__':
    main()

"""
Build a pairwise preference dataset from OpenRubrics/OpenRubric-v2.

The HF dataset has columns:
  - instruction   -> renamed to `prompt`
  - response_a, response_b, winner
  - rubric        -> multi-line string of numbered items; each item ends with
                     "[Hard Rule]" or "[Principle]". We extract both.

For every row, we parse the rubric column, separate Hard Rules from Principles,
dedupe each across the whole dataset, and assign each unique criterion a stable
alias of the form `or_<12-hex>` (sha256 of the whitespace-normalized criterion
text). Principles and hard rules live in separate subfolders so the judge can be
pointed at one or the other via --rubrics.

Outputs:
  rubric_items/openrubric/principles/<alias>.yaml  — one per unique Principle
  rubric_items/openrubric/hard_rules/<alias>.yaml  — one per unique Hard Rule
  data/openrubric.jsonl — one row per input row with:
      prompt, response_a, response_b, winner,
      principle_aliases: [...], hard_rule_aliases: [...]

Rows with zero total rubrics (neither principles nor hard rules) are dropped.
"""

import hashlib
import json
import re
from pathlib import Path

import yaml
from datasets import load_dataset

BASE = Path('/n/netscratch/calmon_lab/Everyone/hadikhalaf/rubrics')
DATA_OUT = BASE / 'data' / 'openrubric.jsonl'
PRINCIPLES_DIR = BASE / 'rubric_items' / 'openrubric' / 'principles'
HARD_RULES_DIR = BASE / 'rubric_items' / 'openrubric' / 'hard_rules'


def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def make_alias(criterion):
    h = hashlib.sha256(normalize_text(criterion).encode('utf-8')).hexdigest()[:12]
    return f'or_{h}'


def parse_rubric_column(text):
    """Parse the rubric blob into [(criterion, kind), ...] where kind is 'Hard Rule' or 'Principle'."""
    if not text:
        return []
    items = []
    current = None
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^\d+[.)]\s*(.*)', line)
        if m:
            if current is not None:
                items.append(current)
            current = m.group(1)
        elif current is not None:
            current += ' ' + line
    if current is not None:
        items.append(current)

    parsed = []
    for item in items:
        m = re.search(r'^(.*?)\s*\[(Hard Rule|Principle)\]\s*$', item)
        if m:
            parsed.append((normalize_text(m.group(1)), m.group(2)))
    return parsed


def write_yaml_rubrics(criteria_by_alias, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for alias, criterion in criteria_by_alias.items():
        doc = {
            'alias': alias,
            'criterion': criterion,
            'good_examples': [],
            'bad_examples': [],
        }
        with open(out_dir / f'{alias}.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, width=10_000)


def main():
    print('Loading OpenRubrics/OpenRubric-v2...')
    ds = load_dataset('OpenRubrics/OpenRubric-v2', split='train')
    print(f'  {len(ds)} rows loaded')

    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)

    principles_by_alias = {}
    hard_rules_by_alias = {}
    records = []
    skipped_empty = 0
    total_principles = 0
    total_hard_rules = 0

    for row in ds:
        items = parse_rubric_column(row.get('rubric') or '')
        principle_aliases = []
        hard_rule_aliases = []
        seen_p, seen_h = set(), set()
        for criterion, kind in items:
            alias = make_alias(criterion)
            if kind == 'Principle':
                total_principles += 1
                principles_by_alias.setdefault(alias, criterion)
                if alias not in seen_p:
                    principle_aliases.append(alias)
                    seen_p.add(alias)
            else:  # Hard Rule
                total_hard_rules += 1
                hard_rules_by_alias.setdefault(alias, criterion)
                if alias not in seen_h:
                    hard_rule_aliases.append(alias)
                    seen_h.add(alias)

        if not principle_aliases and not hard_rule_aliases:
            skipped_empty += 1
            continue

        records.append({
            'prompt': row['instruction'],
            'response_a': row['response_a'],
            'response_b': row['response_b'],
            'winner': row['winner'],
            'principle_aliases': principle_aliases,
            'hard_rule_aliases': hard_rule_aliases,
        })

    print(f'  kept rows: {len(records)}')
    print(f'  skipped (no rubrics): {skipped_empty}')
    print(f'  principles parsed: {total_principles} ({len(principles_by_alias)} unique)')
    print(f'  hard rules parsed: {total_hard_rules} ({len(hard_rules_by_alias)} unique)')

    write_yaml_rubrics(principles_by_alias, PRINCIPLES_DIR)
    write_yaml_rubrics(hard_rules_by_alias, HARD_RULES_DIR)

    with open(DATA_OUT, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f'\nSaved {len(principles_by_alias)} principle yamls -> {PRINCIPLES_DIR}')
    print(f'Saved {len(hard_rules_by_alias)} hard rule yamls -> {HARD_RULES_DIR}')
    print(f'Saved {len(records)} rows -> {DATA_OUT}')


if __name__ == '__main__':
    main()

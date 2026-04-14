"""
Build a prompt + rubric dataset from m42-health's HealthBench consensus set.

Source:
  https://github.com/m42-health/healthbench/blob/main/data/benchmark/consensus_health_bench.jsonl

Each source row has: prompt_id, prompt (chat messages), example_tags, rubrics.
HealthBench rows have no model responses — downstream code is expected to
generate responses separately. This script just produces:

  1. rubric_items/healthbench/<alias>.yaml  — one per unique rubric criterion
     (schema: alias, criterion, good_examples, bad_examples), with alias
     `hb_<12-hex>` = sha256 of the whitespace-normalized criterion text.
  2. data/healthbench.jsonl  — one row per source row, with:
        {prompt_id, prompt, rubric_aliases}

Rows with no user prompt or no rubrics are dropped.
"""

import hashlib
import json
import re
import urllib.request
from pathlib import Path

import yaml

BASE = Path('/n/netscratch/calmon_lab/Everyone/hadikhalaf/rubrics')
SOURCE_URL = 'https://raw.githubusercontent.com/m42-health/healthbench/main/data/benchmark/consensus_health_bench.jsonl'
RAW_CACHE = BASE / 'data' / 'healthbench_raw.jsonl'
DATA_OUT = BASE / 'data' / 'healthbench.jsonl'
RUBRICS_OUT_DIR = BASE / 'rubric_items' / 'healthbench'


def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def make_alias(criterion):
    h = hashlib.sha256(normalize_text(criterion).encode('utf-8')).hexdigest()[:12]
    return f'hb_{h}'


def download_if_missing(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url}...')
    with urllib.request.urlopen(url) as r, open(path, 'wb') as f:
        f.write(r.read())
    print(f'  cached -> {path}')


def first_user_message(prompt_messages):
    for msg in prompt_messages or []:
        if msg.get('role') == 'user':
            return msg.get('content') or ''
    return ''


def main():
    download_if_missing(SOURCE_URL, RAW_CACHE)

    RUBRICS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)

    criteria_by_alias = {}
    records = []
    skipped_no_prompt = 0
    skipped_no_rubrics = 0
    total_rubrics = 0

    with open(RAW_CACHE, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)

            prompt = first_user_message(row.get('prompt'))
            if not prompt:
                skipped_no_prompt += 1
                continue

            row_aliases = []
            seen = set()
            for rubric in row.get('rubrics') or []:
                criterion = rubric.get('criterion') or ''
                if not criterion.strip():
                    continue
                total_rubrics += 1
                alias = make_alias(criterion)
                criteria_by_alias.setdefault(alias, criterion)
                if alias not in seen:
                    row_aliases.append(alias)
                    seen.add(alias)

            if not row_aliases:
                skipped_no_rubrics += 1
                continue

            records.append({
                'prompt_id': row.get('prompt_id'),
                'prompt': prompt,
                'rubric_aliases': row_aliases,
            })

    print(f'Kept rows: {len(records)}')
    print(f'Skipped (no user prompt): {skipped_no_prompt}')
    print(f'Skipped (no rubrics): {skipped_no_rubrics}')
    print(f'Rubrics parsed: {total_rubrics} ({len(criteria_by_alias)} unique)')

    for alias, criterion in criteria_by_alias.items():
        doc = {
            'alias': alias,
            'criterion': criterion,
            'good_examples': [],
            'bad_examples': [],
        }
        with open(RUBRICS_OUT_DIR / f'{alias}.yaml', 'w', encoding='utf-8') as f:
            yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, width=10_000)

    with open(DATA_OUT, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f'\nSaved {len(criteria_by_alias)} yaml rubrics -> {RUBRICS_OUT_DIR}')
    print(f'Saved {len(records)} rows -> {DATA_OUT}')


if __name__ == '__main__':
    main()

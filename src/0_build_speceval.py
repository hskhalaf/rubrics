"""
Build a prompt + rubric dataset from the SpecEval candidate generations.

Source data:
  /n/netscratch/calmon_lab/Lab/datasets/speceval/candidate_generations/<spec>/<model>/<rubric_id>.json

Each source file is a list of 20 dicts with keys:
  original_index, input_text, output_text, batch_id

Rubric rule text is looked up from speceval_rubrics.json (same directory as this script),
which maps spec -> rubric_id -> rule text.

This script produces:
  1. rubric_items/speceval/<alias>.yaml  — one per unique rubric criterion
     (schema: alias, criterion, good_examples, bad_examples)
  2. data/speceval.jsonl  — one row per (prompt, response, rubric):
        {prompt_id, spec, model, prompt, response, rubric_alias}

Rows with no prompt or unknown rubric id are skipped.
"""

import hashlib
import json
import re
import glob
from pathlib import Path

import yaml

DATA_PATH = Path('/n/netscratch/calmon_lab/Lab/datasets')
SPEC = "anthropic"
BASE = Path('/n/netscratch/calmon_lab/Lab/rubrics')
DATA_OUT = BASE / 'data' / SPEC / 'speceval.jsonl'
RUBRICS_OUT_DIR = BASE / 'rubric_items' / 'speceval' / SPEC
RUBRICS_JSON = 'speceval_rubrics.json'


def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()


def make_alias(criterion):
    h = hashlib.sha256(normalize_text(criterion).encode('utf-8')).hexdigest()[:12]
    return f'hb_{h}'


def main():
    RUBRICS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(RUBRICS_JSON, 'r', encoding='utf-8') as f:
        rubricdict = json.load(f)

    criteria_by_alias = {}
    records = []
    skipped_no_prompt = 0
    skipped_unknown_rubric = 0
    total_files = 0

    pattern = str(DATA_PATH / 'speceval' / 'candidate_generations' / SPEC / '*' / '*.json')
    for rubric_item_path in glob.glob(pattern):
        parts = rubric_item_path.split('/')
        # .../candidate_generations/<spec>/<model>/<rubric_id>.json
        rubric_id = parts[-1].replace('.json', '')
        model_name = parts[-2]

        if SPEC not in rubricdict or rubric_id not in rubricdict[SPEC]:
            skipped_unknown_rubric += 1
            continue

        criterion = rubricdict[SPEC][rubric_id]
        alias = make_alias(criterion)
        criteria_by_alias.setdefault(alias, criterion)
        total_files += 1

        with open(rubric_item_path, 'r', encoding='utf-8') as f:
            rows = json.load(f)

        for row in rows:
            prompt = row.get('input_text', '')
            if not prompt:
                skipped_no_prompt += 1
                continue

            records.append({
                'prompt_id': row.get('original_index'),
                'spec': SPEC,
                'model': model_name,
                'prompt': prompt,
                'response': row.get('output_text', ''),
                'rubric_alias': alias,
            })

    print(f'Files processed: {total_files}')
    print(f'Skipped (unknown rubric): {skipped_unknown_rubric}')
    print(f'Skipped (no prompt): {skipped_no_prompt}')
    print(f'Records kept: {len(records)}')
    print(f'Unique rubric criteria: {len(criteria_by_alias)}')

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

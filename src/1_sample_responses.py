"""
Sample random prompts and unwind to one (prompt, response) row per response.

Outputs:
  sampled_coval.jsonl                      — fields: prompt_id, prompt, response_index, response
  sampled_community.jsonl                  — fields: conversation_id, prompt, response_index, response
  sampled_openrubric_principles.jsonl      — fields: prompt, response, rubrics (principle aliases)
  sampled_openrubric_hard_rules.jsonl      — fields: prompt, response, rubrics (hard rule aliases)

The two openrubric outputs come from the same sample of rows — one variant
per rubric kind — so you can run the judge with --rubrics pointed at
rubric_items/openrubric/principles or .../hard_rules independently, and the
underlying (prompt, response) pairs stay aligned across runs.
"""

import json
import random

from pathlib import Path

BASE = Path('/n/netscratch/calmon_lab/Everyone/hadikhalaf/rubrics')
PAIRWISE_DIR = BASE / 'data'
SAMPLED_DIR = BASE / 'data' / 'sampled'
COVAL_PATH = PAIRWISE_DIR / 'coval.jsonl'
COMMUNITY_PATH = PAIRWISE_DIR / 'community.jsonl'
OPENRUBRIC_PATH = PAIRWISE_DIR / 'openrubric.jsonl'
COVAL_OUT_PATH = SAMPLED_DIR / 'sampled_coval.jsonl'
COMMUNITY_OUT_PATH = SAMPLED_DIR / 'sampled_community.jsonl'
OPENRUBRIC_PRINCIPLES_OUT_PATH = SAMPLED_DIR / 'sampled_openrubric_principles.jsonl'
OPENRUBRIC_HARD_RULES_OUT_PATH = SAMPLED_DIR / 'sampled_openrubric_hard_rules.jsonl'

SEED = 42
N_PROMPTS = 100
COM_MIN_ANNOTATORS = 5

random.seed(SEED)


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'Saved {len(rows)} rows -> {path}')


def filter_community(records):
    return [
        rec for rec in records
        if rec.get('assigned_langs') == ['en']
        and rec['total_annotators'] >= COM_MIN_ANNOTATORS
        and rec['responses']
    ]


def unwind_coval(records):
    out = []
    for rec in records:
        for r in rec['responses']:
            out.append({
                'prompt_id': rec['prompt_id'],
                'prompt': str(rec['prompt']),
                'response_index': r['index'],
                'response': str(r['text']),
            })
    return out


def unwind_community(records):
    out = []
    for rec in records:
        for r in rec['responses']:
            out.append({
                'conversation_id': rec['conversation_ids'][0],
                'prompt': str(rec['prompt']),
                'response_index': r['index'],
                'response': str(r['text']),
            })
    return out


def unwind_openrubric(records, alias_field):
    out = []
    for rec in records:
        aliases = list(rec.get(alias_field) or [])
        if not aliases:
            continue
        for resp_field in ('response_a', 'response_b'):
            out.append({
                'prompt': str(rec['prompt']),
                'response': str(rec[resp_field]),
                'rubrics': aliases,
            })
    return out


def main():
    print('Loading datasets...')

    if COVAL_PATH.exists():
        coval_records = load_jsonl(COVAL_PATH)
        print(f'  coval: {len(coval_records):,}')
        coval_sampled = random.sample(coval_records, min(N_PROMPTS, len(coval_records)))
        write_jsonl(COVAL_OUT_PATH, unwind_coval(coval_sampled))
    else:
        print(f'  coval: skipped (missing {COVAL_PATH})')

    if COMMUNITY_PATH.exists():
        community_records = filter_community(load_jsonl(COMMUNITY_PATH))
        print(f'  community (en, >={COM_MIN_ANNOTATORS} ann): {len(community_records):,}')
        community_sampled = random.sample(community_records, min(N_PROMPTS, len(community_records)))
        write_jsonl(COMMUNITY_OUT_PATH, unwind_community(community_sampled))
    else:
        print(f'  community: skipped (missing {COMMUNITY_PATH})')

    if OPENRUBRIC_PATH.exists():
        openrubric_records = load_jsonl(OPENRUBRIC_PATH)
        print(f'  openrubric: {len(openrubric_records):,}')
        openrubric_sampled = random.sample(openrubric_records, min(N_PROMPTS, len(openrubric_records)))
        write_jsonl(OPENRUBRIC_PRINCIPLES_OUT_PATH, unwind_openrubric(openrubric_sampled, 'principle_aliases'))
        write_jsonl(OPENRUBRIC_HARD_RULES_OUT_PATH, unwind_openrubric(openrubric_sampled, 'hard_rule_aliases'))
    else:
        print(f'  openrubric: skipped (missing {OPENRUBRIC_PATH})')


if __name__ == '__main__':
    main()

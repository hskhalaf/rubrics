"""
Build a pairwise preference dataset from Facebook's Community Alignment dataset.

Each row in the CA dataset contains a prompt, four model responses (A, B, C, D), and one annotator's preferred response over the four responses.
Multiple rows can share the same (prompt, responses) tuple, each from a different annotator.

What we do:
  1. Group rows by unique (prompt + 4 responses) to collect all annotator votes per entry.
  2. For every pair of responses (6 pairs from 4 responses), count how many annotators
     preferred each side. Note: annotators pick one favorite out of four, so a vote for A
     counts as preferring A over B, A over C, and A over D — but says nothing about B vs C.
  3. Output one JSONL record per entry with the prompt, responses, and pairwise vote tallies.
"""

import json
from itertools import combinations
from datasets import load_dataset

RESPONSE_KEYS = ['A', 'B', 'C', 'D']
FIELD_TO_KEY = {
    'response_a': 'A',
    'response_b': 'B',
    'response_c': 'C',
    'response_d': 'D',
}
OUT_PATH = '/n/netscratch/calmon_lab/Everyone/hadikhalaf/rubrics/community_pairwise.jsonl'


def group_by_conversation(ds):
    """Group rows by unique (prompt + 4 responses) to collect all annotator votes."""
    convos = {}
    for row in ds:
        key = (row['first_turn_prompt'],
            row['first_turn_response_a'],
            row['first_turn_response_b'],
            row['first_turn_response_c'],
            row['first_turn_response_d'],
        )
        if key not in convos:
            convos[key] = {
                'prompt': row['first_turn_prompt'],
                'responses': {
                    'A': row['first_turn_response_a'],
                    'B': row['first_turn_response_b'],
                    'C': row['first_turn_response_c'],
                    'D': row['first_turn_response_d'],
                },
                'conv_ids': [],
                'prefs': [],
                'langs': set(),
            }
        entry = convos[key]
        entry['conv_ids'].append(row['conversation_id'])
        entry['prefs'].append(row['first_turn_preferred_response'])
        if row['assigned_lang']:
            entry['langs'].add(row['assigned_lang'])
    return convos


def build_pairwise_records(convos):
    records = []
    skipped_no_pref = 0
    skipped_missing_resp = 0

    for entry in convos.values():
        resp_texts = entry['responses']

        if any(v is None for v in resp_texts.values()):
            skipped_missing_resp += 1
            continue

        annotator_prefs = [FIELD_TO_KEY.get(p) for p in entry['prefs'] if p]
        if not annotator_prefs:
            skipped_no_pref += 1
            continue

        pair_counts = {(r1, r2): {'prefer_1': 0, 'prefer_2': 0}
                       for r1, r2 in combinations(RESPONSE_KEYS, 2)}
        for pref in annotator_prefs:
            for r1, r2 in combinations(RESPONSE_KEYS, 2):
                if pref == r1:
                    pair_counts[(r1, r2)]['prefer_1'] += 1
                elif pref == r2:
                    pair_counts[(r1, r2)]['prefer_2'] += 1

        pairs = [
            {
                'response_1_index': r1,
                'response_2_index': r2,
                'prefer_1': pc['prefer_1'],
                'prefer_2': pc['prefer_2'],
            }
            for (r1, r2), pc in pair_counts.items()
        ]

        records.append({
            'conversation_ids': entry['conv_ids'],
            'assigned_langs': sorted(entry['langs']),
            'total_annotators': len(entry['prefs']),
            'prompt': entry['prompt'],
            'responses': [{'index': idx, 'text': resp_texts[idx]} for idx in RESPONSE_KEYS],
            'pairs': pairs,
        })

    print(f"Entries: {len(records)} | Skipped no-pref: {skipped_no_pref} | Skipped missing-resp: {skipped_missing_resp}")
    return records


def main():
    print("Loading dataset...")
    ds = load_dataset("facebook/community-alignment-dataset", split="train")
    print(f"  {len(ds)} rows loaded")

    convos = group_by_conversation(ds)
    print(f"  {len(convos)} unique (prompt+responses) combos")

    records = build_pairwise_records(convos)

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"Saved -> {OUT_PATH}")

    sample = records[0].copy()
    sample['responses'] = [{'index': r['index'], 'text': r['text'][:80] + '...'} for r in sample['responses']]
    sample['conversation_ids'] = sample['conversation_ids'][:3]
    print(f"\n=== Entry 0 ===\n{json.dumps(sample, indent=2)}")

if __name__ == '__main__':
    main()

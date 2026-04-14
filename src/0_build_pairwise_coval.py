"""
Build a pairwise preference dataset from OpenAI's CoVal dataset.

Each entry contains a prompt, four model responses, and assessments from several human annotators. 

Each annotator provides two rankings: 
- a "personal" ranking (their own preference) 
- a "world" ranking (what they believe is best for the world). 

Rankings are strings like "A>B>C=D" encoding a total or partial order over responses.

What this code does:
  1. Filter to single-turn conversations (one user message, no prior assistant turns, no multi-message system prompts)
  2. For every pair of responses within an entry, aggregate annotator votes into preference counts (prefer_1, prefer_2, tie, missing) under both perspectives.
  3. Output one JSONL record per prompt containing the prompt text, all responses, and the pairwise vote tallies.
"""

import json
from itertools import combinations
from pathlib import Path

from datasets import load_dataset

OUT_PATH = Path('/n/netscratch/calmon_lab/Everyone/hadikhalaf/rubrics/data/coval.jsonl')


def parse_ranking(ranking_str):
    if not ranking_str: return {}
    ranks = {}
    for rank_idx, group in enumerate(ranking_str.strip().split('>')):
        for r in group.split('='):
            r = r.strip()
            if r: ranks[r] = rank_idx
    return ranks


def compare(ranks, r1, r2):
    if r1 not in ranks or r2 not in ranks:
        return None
    if ranks[r1] < ranks[r2]:
        return 'prefer_1'
    if ranks[r1] > ranks[r2]:
        return 'prefer_2'
    return 'tie'


def is_single_turn(entry):
    msgs = entry['prompt']['messages']
    if any(m['role'] == 'assistant' for m in msgs): return False
    if sum(1 for m in msgs if m['role'] == 'developer') > 1: return False
    user_msgs = [m['content'] for m in msgs if m['role'] == 'user']
    if len(user_msgs) != 1: return False
    if any(len(r['messages']) > 1 for r in entry['responses']): return False
    return True


def build_pairwise_records(ds):
    records = []
    skipped = 0

    for entry in ds:
        if not is_single_turn(entry):
            skipped += 1
            continue

        prompt_str = next(m['content'] for m in entry['prompt']['messages'] if m['role'] == 'user')
        responses = entry['responses']
        indices = sorted(r['response_index'] for r in responses)
        pair_counts = {
            (r1, r2): {
                'personal': {'prefer_1': 0, 'prefer_2': 0, 'tie': 0, 'missing': 0},
                'world': {'prefer_1': 0, 'prefer_2': 0, 'tie': 0, 'missing': 0},
            }
            for r1, r2 in combinations(indices, 2)
        }

        for assessment in entry['metadata']['assessments']:
            rb = assessment['ranking_blocks']
            personal_ranks = parse_ranking(rb['personal'][0]['ranking'] if rb['personal'] else '')
            world_ranks = parse_ranking(rb['world'][0]['ranking'] if rb['world'] else '')

            for r1, r2 in combinations(indices, 2):
                for perspective, ranks in [('personal', personal_ranks), ('world', world_ranks)]:
                    result = compare(ranks, r1, r2)
                    counts = pair_counts[(r1, r2)][perspective]
                    counts[result or 'missing'] += 1

        pairs = [
            {
                'response_1_index': r1,
                'response_2_index': r2,
                'personal_prefer_1': pc['personal']['prefer_1'],
                'personal_prefer_2': pc['personal']['prefer_2'],
                'personal_tie': pc['personal']['tie'],
                'personal_missing': pc['personal']['missing'],
                'world_prefer_1': pc['world']['prefer_1'],
                'world_prefer_2': pc['world']['prefer_2'],
                'world_tie': pc['world']['tie'],
                'world_missing': pc['world']['missing'],
            }
            for (r1, r2), pc in pair_counts.items()
        ]

        records.append({
            'prompt_id': entry['prompt_id'],
            'prompt': prompt_str,
            'total_annotators': len(entry['metadata']['assessments']),
            'responses': [{'index': r['response_index'], 'text': r['messages'][0]['content']} for r in responses],
            'pairs': pairs,
        })

    print(f"Entries: {len(records)} | Skipped (multi-turn): {skipped}")
    return records


def main():
    print("Loading dataset...")
    ds = load_dataset("openai/coval", "comparisons", split="train")
    print(f"  {len(ds)} entries loaded")

    records = build_pairwise_records(ds)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"Saved -> {OUT_PATH}")
    sample = records[0].copy()
    sample['responses'] = [{'index': r['index'], 'text': r['text'][:80] + '...'} for r in sample['responses']]
    sample['pairs'] = sample['pairs'][:1]
    print(f"\n=== Entry 0 ===\n{json.dumps(sample, indent=2)}")

if __name__ == '__main__':
    main()

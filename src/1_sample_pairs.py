"""
Sample a balanced subset of pairwise preferences for rubric evaluation.
Takes the pairwise datasets from Step 0 and samples N_PROMPTS prompts from each dataset, 
balanced by annotator agreement level:
  - Half from high-agreement prompts (>= 0.75): clear winner among annotators.
  - Half from low-agreement prompts  (<= 0.60): near-tie among annotators.

For each sampled prompt, up to N_PAIRS pairs are selected, prioritizing one high and
one low agreement pair when available. Each output record contains the prompt, both
response texts, the majority human label, and agreement metadata.

Output: sampled_pairs.jsonl
"""

import hashlib
import json
import random

BASE = '/n/netscratch/calmon_lab/Everyone/hadikhalaf/rubrics'
COVAL_PATH = f'{BASE}/coval_pairwise.jsonl'
COMMUNITY_PATH = f'{BASE}/community_pairwise.jsonl'
OUT_PATH = f'{BASE}/sampled_pairs.jsonl'

SEED = 42
N_PROMPTS = 100
N_PAIRS = 3           # pairs sampled per prompt
HIGH_THRESH = 0.75
LOW_THRESH = 0.60

# Community dataset filters
COM_MIN_ANNOTATORS = 5     # minimum total annotators per entry
COM_MIN_ACTIVE_VOTERS = 3  # minimum voters who expressed a preference per pair

random.seed(SEED)


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def pair_stats(p1, p2, tie=0):
    """Compute agreement and winner label for a pair."""
    active = p1 + p2 + tie
    if active == 0:
        return None
    agreement = max(p1, p2) / active
    label = 1 if p1 > p2 else (-1 if p2 > p1 else 0)
    return agreement, label, active


def enrich_coval(records):
    """Extract valid pairs from coval with agreement stats. Uses personal rankings."""
    out = []
    for rec in records:
        resp_map = {r['index']: r['text'] for r in rec['responses']}
        total_ann = rec['total_annotators']
        valid = []
        for p in rec['pairs']:
            stats = pair_stats(p['personal_prefer_1'], p['personal_prefer_2'], p['personal_tie'])
            if stats is None:
                continue
            agr, label, active = stats
            valid.append({
                'r1': p['response_1_index'],
                'r2': p['response_2_index'],
                'response_a': resp_map[p['response_1_index']],
                'response_b': resp_map[p['response_2_index']],
                'agreement': agr,
                'human_label': label,
                'active_voters': active,
                'total_annotators': total_ann,
            })
        if valid:
            out.append({'prompt': rec['prompt'], 'pairs': valid})
    return out


def enrich_community(records):
    """Extract valid pairs from community. English-only, >= COM_MIN_ANNOTATORS."""
    out = []
    for rec in records:
        if not rec['pairs']:
            continue
        if rec.get('assigned_langs') != ['en']:
            continue
        total_ann = rec['total_annotators']
        if total_ann < COM_MIN_ANNOTATORS:
            continue

        resp_map = {r['index']: r['text'] for r in rec['responses']}
        valid = []
        for p in rec['pairs']:
            stats = pair_stats(p['prefer_1'], p['prefer_2'])
            if stats is None:
                continue
            agr, label, active = stats
            if active < COM_MIN_ACTIVE_VOTERS:
                continue
            valid.append({
                'r1': p['response_1_index'],
                'r2': p['response_2_index'],
                'response_a': resp_map[p['response_1_index']],
                'response_b': resp_map[p['response_2_index']],
                'agreement': agr,
                'human_label': label,
                'active_voters': active,
                'total_annotators': total_ann,
            })
        if valid:
            out.append({'prompt': rec['prompt'], 'pairs': valid})
    return out


def sample_from(entries, n_prompts, n_pairs, tag):
    """Sample n_prompts entries (half high-agreement, half low-agreement)."""
    has_high = lambda e: any(p['agreement'] >= HIGH_THRESH for p in e['pairs'])
    has_low = lambda e: any(p['agreement'] <= LOW_THRESH for p in e['pairs'])

    high_pool = [e for e in entries if has_high(e)]
    low_pool = [e for e in entries if has_low(e)]
    print(f'  [{tag}] valid={len(entries):,}  high_pool={len(high_pool):,}  low_pool={len(low_pool):,}')

    half = n_prompts // 2
    sampled_high = random.sample(high_pool, min(half, len(high_pool)))
    used = {e['prompt'] for e in sampled_high}
    low_candidates = [e for e in low_pool if e['prompt'] not in used]
    sampled_low = random.sample(low_candidates, min(half, len(low_candidates)))

    out = []
    for entry in sampled_high + sampled_low:
        hi = [p for p in entry['pairs'] if p['agreement'] >= HIGH_THRESH]
        lo = [p for p in entry['pairs'] if p['agreement'] <= LOW_THRESH]

        chosen = []
        if hi:
            chosen.append(random.choice(hi))
        if lo:
            rest = [p for p in lo if p not in chosen]
            if rest:
                chosen.append(random.choice(rest))
        remaining = [p for p in entry['pairs'] if p not in chosen]
        random.shuffle(remaining)
        chosen.extend(remaining[:n_pairs - len(chosen)])

        for p in chosen:
            uid = hashlib.md5(
                f"{tag}|{entry['prompt'][:80]}|{p['r1']}|{p['r2']}".encode()
            ).hexdigest()[:10]
            out.append({
                'pair_id': uid,
                'dataset': tag,
                'prompt': entry['prompt'],
                'response_a': p['response_a'],
                'response_b': p['response_b'],
                'pair_key': f"{p['r1']}_vs_{p['r2']}",
                'human_label': p['human_label'],
                'agreement': round(p['agreement'], 4),
                'active_voters': p['active_voters'],
                'total_annotators': p['total_annotators'],
            })
    return out


def main():
    print('Loading datasets...')
    coval_records = load_jsonl(COVAL_PATH)
    community_records = load_jsonl(COMMUNITY_PATH)
    print(f'  coval: {len(coval_records):,} entries | community: {len(community_records):,} entries')

    coval_entries = enrich_coval(coval_records)
    community_entries = enrich_community(community_records)
    print(f'  coval valid: {len(coval_entries):,} | community valid (>={COM_MIN_ANNOTATORS} ann): {len(community_entries):,}')

    print('\nSampling...')
    coval_pairs = sample_from(coval_entries, N_PROMPTS, N_PAIRS, 'coval')
    community_pairs = sample_from(community_entries, N_PROMPTS, N_PAIRS, 'community')

    for tag, pairs in [('coval', coval_pairs), ('community', community_pairs)]:
        hi = sum(1 for p in pairs if p['agreement'] >= HIGH_THRESH)
        lo = sum(1 for p in pairs if p['agreement'] <= LOW_THRESH)
        mid = len(pairs) - hi - lo
        print(f'  [{tag}] {len(pairs)} pairs — {hi} high / {lo} low / {mid} mid agreement')

    all_pairs = coval_pairs + community_pairs
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        for rec in all_pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'\nSaved {len(all_pairs)} pairs -> {OUT_PATH}')


if __name__ == '__main__':
    main()

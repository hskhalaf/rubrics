# ignore for now, needs fixing

# """
# Learn a DNF (Disjunctive Normal Form) classifier: f̂: {0,1}^R → {0,1}.

# Given pairwise preference data and LLM judge rubric scores, learns an interpretable
# boolean formula that predicts whether a response is preferred (y=1) or rejected (y=0).

# Pipeline:
#   1. Load preference pairs from sampled_pairs.jsonl. Each pair (A preferred over B)
#      is expanded into two samples: (A, y=1) and (B, y=0). Ties are skipped.
#   2. Load judge scoring results. For each (response, rubric), majority-vote across
#      K=5 samples, then binarize: majority=1 → 1, else → 0.
#   3. Build feature matrix C ∈ {0,1}^{n×R} where R = number of rubrics.
#   4. 80/20 train/test split, fit BooleanRuleCG (column generation, DNF mode).
#   5. Print the learned DNF formula, accuracy, and per-rubric feature statistics.

# Uses aix360's BooleanRuleCG with SCS solver.

# Usage:
#   python learn_dnf.py results/results_*.jsonl
#   python learn_dnf.py results/results_*.jsonl --rubrics atomic --dataset coval
#   python learn_dnf.py results/results_*.jsonl --rubrics generic --dataset community
# """

# import argparse
# import glob
# import json
# import re
# import sys
# import warnings
# from collections import Counter, defaultdict

# warnings.filterwarnings('ignore', category=UserWarning)

# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from aix360.algorithms.rbm.boolean_rule_cg import BooleanRuleCG

# BASE = '/n/netscratch/calmon_lab/Everyone/hadikhalaf/rubrics'
# PAIRS_PATH = f'{BASE}/sampled_pairs.jsonl'
# ATOMIC_PATH = f'{BASE}/atomic_rubrics.txt'
# GENERIC_PATH = f'{BASE}/generic_rubrics.txt'

# SEED = 42
# TEST_SIZE = 0.2


# # -- Score parsing (from analyze_variance.py) --

# def parse_score(raw):
#     """Extract integer score (1, 0, -1) from a raw JSON response string."""
#     if not raw:
#         return None
#     cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE).strip()
#     try:
#         s = int(json.loads(cleaned)['score'])
#         return s if s in (1, 0, -1) else None
#     except Exception:
#         pass
#     m = re.search(r'"score"\s*:\s*(-1|0|1)', cleaned)
#     return int(m.group(1)) if m else None


# # -- Data loading --

# def load_pairs(dataset_filter=None):
#     """Load sampled pairs, skip ties, return label map: (pair_id, resp_id) → y."""
#     with open(PAIRS_PATH, encoding='utf-8') as f:
#         pairs = [json.loads(line) for line in f]

#     labels = {}
#     skipped = 0
#     for p in pairs:
#         if dataset_filter and p.get('dataset') not in dataset_filter:
#             continue
#         hl = p['human_label']
#         if hl == 0:
#             skipped += 1
#             continue
#         labels[(p['pair_id'], 'A')] = 1 if hl == 1 else 0
#         labels[(p['pair_id'], 'B')] = 0 if hl == 1 else 1

#     ds_label = ', '.join(dataset_filter) if dataset_filter else 'all'
#     print(f'Pairs: {len(pairs)} total, {skipped} ties skipped -> {len(labels)} samples (dataset: {ds_label})')
#     return labels


# def load_results(paths, rubric_filter=None):
#     """Load results JSONL files, parse scores. Optionally filter by rubric type."""
#     records = []
#     for pattern in paths:
#         for path in sorted(glob.glob(pattern)) or [pattern]:
#             with open(path, encoding='utf-8') as f:
#                 for line in f:
#                     r = json.loads(line)
#                     if rubric_filter and r.get('rubric_type') not in rubric_filter:
#                         continue
#                     score = parse_score(r.get('raw_response', ''))
#                     if score is not None:
#                         r['score'] = score
#                         records.append(r)
#     rtypes = rubric_filter or ['atomic', 'generic']
#     print(f'Loaded {len(records):,} scored records (rubrics: {", ".join(rtypes)})')
#     return records


# def load_rubric_names():
#     """Load rubric text for display. Returns dict: (type, index) → short name."""
#     names = {}
#     for rtype, path in [('atomic', ATOMIC_PATH), ('generic', GENERIC_PATH)]:
#         with open(path, encoding='utf-8') as f:
#             for i, line in enumerate(f, 1):
#                 line = line.strip()
#                 if line:
#                     text = re.sub(r'^\d+[.)]\s*', '', line)
#                     names[(rtype, i)] = text
#     return names


# # -- Feature matrix construction --

# def build_features(records, labels):
#     """Build binary feature matrix and label vector from scored records.

#     For each (pair_id, response_id, rubric_type, rubric_index), takes the
#     majority vote across K samples, then binarizes: score=1 → 1, else → 0.

#     Returns: (configs, features_by_config)
#       configs: list of (model, cot) tuples
#       features_by_config: dict (model,cot) → (X DataFrame, y array, sample_keys)
#     """
#     # Group scores by config and sample
#     # key: (model, cot, pair_id, resp_id, rtype, ridx) → [scores]
#     groups = defaultdict(list)
#     for r in records:
#         key = (r['model'], r['cot'], r['pair_id'], r['response_id'],
#                r['rubric_type'], r['rubric_index'])
#         groups[key].append(r['score'])

#     # Collect all rubric columns (sorted consistently)
#     rubric_cols = sorted({(k[4], k[5]) for k in groups.keys()})

#     # Build per-config feature matrices
#     config_data = defaultdict(lambda: defaultdict(dict))  # (model,cot) → (pair_id,resp_id) → {col: val}
#     for key, scores in groups.items():
#         model, cot, pair_id, resp_id, rtype, ridx = key
#         # Majority vote then binarize: 1 if majority says 1, else 0
#         majority = Counter(scores).most_common(1)[0][0]
#         binary = 1 if majority == 1 else 0
#         config_data[(model, cot)][(pair_id, resp_id)][(rtype, ridx)] = binary

#     results = {}
#     for config, sample_features in config_data.items():
#         # Inner join with labels
#         sample_keys = []
#         rows = []
#         ys = []
#         for (pid, rid), feats in sample_features.items():
#             if (pid, rid) not in labels:
#                 continue
#             sample_keys.append((pid, rid))
#             rows.append([feats.get(col, 0) for col in rubric_cols])
#             ys.append(labels[(pid, rid)])

#         # BooleanRuleCG expects 3-level MultiIndex: (feature, operator, threshold)
#         col_names = pd.MultiIndex.from_tuples(
#             [(f'{rtype}_{ridx}', '>=', 1) for rtype, ridx in rubric_cols]
#         )
#         X = pd.DataFrame(rows, columns=col_names).astype(int)
#         y = np.array(ys)

#         results[config] = (X, y, sample_keys, rubric_cols)
#         model, cot = config
#         print(f'  [{model} cot={cot}] {X.shape[0]} samples x {X.shape[1]} features, '
#               f'y=1: {y.sum()}, y=0: {len(y) - y.sum()}')

#     return results


# # -- Training and evaluation --

# def train_and_evaluate(X, y, rubric_cols, rubric_names, config_label):
#     """Train BooleanRuleCG, evaluate, and print results."""
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
#     )
#     print(f'\n  Train: {len(y_train)} | Test: {len(y_test)}')

#     clf = BooleanRuleCG(lambda0=0.001, lambda1=0.001, CNF=False, silent=True, solver='SCS')
#     clf.fit(X_train, y_train)

#     y_pred_train = clf.predict(X_train)
#     y_pred_test = clf.predict(X_test)

#     print(f'\n  Train accuracy: {accuracy_score(y_train, y_pred_train):.4f}')
#     print(f'  Test  accuracy: {accuracy_score(y_test, y_pred_test):.4f}')
#     print(f'\n  Classification report (test):')
#     print(classification_report(y_test, y_pred_test, target_names=['rejected', 'preferred'], digits=4))

#     # Print learned DNF
#     explanation = clf.explain()
#     print(f'  Learned {"CNF" if explanation["isCNF"] else "DNF"}:')
#     for i, rule in enumerate(explanation['rules'], 1):
#         print(f'    Clause {i}: {rule}')

#     # Per-rubric feature stats
#     print(f'\n  Per-rubric stats (fraction=1 in preferred vs rejected):')
#     preferred_mask = y == 1
#     rejected_mask = y == 0
#     for j, (rtype, ridx) in enumerate(rubric_cols):
#         col = X.iloc[:, j]
#         pref_rate = col[preferred_mask].mean()
#         rej_rate = col[rejected_mask].mean()
#         name = rubric_names.get((rtype, ridx), f'{rtype}_{ridx}')
#         print(f'    {rtype}_{ridx}: pref={pref_rate:.3f}  rej={rej_rate:.3f}  '
#               f'diff={pref_rate - rej_rate:+.3f}  | {name[:60]}')

#     return clf


# def main():
#     parser = argparse.ArgumentParser(description='Learn DNF from rubric scores')
#     parser.add_argument('results', nargs='+', help='Results JSONL file(s) or glob patterns')
#     parser.add_argument('--rubrics', choices=['atomic', 'generic', 'all'], default='all',
#                         help='Which rubric types to use as features (default: all)')
#     parser.add_argument('--dataset', choices=['coval', 'community', 'all'], default='all',
#                         help='Which dataset to use (default: all)')
#     args = parser.parse_args()

#     rubric_filter = None if args.rubrics == 'all' else [args.rubrics]
#     dataset_filter = None if args.dataset == 'all' else [args.dataset]

#     labels = load_pairs(dataset_filter)
#     records = load_results(args.results, rubric_filter)
#     rubric_names = load_rubric_names()

#     print('\nBuilding feature matrices...')
#     config_results = build_features(records, labels)

#     for config in sorted(config_results.keys()):
#         model, cot = config
#         X, y, sample_keys, rubric_cols = config_results[config]
#         print(f'\n{"="*70}')
#         print(f'  Config: model={model}  cot={cot}')
#         print(f'{"="*70}')
#         train_and_evaluate(X, y, rubric_cols, rubric_names, f'{model}_cot={cot}')


# if __name__ == '__main__':
#     main()

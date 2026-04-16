"""
Learn a DNF (Disjunctive Normal Form) classifier: f̂: {0,1}^R → {0,1}.

Takes a judged JSONL file (output of 2_judge.py) where each row has:
  - judge: {alias: [{content, reasoning}, ...K items], ...}
  - label: 0 or 1   (preferred=1, rejected=0)

Rows without a valid `label` field are skipped.

Pipeline:
  1. Parse judge scores: for each (row, rubric), majority-vote across K samples,
     then binarize: majority=1 → feature=1, else → feature=0.
  2. Build binary feature matrix C ∈ {0,1}^{n×R} (R = number of rubric aliases).
  3. 80/20 train/test split, fit BooleanRuleCG (column generation, DNF mode).
  4. Print the learned DNF formula, accuracy, and per-rubric feature statistics.

Usage:
  python 3_learn_dnf.py data/judged/sampled_openrubric__Qwen_Qwen3-0-6B_nocot.jsonl
"""

import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from aix360.algorithms.rbm.boolean_rule_cg import BooleanRuleCG

SEED = 42
TEST_SIZE = 0.2


def parse_score(raw):
    """Extract integer score (1, 0, -1) from a raw JSON response string."""
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


def majority_binary(samples):
    """Parse K samples, majority-vote, binarize: majority=1 → 1, else → 0."""
    scores = [parse_score((s or {}).get('content') or '') for s in samples]
    scores = [s for s in scores if s is not None]
    if not scores:
        return None
    majority = Counter(scores).most_common(1)[0][0]
    return 1 if majority == 1 else 0


def load_judged(path):
    """Load judged JSONL, return (feature_rows, labels, aliases).

    Each feature_row is a dict {alias: 0|1} from majority-vote binarization.
    Only rows with label ∈ {0, 1} are kept.
    """
    feature_rows = []
    labels = []
    all_aliases = set()
    skipped_no_label = 0
    skipped_no_judge = 0

    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            label = rec.get('label')
            if label not in (0, 1):
                skipped_no_label += 1
                continue
            judge = rec.get('judge') or {}
            if not judge:
                skipped_no_judge += 1
                continue

            feats = {}
            for alias, samples in judge.items():
                val = majority_binary(samples)
                if val is not None:
                    feats[alias] = val
                    all_aliases.add(alias)

            if feats:
                feature_rows.append(feats)
                labels.append(label)

    aliases = sorted(all_aliases)
    print(f'Loaded {len(labels)} labelled rows from {Path(path).name}')
    print(f'  skipped: {skipped_no_label} (no label) + {skipped_no_judge} (no judge)')
    print(f'  rubric features: {len(aliases)}')
    return feature_rows, labels, aliases


def build_matrix(feature_rows, labels, aliases):
    """Build DataFrame X (BooleanRuleCG format) and array y."""
    col_names = pd.MultiIndex.from_tuples(
        [(alias, '>=', 1) for alias in aliases]
    )
    rows = [[feats.get(alias, 0) for alias in aliases] for feats in feature_rows]
    X = pd.DataFrame(rows, columns=col_names).astype(int)
    y = np.array(labels)
    print(f'  matrix: {X.shape[0]} rows x {X.shape[1]} features | y=1: {y.sum()} y=0: {len(y) - y.sum()}')
    return X, y


def train_and_evaluate(X, y, aliases):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f'\n  train: {len(y_train)} | test: {len(y_test)}')

    clf = BooleanRuleCG(lambda0=0.001, lambda1=0.001, CNF=False, silent=True, solver='SCS')
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    print(f'\n  train accuracy: {accuracy_score(y_train, y_pred_train):.4f}')
    print(f'  test  accuracy: {accuracy_score(y_test, y_pred_test):.4f}')
    print(f'\n  classification report (test):')
    print(classification_report(y_test, y_pred_test, target_names=['rejected', 'preferred'], digits=4))

    explanation = clf.explain()
    print(f'  learned {"CNF" if explanation["isCNF"] else "DNF"}:')
    for i, rule in enumerate(explanation['rules'], 1):
        print(f'    clause {i}: {rule}')

    # Per-rubric: fraction=1 in preferred vs rejected
    print(f'\n  per-rubric feature rates:')
    pref_mask = y == 1
    rej_mask = y == 0
    for j, alias in enumerate(aliases):
        col = X.iloc[:, j]
        pref_rate = col[pref_mask].mean() if pref_mask.any() else 0
        rej_rate = col[rej_mask].mean() if rej_mask.any() else 0
        print(f'    {alias:40} pref={pref_rate:.3f}  rej={rej_rate:.3f}  diff={pref_rate - rej_rate:+.3f}')

    return clf


def main():
    paths = sys.argv[1:]
    if not paths:
        print('Usage: python 3_learn_dnf.py <judged.jsonl> [...]')
        sys.exit(1)

    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f'[skip] {path} not found')
            continue

        print(f'\n{"="*70}')
        print(f'  {path.name}')
        print(f'{"="*70}')

        feature_rows, labels, aliases = load_judged(path)
        if len(labels) < 10:
            print('  too few labelled rows, skipping')
            continue

        X, y = build_matrix(feature_rows, labels, aliases)
        train_and_evaluate(X, y, aliases)


if __name__ == '__main__':
    main()

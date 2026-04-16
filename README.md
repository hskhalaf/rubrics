# Rubric-Based Preference Analysis

Learn interpretable rules that explain human preference data using LLM-as-judge rubric scores.

## Setup

```bash
pip install datasets openai pyyaml numpy pandas scikit-learn cvxpy aix360
```

A vLLM server is required for the judging step.

## Directory Layout

```
rubric_items/              # rubric definitions (yaml)
  atomic/                  #   hand-written atomic criteria
  generic/                 #   hand-written generic criteria
  openrubric/              #   extracted from OpenRubric-v2
    principles/
    hard_rules/
  healthbench/             #   extracted from HealthBench
data/                      # generated datasets
  *.jsonl                  #   stage 0 pairwise / raw outputs
  sampled/                 #   stage 1 sampled (prompt, response) rows
  judged/                  #   stage 2 judge outputs
src/                       # pipeline scripts
```

## Pipeline

### 0. Build raw datasets

Extract pairwise preference records and rubric criteria from source datasets.

```bash
python src/0_build_pairwise_coval.py        # -> data/coval.jsonl
python src/0_build_pairwise_community.py    # -> data/community.jsonl
python src/0_build_openrubric.py            # -> data/openrubric.jsonl + rubric_items/openrubric/
python src/0_build_healthbench.py           # -> data/healthbench.jsonl + rubric_items/healthbench/
```

### 1. Sample and unwind

Sample N prompts per dataset and unwind to one (prompt, response) row per response.

```bash
python src/1_sample_responses.py            # -> data/sampled/sampled_*.jsonl
```

### 2. Score with LLM judge

Start a vLLM server with prefix caching, then score each row on every rubric in the given folder.

```bash
# Start vLLM server (separate terminal)
vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3 --enable-prefix-caching

# Score — one run per (dataset, rubric set) combo
python src/2_judge.py --data data/sampled/sampled_coval.jsonl \
    --rubrics rubric_items/atomic --model Qwen/Qwen3-0.6B --no-cot -k 5

python src/2_judge.py --data data/sampled/sampled_openrubric_principles.jsonl \
    --rubrics rubric_items/openrubric/principles --model Qwen/Qwen3-0.6B --no-cot -k 5

# Resume a crashed run (reuses already-scored (row, rubric) pairs)
python src/2_judge.py --data ... --model ... --resume
```

Output: `data/judged/<input_stem>__<model>_<cot>.jsonl`. Each row is the original input row with a new `judge` field mapping rubric aliases to K raw judge outputs.

### 3. Analyze judge variance

Measure expected inconsistency per rubric across K repeated samples.

```bash
python src/3_analyze_variance.py data/judged/*.jsonl
```

### 4. Learn DNF

Learn an interpretable DNF formula predicting human preference from rubric scores. Requires a `label` field (0 or 1) in each judged row.

```bash
python src/3_learn_dnf.py data/judged/sampled_openrubric__Qwen_Qwen3-0-6B_nocot.jsonl
```

## Files

| File | Description |
|------|-------------|
| `src/0_build_pairwise_coval.py` | Extract pairwise preferences from OpenAI CoVal |
| `src/0_build_pairwise_community.py` | Extract pairwise preferences from Facebook Community Alignment |
| `src/0_build_openrubric.py` | Extract principle + hard-rule criteria from OpenRubric-v2 |
| `src/0_build_healthbench.py` | Extract rubric criteria from HealthBench consensus set |
| `src/1_sample_responses.py` | Sample prompts and unwind to (prompt, response) rows |
| `src/2_judge.py` | LLM-as-judge rubric scoring via vLLM (generic, resumable) |
| `src/3_analyze_variance.py` | Measure judge scoring inconsistency across K samples |
| `src/3_learn_dnf.py` | Learn DNF classifier from rubric scores + binary labels |

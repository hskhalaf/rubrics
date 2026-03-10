# Rubric-Based Preference Analysis

Learn interpretable rules that explain human preference data using LLM-as-judge rubric scores.

## Setup

```bash
pip install datasets openai numpy pandas scikit-learn cvxpy aix360
```

A vLLM server is required for the judging step (step 3).

## Pipeline

Run scripts in this order:

### 1. Build pairwise datasets

Extract pairwise preference records from the raw HuggingFace datasets.

```bash
python build_pairwise_coval.py        # -> coval_pairwise.jsonl
python build_pairwise_community.py    # -> community_pairwise.jsonl
```

### 2. Sample pairs

Sample a balanced subset (by annotator agreement) for evaluation.

```bash
python sample_pairs.py                # -> sampled_pairs.jsonl
```

### 3. Score with LLM judge

Start a vLLM server, then score each (response, rubric) pair. Run once per model/cot config.

```bash
# Start vLLM server (separate terminal)
vllm serve Qwen/Qwen3-0.6B --port 8000

# Score with and without chain-of-thought
python judge.py --model Qwen/Qwen3-0.6B --cot    --port 8000
python judge.py --model Qwen/Qwen3-0.6B --no-cot --port 8000
```

Output goes to `results/results_<model>_<cot|nocot>.jsonl`.

### 4. Analyze judge variance

Check how consistently the judge scores each rubric across K=5 repeated samples.

```bash
python analyze_variance.py results/results_*.jsonl
python analyze_variance.py results/results_Qwen_Qwen3-0-6B_cot.jsonl   # single config
```

### 5. Learn DNF

Learn an interpretable DNF formula predicting human preference from rubric scores.

```bash
# All rubrics, all datasets
python learn_dnf.py results/results_*.jsonl

# Filter by rubric type and/or dataset
python learn_dnf.py results/results_*.jsonl --rubrics atomic --dataset coval
python learn_dnf.py results/results_*.jsonl --rubrics generic --dataset community
```

## Files

| File | Description |
|------|-------------|
| `build_pairwise_coval.py` | Extract pairwise preferences from OpenAI CoVal |
| `build_pairwise_community.py` | Extract pairwise preferences from Facebook Community Alignment |
| `sample_pairs.py` | Sample agreement-balanced subset for evaluation |
| `judge.py` | LLM-as-judge rubric scoring via vLLM |
| `analyze_variance.py` | Measure judge scoring variance across repeated samples |
| `learn_dnf.py` | Learn DNF classifier from rubric scores |
| `atomic_rubrics.txt` | 10 atomic rubric criteria (factual, safety, formatting) |
| `generic_rubrics.txt` | 10 generic rubric criteria (quality, tone, accuracy) |

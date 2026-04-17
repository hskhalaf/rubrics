"""
Score each row of a (prompt, response) dataset on rubrics using a vLLM-served LLM as judge.

Input dataset is a JSONL file where each row has at least:
  - prompt    (str)
  - response  (str)
  - rubrics   (optional list of rubric aliases; if present, only those rubrics are scored)

Rubrics live under a folder (--rubrics) and are searched recursively for *.yaml files.
Each yaml defines `alias`, `criterion`, and optional `good_examples` / `bad_examples`.

K samples per (row, rubric) are obtained in a single API call. You can choose to turn off thinking.

Output: add a new column
  judge: {
    <rubric_alias>: [
      {"content": <str>, "reasoning": <str|None>},
      ...  # K items
    ],
    ...
  }

Rows are written as they complete, so crashes don't lose progress.
Pass --resume to continue from an existing output file: already-scored
(row, rubric) pairs are kept, only missing ones are recomputed.

Usage:
  python 2_judge.py --data data/sampled/sampled_coval.jsonl --model Qwen/Qwen3-0.6B --no-cot -k 5
  python 2_judge.py --data ... --model ... --resume
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import yaml
from openai import AsyncOpenAI
from tqdm import tqdm

BASE = Path('/n/netscratch/calmon_lab/Lab/rubrics')
DEFAULT_RUBRICS_DIR = BASE / 'rubric_items'
JUDGED_DIR = BASE / 'data' / 'judged'
SPECEVAL_PROVIDERS = ('anthropic', 'google', 'openai')

CONCURRENCY = 512
TEMPERATURE = 0.8
MAX_TOKENS_COT = 24000
MAX_TOKENS_NOCOT = 15000


SYSTEM_PROMPT = """\
You are a response quality evaluator.

Given a user prompt and a model response, decide whether the response satisfies the stated quality criterion.

Return a JSON object with exactly two fields:
  "rationale": a concise explanation of your judgment (1-3 sentences)
  "score": an integer 1 if the response satisfies the criterion, -1 if it does not, 0 if the criterion does not apply to this case

Output valid JSON only without any other text."""

USER_TEMPLATE = """\
## User Prompt
{prompt}

## Response
{response}

## Quality Criterion
{criterion}
{examples_block}
Return JSON with "rationale" and "score":"""


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_rubrics(rubrics_dir):
    rubrics = []
    seen_aliases = set()
    for yaml_path in sorted(rubrics_dir.rglob('*.yaml')):
        data = yaml.safe_load(yaml_path.read_text())
        if data['alias'] != yaml_path.stem:
            raise ValueError(f'{yaml_path}: alias {data["alias"]!r} != filename {yaml_path.stem!r}')
        if data['alias'] in seen_aliases:
            raise ValueError(f'{yaml_path}: duplicate alias {data["alias"]!r}')
        seen_aliases.add(data['alias'])
        rubrics.append({
            'alias': data['alias'],
            'criterion': data['criterion'],
            'good_examples': data.get('good_examples') or [],
            'bad_examples': data.get('bad_examples') or [],
        })
    return rubrics


def format_examples(good, bad):
    if not good and not bad:
        return ''
    lines = ['\n## Examples']
    for ex in good:
        lines.append(f"[SATISFIES]\nPrompt: {ex['prompt']}\nResponse: {ex['response']}\n")
    for ex in bad:
        lines.append(f"[VIOLATES]\nPrompt: {ex['prompt']}\nResponse: {ex['response']}\n")
    return '\n'.join(lines) + '\n'


def select_rubrics_for_row(row, all_rubrics, rubric_by_alias):
    aliases = row.get('rubrics')
    if aliases is None:
        return all_rubrics
    missing = [a for a in aliases if a not in rubric_by_alias]
    if missing:
        raise ValueError(f'Unknown rubric aliases in row: {missing}')
    return [rubric_by_alias[a] for a in aliases]


def row_key(row):
    h = hashlib.sha256()
    h.update(row['prompt'].encode('utf-8'))
    h.update(b'\x00')
    h.update(row['response'].encode('utf-8'))
    return h.hexdigest()[:16]


def load_existing_judge(out_path):
    """Return {row_key: {alias: samples}} from an existing output file."""
    existing = {}
    if not out_path.exists():
        return existing
    for line in out_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if 'prompt' in rec and 'response' in rec and 'judge' in rec:
            existing[row_key(rec)] = rec['judge']
    return existing


def build_messages(row, rubric):
    examples_block = format_examples(rubric['good_examples'], rubric['bad_examples'])
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': USER_TEMPLATE.format(
            prompt=row['prompt'], response=row['response'],
            criterion=rubric['criterion'], examples_block=examples_block)},
    ]


async def run(rows, row_state, pending, work, out_path, model, cot, k, port):
    client = AsyncOpenAI(base_url=f'http://localhost:{port}/v1', api_key='EMPTY')
    sem = asyncio.Semaphore(CONCURRENCY)
    max_tokens = MAX_TOKENS_COT if cot else MAX_TOKENS_NOCOT

    total = len(work)
    completed = 0
    lock = asyncio.Lock()
    f = open(out_path, 'w', encoding='utf-8')

    def write_row(idx):
        f.write(json.dumps({**rows[idx], 'judge': row_state[idx]}, ensure_ascii=False) + '\n')
        f.flush()

    # Pre-write rows that have no pending work (already fully covered by resume state).
    for idx in range(len(rows)):
        if pending[idx] == 0:
            write_row(idx)

    async def score_one(row_idx, alias, msgs):
        nonlocal completed
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model, messages=msgs, max_tokens=max_tokens,
                    n=k, temperature=TEMPERATURE,
                    extra_body={'chat_template_kwargs': {'enable_thinking': cot}},
                )
                samples = [
                    {
                        'content': choice.message.content,
                        'reasoning': getattr(choice.message, 'reasoning_content', None),
                    }
                    for choice in resp.choices
                ]
            except Exception as e:
                samples = [{'content': f'ERROR: {e}', 'reasoning': None} for _ in range(k)]

        async with lock:
            row_state[row_idx][alias] = samples
            pending[row_idx] -= 1
            completed += 1
            if pending[row_idx] == 0:
                write_row(row_idx)
            if completed % 500 == 0:
                print(f'  {completed:,}/{total:,}')

    try:
        await asyncio.gather(*(score_one(r, a, m) for r, a, m in work))
    finally:
        f.close()


def main():
    parser = argparse.ArgumentParser(description='Score responses on rubrics via vLLM')
    parser.add_argument('--provider', choices=SPECEVAL_PROVIDERS, default=None,
                        help=f'speceval provider {SPECEVAL_PROVIDERS}; auto-sets --data and --rubrics')
    parser.add_argument('--data', type=Path, default=None, help='input jsonl with prompt/response rows')
    parser.add_argument('--out', type=Path, default=None, help='output jsonl (default: data/judged/<stem>_<model>_<cot>.jsonl)')
    parser.add_argument('--model', required=True)
    parser.add_argument('--cot', action='store_true', dest='cot')
    parser.add_argument('--no-cot', action='store_false', dest='cot')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('-k', '--k', type=int, default=5, help='samples per (row, rubric)')
    parser.add_argument('--rubrics', type=Path, default=None, help='folder containing rubric yaml files (searched recursively)')
    parser.add_argument('--resume', action='store_true', help='reuse already-scored (row, rubric) pairs from the output file')
    parser.set_defaults(cot=False)
    args = parser.parse_args()

    if args.provider:
        args.data = args.data or (BASE / 'data' / 'speceval' / args.provider / 'speceval.jsonl')
        args.rubrics = args.rubrics or (BASE / 'rubric_items' / 'speceval' / args.provider)
    else:
        if args.data is None:
            parser.error('--data is required when --provider is not specified')
        args.rubrics = args.rubrics or DEFAULT_RUBRICS_DIR

    all_rubrics = load_rubrics(args.rubrics)
    rubric_by_alias = {r['alias']: r for r in all_rubrics}
    print(f'Rubrics: {len(all_rubrics)} loaded from {args.rubrics}')

    rows = load_jsonl(args.data)

    slug = args.model.replace('/', '_').replace('.', '-')
    cot_tag = 'cot' if args.cot else 'nocot'
    prefix = f'speceval_{args.provider}' if args.provider else args.data.stem
    out_path = args.out or (JUDGED_DIR / f'{prefix}_{slug}_{cot_tag}.jsonl')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_existing_judge(out_path) if args.resume else {}
    if args.resume:
        print(f'Resume: {len(existing)} prior rows loaded from {out_path}')

    row_state = []
    pending = []
    work = []
    for idx, row in tqdm(enumerate(rows)):
        if 'prompt' not in row or 'response' not in row:
            raise ValueError(f'row {idx} missing prompt/response')
        prior = existing.get(row_key(row), {})
        state = {}
        missing = 0
        for rubric in select_rubrics_for_row(row, all_rubrics, rubric_by_alias):
            if rubric['alias'] in prior:
                state[rubric['alias']] = prior[rubric['alias']]
            else:
                work.append((idx, rubric['alias'], build_messages(row, rubric)))
                missing += 1
        row_state.append(state)
        pending.append(missing)

    reused = sum(len(s) for s in row_state)
    print(f'Rows: {len(rows)} | new work items: {len(work)} | reused: {reused} | k={args.k} | temp={TEMPERATURE}')
    print(f'Model: {args.model} | cot: {args.cot} | concurrency: {CONCURRENCY}')

    asyncio.run(run(rows, row_state, pending, work, out_path, args.model, args.cot, args.k, args.port))
    print(f'Saved {len(rows)} rows -> {out_path}')


if __name__ == '__main__':
    main()

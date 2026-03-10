"""
Score each (prompt, response, rubric) pair using a vLLM-served LLM as judge.

Takes the sampled preference pairs (sampled_pairs.jsonl) and evaluates each response
independently against every rubric (atomic + generic).  
K=5 independent samples are drawn to measure scoring variance.

The judge returns a JSON with a rationale and a score:
  1  = response satisfies the criterion
  -1 = response violates the criterion
  0  = criterion does not apply

Requires a vLLM server running the target model!

Usage:
  python judge.py --model Qwen/Qwen3-0.6B --cot   --port 8000
  python judge.py --model Qwen/Qwen3-0.6B --no-cot --port 8000
"""

import argparse
import asyncio
import json
import re

from openai import AsyncOpenAI

BASE = '/n/netscratch/calmon_lab/Everyone/hadikhalaf/rubrics'
PAIRS_PATH = f'{BASE}/sampled_pairs.jsonl'
ATOMIC_PATH = f'{BASE}/atomic_rubrics.txt'
GENERIC_PATH = f'{BASE}/generic_rubrics.txt'

K = 5               # independent samples per (prompt, response, rubric)
CONCURRENCY = 512   # max simultaneous requests
MAX_TOKENS_COT = 15000
MAX_TOKENS_NOCOT = 24000
MAX_PAIRS = None   


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
{rubric}

Return JSON with "rationale" and "score":"""


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_rubrics(path):
    with open(path, encoding='utf-8') as f:
        return [re.sub(r'^\d+[.)]\s*', '', line.strip()) for line in f if line.strip()]


def build_work_items(pairs, rubrics_atomic, rubrics_generic, model, cot):
    work = []
    for pair in pairs:
        for resp_id, resp_text in [('A', pair['response_a']), ('B', pair['response_b'])]:
            for rtype, rubrics in [('atomic', rubrics_atomic), ('generic', rubrics_generic)]:
                for ridx, rtxt in enumerate(rubrics, 1):
                    msgs = [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': USER_TEMPLATE.format(
                            prompt=pair['prompt'], response=resp_text, rubric=rtxt)},
                    ]
                    meta = {
                        'pair_id': pair['pair_id'],
                        'dataset': pair['dataset'],
                        'human_label': pair['human_label'],
                        'agreement': pair['agreement'],
                        'response_id': resp_id,
                        'rubric_type': rtype,
                        'rubric_index': ridx,
                        'rubric_text': rtxt,
                        'model': model,
                        'cot': cot,
                    }
                    for sk in range(1, K + 1):
                        work.append((msgs, {**meta, 'sample_k': sk}))
    return work


async def run(pairs, rubrics_atomic, rubrics_generic, model, cot, port, out_path):
    client = AsyncOpenAI(base_url=f'http://localhost:{port}/v1', api_key='EMPTY')
    sem = asyncio.Semaphore(CONCURRENCY)
    write_lock = asyncio.Lock()
    max_tokens = MAX_TOKENS_COT if cot else MAX_TOKENS_NOCOT

    work = build_work_items(pairs, rubrics_atomic, rubrics_generic, model, cot)
    total = len(work)
    completed = 0
    n_rubrics = len(rubrics_atomic) + len(rubrics_generic)
    print(f'Total calls: {total:,}  ({len(pairs)} pairs x 2 responses x {n_rubrics} rubrics x {K} samples)')

    async def score_one(msgs, meta):
        nonlocal completed
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model, messages=msgs, max_tokens=max_tokens,
                    extra_body={'chat_template_kwargs': {'enable_thinking': cot}},
                )
                content = resp.choices[0].message.content
                reasoning = getattr(resp.choices[0].message, 'reasoning_content', None)
            except Exception as e:
                content = f'ERROR: {e}'
                reasoning = None

        async with write_lock:
            f.write(json.dumps({**meta, 'raw_response': content, 'reasoning': reasoning}, ensure_ascii=False) + '\n')
            f.flush()
            completed += 1
            if completed % 1000 == 0:
                print(f'  {completed:,}/{total:,}')

    with open(out_path, 'w', encoding='utf-8') as f:
        await asyncio.gather(*(score_one(m, meta) for m, meta in work))
    print(f'Done — saved {total:,} records -> {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Score responses on rubrics via vLLM')
    parser.add_argument('--model', required=True)
    parser.add_argument('--cot', action='store_true', dest='cot')
    parser.add_argument('--no-cot', action='store_false', dest='cot')
    parser.add_argument('--port', type=int, default=8000)
    parser.set_defaults(cot=False)
    args = parser.parse_args()

    pairs = load_jsonl(PAIRS_PATH)[:MAX_PAIRS]
    rubrics_atomic = load_rubrics(ATOMIC_PATH)
    rubrics_generic = load_rubrics(GENERIC_PATH)

    print(f'Pairs: {len(pairs)} | Atomic: {len(rubrics_atomic)} | Generic: {len(rubrics_generic)}')
    print(f'Model: {args.model} | CoT: {args.cot} | k={K} | concurrency={CONCURRENCY}')

    slug = args.model.replace('/', '_').replace('.', '-')
    cot_tag = 'cot' if args.cot else 'nocot'
    out_path = f'{BASE}/results_{slug}_{cot_tag}.jsonl'

    asyncio.run(run(pairs, rubrics_atomic, rubrics_generic, args.model, args.cot, args.port, out_path))


if __name__ == '__main__':
    main()

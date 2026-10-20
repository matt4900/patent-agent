"""
Evaluate a model with Langfuse using an LLM-as-a-judge.

Make sure to run:
    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...
    export LANGFUSE_HOST=https://[url]

"""

import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Mxfp4Config, BitsAndBytesConfig
import pprint
import os
import sys
from peft import PeftModel, AutoPeftModelForCausalLM
from accelerate import Accelerator, cpu_offload

from safetensors.torch import load_file, save_file
from safetensors import safe_open
import os
import requests
import time

from langfuse import Evaluation, get_client

model_name = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"

langfuse = get_client()

torch.set_num_threads(1)

with open("outcompletedesc.jsonl", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

pp = pprint.PrettyPrinter(indent=4, sort_dicts=False, width=100)

# map our fine tuned dataset output because a langfuse dataset has set columns and we need to get the output response from the input
finetuned_responses = { r["title"].strip(): r["new_description"] for r in rows}

JUDGE_MODEL = model_name                   # grades every answer
BASE_MODEL = model_name                    # not fine-tuned (same model as the judge)
FINE_TUNED_MODEL = "mimo-patent-model7"

DATASET_NAME = "patents1-desc"

def get_or_create_dataset():
    try:
        return langfuse.get_dataset(DATASET_NAME)
    except Exception:
        langfuse.create_dataset(name=DATASET_NAME)
        for row in rows:
            langfuse.create_dataset_item(
                dataset_name=DATASET_NAME,
                input="Write the description for: " + row["title"] + "\nAbstract:" + row["abstract"],
                expected_output=row["description"],
                metadata={"title": row["title"] },
            )
        return langfuse.get_dataset(DATASET_NAME)


def base_task(*, item, **kwargs):
    prompt = item["input"] if isinstance(item, dict) else item.input

    message =  [
                    { 'content': "You are a patent writer. You take a title and abstract for a patent and generate the description for the patent.", 
                        'role': 'system',
                        'thinking': None },
                    { 'content': prompt, 
                        'role': 'user',
                        'thinking': None }
                ]

    resp = requests.post(
        "http://172.17.0.3:1090/v1/chat/completions",
        json={
            "messages": message,
            "max_tokens": 6000,
            "temperature": 0.6,
        },
        timeout=1000,
    )
    resp.raise_for_status()
    response = resp.json()["choices"][0]["message"]["content"]
    print(response)
    time.sleep(5)

    return response
 
 
def finetuned_task(*, item, **kwargs):
    """No model call: returns the stored fine-tuned response for this prompt."""
    meta = item["metadata"] if isinstance(item, dict) else item.metadata
    title = meta["title"]

    if title not in finetuned_responses:
        raise KeyError(f"No fine-tuned response for prompt: {title[:80]!r}")
    
    return finetuned_responses[title]

    
JUDGE_PROMPT = """You are grading an AI assistant's answer against a reference answer.

QUESTION:
{input}

REFERENCE ANSWER (ground truth):
{expected_output}

ASSISTANT'S ANSWER:
{output}

Grade how well the assistant's answer matches the reference in meaning and factual content.
Wording may differ. Penalize wrong, missing, or contradictory facts. Do not reward length.

Scoring:
  1.0 = fully correct and complete
  0.8 = close or almost the reference answer
  0.5 = partially correct or missing some details
  0.2 = details deviate significantly from reference answer
  0.0 = incorrect or off-topic

Respond with JSON only: {{"score": <0.0|0.2|0.5|0.8|1.0>, "reasoning": "<one or two sentences>"}}"""


def llm_judge_correctness(*, input, output, expected_output, metadata=None, **kwargs):
    if not expected_output:
        return None

    resp = requests.post(
        "http://172.17.0.3:1090/v1/chat/completions",
        json={
            "messages": [{
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    input=input, output=output, expected_output=expected_output
                ),
            }],
            "max_tokens": 12000,
            "temperature": 0.6,
        },
        timeout=1500,
    )
    resp.raise_for_status()
    
    response = resp.json()
    #print(response["choices"][0]["message"]["content"])
    time.sleep(5)

    try:
        verdict = json.loads(response["choices"][0]["message"]["content"])
        return Evaluation(
            name="correctness",
            value=float(verdict["score"]),
            comment=verdict.get("reasoning", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def avg_correctness(result):
    scores = [
        e.value
        for r in result.item_results
        for e in r.evaluations
        if e.name == "correctness" and isinstance(e.value, (int, float))
    ]
    return sum(scores) / len(scores) if scores else float("nan")

def run(label, task):
    dataset = get_or_create_dataset()
    return dataset.run_experiment(
        name=f"{label}-{model_name}",
        description=f"{label}: {model_name}",
        task=task,
        evaluators=[llm_judge_correctness],
    )

base_result = run("base", base_task)
ft_result = run("fine-tuned", finetuned_task)

base_score = avg_correctness(base_result)
ft_score = avg_correctness(ft_result)

print("\n=== Results ===")
print(f"Base       ({BASE_MODEL}): {base_score:.3f}")
print(f"Fine-tuned ({FINE_TUNED_MODEL}): {ft_score:.3f}")
print(f"Improvement: {ft_score - base_score:+.3f}")

langfuse.flush()
"""
Academic QA generation pipeline (single document, English output).

Reads data/academic/en/config/Ch*/0/0.json + corresponding doc TXT files,
generates QA triples (factual, multi-hop, summarization) using LocalModelClient
(Ministral-8B-Instruct-2410 loaded locally, no API key required).

Output written to output/academic/en/config/Ch*/0/0.json

Usage (from qar_generation/):
    python code/academic/en/qra_pipeline_single_doc.py
"""

import sys
import os
import glob
import json
import logging
import re
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                    format='%(asctime)s [%(levelname)s] %(message)s')

QAR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
MODEL_PATH = '/home/ensta/data/Ministral-8B-Instruct-2410'
INPUT_DIR = os.path.join(QAR_DIR, 'data', 'academic', 'en', 'config')
OUTPUT_DIR = os.path.join(QAR_DIR, 'output', 'academic', 'en', 'config')
PROMPT_FILE = os.path.join(QAR_DIR, 'prompts', 'academic_en.jsonl')
DOC_DIR = os.path.join(QAR_DIR, 'data', 'academic', 'en', 'doc')

# Excerpt length injected into prompts (chars)
EXCERPT_LEN = 8000


# ---------------------------------------------------------------------------
# Local model client
# ---------------------------------------------------------------------------

class LocalModelClient:
    def __init__(self, model_path: str):
        logging.info(f"[LocalModelClient] Loading tokenizer from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        logging.info("[LocalModelClient] Loading model (float16)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, local_files_only=True)
        self.model = self.model.cuda()
        used = torch.cuda.memory_allocated() / 1024**3
        logging.info(f"[LocalModelClient] Model on GPU -- Memory used: {used:.1f} GB")

    def generate(self, tasks: List[Dict], max_new_tokens: int = 2048) -> List[str]:
        responses = []
        for i, task in enumerate(tasks):
            print(f"[LocalModelClient] Generating {i+1}/{len(tasks)}...")
            messages = [
                {"role": "system", "content": task["system_prompt"]},
                {"role": "user", "content": task["user_prompt"]},
            ]
            inputs = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs.shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            responses.append(text)
            print(f"[LocalModelClient] Generation {i+1}/{len(tasks)} done.")
        return responses


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(response: str) -> list:
    """Extract JSON array from LLM response, handling markdown code blocks."""
    # Strip markdown fences
    text = re.sub(r'```(?:json)?\s*', '', response).strip()
    text = text.rstrip('`').strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to extract first JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    logging.warning(f"[postprocess] Failed to parse JSON (first 200 chars): {text[:200]}")
    return []


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def read_prompts(prompt_file: str) -> Dict[str, Dict]:
    prompts = {}
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            p = json.loads(line)
            prompts[p['prompt_type']] = p
    return prompts


def process_chapter(client: LocalModelClient, config_path: str, prompts: Dict) -> None:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Load document text
    doc_path = config_path.replace(
        os.path.join('data', 'academic', 'en', 'config'),
        os.path.join('data', 'academic', 'en', 'doc')
    ).replace('.json', '.txt')

    if not os.path.exists(doc_path):
        logging.error(f"Doc not found: {doc_path}")
        return

    with open(doc_path, 'r', encoding='utf-8') as f:
        doc_content = f.read()

    # Inject document excerpt into config for prompt formatting
    config['document_excerpt'] = doc_content[:EXCERPT_LEN]
    config['Generated Article'] = doc_content

    # Generate QA for each type
    qa_tasks = []
    qa_keys = []
    for prompt_type, qa_key in [
        ('Factual Question', 'qa_fact_based'),
        ('Multi-hop Reasoning Question', 'qa_multi_hop'),
        ('Summarization Question', 'qa_summary'),
    ]:
        if prompt_type not in prompts:
            logging.warning(f"Prompt type not found: {prompt_type}")
            continue
        p = prompts[prompt_type]
        qa_tasks.append({
            'system_prompt': p['system_prompt'],
            'user_prompt': p['user_prompt'].format(config=config),
        })
        qa_keys.append(qa_key)

    responses = client.generate(qa_tasks)

    for key, resp in zip(qa_keys, responses):
        parsed = parse_json_response(resp)
        config[key] = parsed
        logging.info(f"  {key}: {len(parsed)} items")

    # Write output config
    rel = os.path.relpath(config_path, INPUT_DIR)
    out_path = os.path.join(OUTPUT_DIR, rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    logging.info(f"Saved: {out_path}")


def main():
    if not torch.cuda.is_available():
        logging.warning("No GPU detected — this will be very slow.")

    client = LocalModelClient(MODEL_PATH)
    prompts = read_prompts(PROMPT_FILE)

    config_files = glob.glob(os.path.join(INPUT_DIR, '**', '*.json'), recursive=True)
    logging.info(f"Found {len(config_files)} chapter config(s).")

    for config_path in sorted(config_files):
        topic = config_path.split(os.sep)[-3]
        logging.info(f"\n=== Processing {topic} ===")
        try:
            process_chapter(client, config_path, prompts)
        except Exception as e:
            logging.error(f"Error processing {config_path}: {e}")


if __name__ == '__main__':
    main()

"""
Extracts verbatim French references for QA items that lack them.

Reads output/academic/en/config/Ch*/0/0.json, sends each QA item
(without a ref) to LocalModelClient 1-by-1 to avoid JSON truncation,
then saves the updated config in-place.

CRITICAL: References MUST be verbatim French sentences from the source doc.
The prompt enforces this with an explicit language rule.

Usage (from qar_generation/):
    python add_references_academic.py
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

QAR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = '/home/ensta/data/Ministral-8B-Instruct-2410'
CONFIG_DIR = os.path.join(QAR_DIR, 'output', 'academic', 'en', 'config')
PROMPT_FILE = os.path.join(QAR_DIR, 'prompts', 'academic_en.jsonl')

QA_KEYS = ['qa_fact_based', 'qa_multi_hop', 'qa_summary']


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

    def generate_one(self, system_prompt: str, user_prompt: str,
                     max_new_tokens: int = 2048) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(response: str) -> list:
    text = re.sub(r'```(?:json)?\s*', '', response).strip().rstrip('`').strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    logging.warning(f"[parse_json] Failed: {text[:200]}")
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        logging.warning("No GPU detected — this will be very slow.")

    # Load reference prompt
    ref_prompt = None
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            p = json.loads(line)
            if p['prompt_type'] == 'single document reference':
                ref_prompt = p
                break
    if ref_prompt is None:
        logging.error("'single document reference' prompt not found in prompt file.")
        sys.exit(1)

    client = LocalModelClient(MODEL_PATH)

    config_files = sorted(glob.glob(os.path.join(CONFIG_DIR, '**', '*.json'), recursive=True))
    logging.info(f"Found {len(config_files)} chapter config(s) in output dir.")

    total_added = 0
    total_missing = 0

    for config_path in config_files:
        topic = config_path.split(os.sep)[-3]
        logging.info(f"\n=== Processing {topic} ===")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Load French source document
        doc_content = config.get('Generated Article', '')
        if not doc_content:
            # Try loading from data dir
            data_doc = config_path.replace(
                os.path.join('output', 'academic', 'en', 'config'),
                os.path.join('data', 'academic', 'en', 'doc')
            ).replace('.json', '.txt')
            if os.path.exists(data_doc):
                with open(data_doc, 'r', encoding='utf-8') as f:
                    doc_content = f.read()
            else:
                logging.warning(f"No doc found for {topic}, skipping.")
                continue

        modified = False

        for key in QA_KEYS:
            items = config.get(key, [])
            if not isinstance(items, list):
                continue

            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                # Skip if already has a valid ref
                if item.get('ref') and isinstance(item['ref'], list) and len(item['ref']) > 0:
                    continue

                total_missing += 1
                logging.info(f"  [{key}][{idx}] Extracting ref for: {item.get('question', '')[:60]}...")

                # Build prompt without .format() to avoid crashes on { } in French doc
                doc_excerpt = doc_content[:12000]
                qa_pairs_str = json.dumps([item], ensure_ascii=False)
                user_prompt = (
                    "CRITICAL LANGUAGE RULE: The source document is written in FRENCH.\n"
                    "All references MUST be verbatim sentences copied DIRECTLY from the French document.\n"
                    "Do NOT translate. Do NOT paraphrase. Copy the exact French text character by character.\n\n"
                    f"Source document (French):\n{doc_excerpt}\n\n"
                    f"For each question-answer pair below, find the verbatim French sentence(s) from the document "
                    f"that support the answer. Copy them exactly.\n\n"
                    f"QA pairs:\n{qa_pairs_str}\n\n"
                    "Return a JSON array with the same items, each with a 'ref' field containing a list of verbatim French sentences:\n"
                    '[{"question type": "...", "question": "...", "answer": "...", "ref": ["verbatim French sentence 1", ...]}, ...]\n'
                    "Output only the JSON array, no other text."
                )

                response = client.generate_one(ref_prompt['system_prompt'], user_prompt)
                parsed = parse_json_response(response)

                if parsed and isinstance(parsed[0], dict) and parsed[0].get('ref'):
                    config[key][idx] = parsed[0]
                    total_added += 1
                    logging.info(f"    → {len(parsed[0]['ref'])} ref(s) added.")
                    modified = True
                else:
                    logging.warning(f"    → No ref extracted.")

        if modified:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            logging.info(f"  Saved updated config: {config_path}")

    logging.info(f"\nDone. Added refs to {total_added}/{total_missing} items.")


if __name__ == '__main__':
    main()

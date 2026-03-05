"""
Combines all chapter QA configs into a single eval_dataset.json.

Reads output/academic/en/config/Ch*/0/0.json and flattens
qa_fact_based / qa_multi_hop / qa_summary into one list.

Usage (from qar_generation/):
    python combine_dataset.py
"""

import json
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True,
                    format='%(asctime)s [%(levelname)s] %(message)s')

QAR_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(QAR_DIR, 'output', 'academic', 'en', 'config')
OUTPUT_FILE = os.path.join(QAR_DIR, 'output', 'academic', 'en', 'eval_dataset.json')

CHAPTERS = [
    'Ch1_Intro_ML',
    'Ch2_Decision_Trees',
    'Ch3_SVM',
    'Ch4_Neural_Networks',
    'Ch5_Unsupervised',
]

QA_KEYS = ['qa_fact_based', 'qa_multi_hop', 'qa_summary']

TYPE_MAP = {
    'qa_fact_based': 'qa_fact_based',
    'qa_multi_hop': 'qa_multi_hop',
    'qa_summary': 'qa_summary',
}


def main():
    all_questions = []

    for chapter in CHAPTERS:
        config_path = os.path.join(CONFIG_DIR, chapter, '0', '0.json')
        if not os.path.exists(config_path):
            logging.warning(f"Config not found: {config_path}")
            continue

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        chapter_count = 0
        for key in QA_KEYS:
            items = config.get(key, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                entry = {
                    'chapter': chapter,
                    'question_type': TYPE_MAP[key],
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'ref': item.get('ref', []),
                }
                all_questions.append(entry)
                chapter_count += 1

        logging.info(f"{chapter}: {chapter_count} questions")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, ensure_ascii=False, indent=2)

    refs_ok = sum(1 for q in all_questions if q.get('ref'))
    logging.info(f"\nTotal: {len(all_questions)} questions, {refs_ok}/{len(all_questions)} with refs")
    logging.info(f"Saved to: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()

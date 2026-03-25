"""
Prepares data for the 'academic' domain of the RAGEval pipeline.

Reads real_doc.txt (MI201 ENSTA Paris), cleans the raw pdftotext output,
splits into 5 chapters and writes config JSON + doc TXT files.

Cleaning fixes:
  - Accent artifacts from pdftotext: U+00B4 (acute) and U+0060 (grave) + letter
  - Cedilla artifacts: c + U+00B8 -> ç
  - Table of contents lines (. . . . . .)
  - Page headers (MI201-ENSTA Paris N)
  - Multiple blank lines

Usage (from qar_generation/):
    python prepare_academic_data.py
"""

import json
import os
import re
import sys

REAL_DOC = '/home/ensta/ensta-hidouri/Synthetic-Data-Generation-for-RAG-Systems/taxonomy/data/real_doc.txt'
QAR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(QAR_DIR, 'data', 'academic', 'en')

COURSE_CODE = 'MI201'
INSTITUTION = 'ENSTA Paris'
LANGUAGE = 'french'

CHAPTERS = [
    {
        'topic': 'Ch1_Intro_ML',
        'title': 'Introduction to Machine Learning',
        'pages': 'pages 7-32',
        'keywords': ['machine learning', 'k-NN', 'Bayesian', 'generalization',
                     'supervised learning', 'feature extraction', 'linear regression', 'bias-variance'],
    },
    {
        'topic': 'Ch2_Decision_Trees',
        'title': 'Decision Trees and Ensemble Methods',
        'pages': 'pages 33-50',
        'keywords': ['decision tree', 'random forest', 'boosting', 'bagging',
                     'Gini impurity', 'entropy', 'AdaBoost', 'gradient boosting'],
    },
    {
        'topic': 'Ch3_SVM',
        'title': 'Regularization and Support Vector Machines',
        'pages': 'pages 51-70',
        'keywords': ['SVM', 'kernel', 'regularization', 'L1', 'L2',
                     'support vectors', 'margin', 'hyperplane'],
    },
    {
        'topic': 'Ch4_Neural_Networks',
        'title': 'Introduction to Neural Networks',
        'pages': 'pages 71-88',
        'keywords': ['neural network', 'backpropagation', 'activation function',
                     'gradient descent', 'deep learning', 'perceptron'],
    },
    {
        'topic': 'Ch5_Unsupervised',
        'title': 'Unsupervised Learning',
        'pages': 'pages 89-104',
        'keywords': ['k-means', 'clustering', 'PCA', 'dimensionality reduction',
                     'DBSCAN', 'EM algorithm', 'Gaussian mixture'],
    },
]


def clean_text(text: str) -> str:
    """
    Fix pdftotext encoding artifacts and remove structural noise.

    pdftotext extracts accented characters as two separate Unicode codepoints:
      U+00B4 (ACUTE ACCENT)  + letter  ->  letter with acute (é, á, ó, ú, í)
      U+0060 (GRAVE ACCENT)  + letter  ->  letter with grave (è, à, ù)
      U+02C6 (CIRCUMFLEX)    + letter  ->  letter with circumflex (ê, â, î, ô, û)
      c + U+00B8 (CEDILLA)            ->  ç
    """
    # --- 1. Fix acute accent artifacts (U+00B4) ---
    # Remove spurious space before acute
    text = text.replace(' \u00B4', '\u00B4')
    acute = '\u00B4'
    for base, accented in [('e', 'é'), ('E', 'É'), ('a', 'á'), ('A', 'Á'),
                            ('o', 'ó'), ('O', 'Ó'), ('u', 'ú'), ('U', 'Ú'),
                            ('i', 'í'), ('I', 'Í')]:
        text = text.replace(acute + base, accented)

    # --- 2. Fix grave accent artifacts (U+0060) ---
    grave = '\u0060'
    for base, accented in [('e', 'è'), ('E', 'È'), ('a', 'à'), ('A', 'À'),
                            ('u', 'ù'), ('U', 'Ù')]:
        text = text.replace(grave + base, accented)

    # --- 3. Fix circumflex artifacts (U+02C6) ---
    circ = '\u02C6'
    for base, accented in [('e', 'ê'), ('E', 'Ê'), ('a', 'â'), ('A', 'Â'),
                            ('i', 'î'), ('I', 'Î'), ('o', 'ô'), ('O', 'Ô'),
                            ('u', 'û'), ('U', 'Û')]:
        text = text.replace(circ + base, accented)
    # Also handle ^ as circumflex marker
    text = re.sub(r'\^e', 'ê', text)
    text = re.sub(r'\^a', 'â', text)
    text = re.sub(r'\^i', 'î', text)
    text = re.sub(r'\^o', 'ô', text)
    text = re.sub(r'\^u', 'û', text)

    # --- 4. Fix cedilla artifacts ---
    text = text.replace('c\u00B8', 'ç').replace('C\u00B8', 'Ç')
    text = text.replace('c\u0327', 'ç').replace('C\u0327', 'Ç')

    # --- 5. Fix diaeresis (tréma) ---
    diaeresis = '\u00A8'
    for base, accented in [('e', 'ë'), ('E', 'Ë'), ('i', 'ï'), ('I', 'Ï'),
                            ('u', 'ü'), ('U', 'Ü'), ('a', 'ä'), ('A', 'Ä')]:
        text = text.replace(diaeresis + base, accented)

    # --- 6. Remove TOC lines (". . . . . . N") ---
    text = re.sub(r'[^\n]*\.{4,}[^\n]*\n', '', text)

    # --- 7. Remove page headers ("MI201-ENSTA Paris 42") ---
    text = re.sub(r'MI201-ENSTA Paris \d+[A-Z]?[^\n]*\n?', '', text)

    # --- 8. Remove ALL-CAPS structural headers (e.g. "CONTENTS CONTENTS") ---
    text = re.sub(r'^[A-Z ]{6,}\n', '', text, flags=re.MULTILINE)

    # --- 9. Collapse multiple blank lines ---
    text = re.sub(r'\n{3,}', '\n\n', text)

    # --- 10. Skip to body (after TOC) ---
    body_marker = 'Apprentissage automatique: introduction\n1.1'
    body_start = text.find(body_marker)
    if body_start == -1:
        print(f"  WARNING: 'Chapter 1' marker not found, using offset 0")
        return text.strip()
    print(f"  Body starts at char {body_start} ('Chapter 1' marker found)")
    return text[body_start:].strip()


def split_document(body: str, n: int = 5) -> list:
    """
    Split cleaned body text into 5 chapter texts.
    The body starts at "Chapter 1\n" which is the first chapter heading.
    Each chapter is split at equal character intervals aligned to a newline.
    """
    total = len(body)
    chunk_size = total // n
    chapters = []
    start = 0
    for i in range(n):
        if i == n - 1:
            chapters.append(body[start:])
        else:
            end = (i + 1) * chunk_size
            # Align to next newline
            newline_pos = body.find('\n', end)
            if newline_pos == -1:
                newline_pos = end
            chapters.append(body[start:newline_pos])
            start = newline_pos + 1
    return chapters


def main():
    print('=' * 60)
    print('RAGEval data preparation -- Academic domain (MI201)')
    print('=' * 60)

    if not os.path.exists(REAL_DOC):
        print(f'ERROR: file not found: {REAL_DOC}')
        sys.exit(1)

    with open(REAL_DOC, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()
    print(f'Raw document: {len(raw)} chars')

    body = clean_text(raw)
    print(f'After cleaning: {len(body)} chars')

    # Remove old Chapitre_0 directory if it exists
    old_dir = os.path.join(DATA_DIR, 'config', 'Chapitre_0')
    if os.path.exists(old_dir):
        import shutil
        shutil.rmtree(old_dir)
        print('Removed old Chapitre_0 directory.')

    chapter_texts = split_document(body, n=len(CHAPTERS))
    print(f'Split into {len(chapter_texts)} chapters.\n')

    for i, (ch, text) in enumerate(zip(CHAPTERS, chapter_texts)):
        topic = ch['topic']
        title = ch['title']

        # Extract a content summary (first 500 chars)
        content_summary = text[:500].replace('\n', ' ').strip()

        # Config JSON
        config = {
            'course_code': COURSE_CODE,
            'course_title': title,
            'institution': INSTITUTION,
            'instructors': ['Gianni Franchi', 'Stephane Herbin', 'Adrien Chan Hon Tong'],
            'language': LANGUAGE,
            'chapter': f'Chapter {i+1}: {title}',
            'topic': topic,
            'keywords': ch['keywords'],
            'pages': ch['pages'],
            'content_summary': content_summary,
        }

        # Write config
        config_dir = os.path.join(DATA_DIR, 'config', topic, '0')
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, '0.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

        # Write doc
        doc_dir = os.path.join(DATA_DIR, 'doc', topic, '0')
        os.makedirs(doc_dir, exist_ok=True)
        doc_path = os.path.join(doc_dir, '0.txt')
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f'  Chapter {i+1}: {topic} ({len(text)} chars)')
        print(f'    Preview: {content_summary[:80]}...')

    print()
    print('Data ready. Run order:')
    print('  1. sbatch scripts/run_academic_single_doc.sbatch')
    print('  2. sbatch scripts/run_add_references.sbatch')
    print('  3. python combine_dataset.py')
    print('  4. sbatch scripts/run_evaluate_rag.sbatch')


if __name__ == '__main__':
    main()

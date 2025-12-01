# Data Directory

This directory contains the PDF textbooks used for testing and generation.

## Required File

Place your PDF textbook here:
- **File name:** `sample_textbook.pdf`
- **Location:** `generation/multi-agent/data/sample_textbook.pdf`

## Recommended PDF Sources

For testing, use an academic textbook in PDF format. Good options:

### Free Academic Textbooks:
1. **OpenStax** (https://openstax.org/)
   - Free, peer-reviewed textbooks
   - Subjects: Physics, Biology, Chemistry, Math, etc.
   - Download PDF directly

2. **arXiv Papers** (https://arxiv.org/)
   - Research papers in PDF
   - Well-structured, academic content
   - Good for testing

3. **Project Gutenberg** (https://www.gutenberg.org/)
   - Classic textbooks in public domain
   - Available in PDF format

### What Makes a Good Test PDF:
- Clear table of contents
- Well-structured chapters and sections
- Contains definitions and equations
- 50-200 pages (not too large for testing)
- Academic or technical content

## Current PDF Status

After adding your PDF, verify it:
```bash
cd generation/multi-agent
python scripts/test_pdf_processor.py
```

This will test if the PDF can be processed correctly.

## Note

The PDF file itself is NOT tracked in git (see .gitignore).
Each team member needs to add their own PDF for local testing.

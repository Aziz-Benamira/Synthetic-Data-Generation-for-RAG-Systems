# Data Directory

This directory contains all data used in the project.

## Structure

- **raw/** - Original, unprocessed data (PDFs, documents, etc.)
- **processed/** - Cleaned and processed data ready for use
- **datasets/** - Generated synthetic datasets ready for publication

## Notes

- Raw data files are not tracked in git (see .gitignore)
- Add a README in each subdirectory describing the specific data
- Use consistent naming conventions: `{source}_{type}_{date}.{ext}`

## Example Structure

```
data/
├── raw/
│   ├── textbooks/
│   │   ├── physics_101.pdf
│   │   └── chemistry_basics.pdf
│   └── industrial/
│       └── edf_documents/
├── processed/
│   ├── physics_101_chunks.json
│   └── chemistry_basics_chunks.json
└── datasets/
    ├── physics_qa_v1.jsonl
    └── chemistry_qa_v1.jsonl
```

# Scripts

Utility scripts for data processing, setup, and automation.

## Scripts

- `setup_environment.py` - Initial environment setup
- `download_data.py` - Download and prepare datasets
- `run_pipeline.py` - Run the full generation pipeline
- `evaluate_dataset.py` - Evaluate generated datasets
- `export_to_huggingface.py` - Export datasets to HuggingFace

## Usage

Each script includes help documentation:
```bash
python script_name.py --help
```

## Example

```bash
# Run the full pipeline
python scripts/run_pipeline.py --input data/raw/textbook.pdf --output data/datasets/output.jsonl --num-questions 100

# Evaluate a dataset
python scripts/evaluate_dataset.py --dataset data/datasets/output.jsonl --metrics faithfulness,relevancy
```

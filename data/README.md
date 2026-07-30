# Data

This folder is **gitignored** — no data files are ever committed.

- `raw/` — datasets as downloaded (created automatically at runtime).
- `processed/` — cleaned/derived data produced by the pipeline.

To populate the Step 1 dataset, run from the project root:

```bash
python -m src.data.load_diabetes --config configs/data.yaml
```

This downloads the UCI Diabetes 130-US Hospitals dataset and writes
`processed/diabetes.parquet`.

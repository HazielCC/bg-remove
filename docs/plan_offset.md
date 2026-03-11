# Plan: Add Offset Support to Dataset Download

## Objective
The user wants to download chunks of a HuggingFace dataset (e.g. from position 10 to 20, 20 to 30) instead of just the first `N` samples. We need to add an `offset` (or `start_index`) parameter to the dataset download logic.

## Scope
Modify `backend/routers/datasets.py` and `backend/ml/dataset.py` to support an `offset` parameter during dataset download.

## Tasks

### 1. Update `backend/routers/datasets.py`
- Update `DownloadRequest` schema to include an `offset: int = 0` field.
- Pass `offset=req.offset` to `HFMattingDataset.prepare_from_hf`.

### 2. Update `backend/ml/dataset.py`
- In `HFMattingDataset.prepare_from_hf`, add `offset: int = 0` to the signature.
- Pass `offset` down to `_download_via_datasets` and `_download_via_snapshot`.
- In `_download_via_datasets`:
  - If `offset > 0` and `max_samples` is provided, slice the dataset: `ds.select(range(offset, min(offset + max_samples, len(ds))))`
  - If `offset > 0` but no `max_samples`, slice to the end: `ds.select(range(offset, len(ds)))`
- In `_download_via_snapshot`:
  - Instead of `src_image_files[:limit]`, use `src_image_files[offset : offset + limit]` (or just `src_image_files[offset:]` if no limit).
  - Make sure the saved filenames still make sense (e.g. they should ideally reflect the real index, so instead of `i:06d` where `i` is the loop counter, we should probably use `(i + offset):06d` or similar, or just let it save starting from 0 to N-1 for that chunk). Let's save them as `(offset + i):06d.jpg` so chunks don't overwrite each other if downloaded to the same dir.

### 3. Verify
- Compile python files to ensure no syntax errors.
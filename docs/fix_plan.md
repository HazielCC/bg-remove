# Plan: Fix NameError in routers/datasets.py

## Objective
Fix the `NameError: name 'Field' is not defined` caused by the missing import of `Field` from `pydantic` in `backend/routers/datasets.py`.

## Scope
File: `backend/routers/datasets.py`

## Task
1. Search for `from pydantic import BaseModel` in `backend/routers/datasets.py`.
2. Replace it with `from pydantic import BaseModel, Field`.
3. Verify compilation.
# Plan: Fix Concurrency and Optimizer in ML Pipeline

## Objective
Fix a critical thread-safety issue in the Server-Sent Events (SSE) metric streaming from the ML training thread back to the FastAPI event loop. Update the optimizer to a modern standard (`AdamW`) for faster convergence in MODNet fine-tuning.

## Scope of Changes
All changes will be isolated to the backend ML pipeline, specifically `backend/ml/trainer.py` and a minor adjustment in `backend/routers/training.py`.

## Tasks

### 1. Fix Thread-Safety for SSE (Metric Streaming)
The training runs in a separate background thread via `asyncio.to_thread()`, but it attempts to put events directly into an `asyncio.Queue` using `put_nowait()`. `asyncio.Queue` is not thread-safe.
*   **Action:** Modify `_send_event` in `backend/ml/trainer.py` to accept the main thread's event loop as an argument or capture it globally.
*   **Action:** Use `loop.call_soon_threadsafe(queue.put_nowait, event)` to safely cross the thread boundary and deliver the metrics to the FastAPI event loop.
*   **Action:** Update `run_supervised_training` and `run_soc_adaptation` to accept the `loop` parameter from the router.
*   **Action:** Update `backend/routers/training.py` to pass `asyncio.get_running_loop()` when invoking the training functions.

### 2. Upgrade Optimizer to AdamW
The current implementation uses standard SGD with momentum. For fine-tuning deep convolutional networks like MODNet, AdamW provides significantly faster convergence and better generalization.
*   **Action:** In `backend/ml/trainer.py` (`run_supervised_training`), replace `torch.optim.SGD` with `torch.optim.AdamW`.
*   **Action:** Adjust the default learning rate in the configuration (AdamW typically requires a smaller learning rate than SGD, e.g., `0.001` or `0.0005`). We will modify the default `lr` in `TrainingConfig` to a more sensible value for AdamW (`0.001`).
*   **Action:** Update the learning rate scheduler to `CosineAnnealingLR` or a StepLR configured for AdamW dynamics, which is more robust for fine-tuning.

### 3. Verification
*   **Action:** Ensure the FastAPI application starts without errors.
*   **Action:** Verify that the `TrainingConfig` dataclass and router schemas are correctly aligned with the new optimizer defaults.
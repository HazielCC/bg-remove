# Plan: Fix SSE Broadcasting and Address Benchmark Questions

## Objective
1. Fix the real-time SSE issue where epochs/progress don't update automatically without a refresh.
2. Explain to the user why the Benchmark section allows manual configuration.

## Scope
File: `backend/routers/training.py`
File: `backend/ml/trainer.py`

## Task 1: Fix SSE Broadcasting
The current implementation uses a single `_event_queue: asyncio.Queue | None = None`. This means if a user has two tabs open (e.g., Train and Monitor), or if a previous connection hasn't been garbage collected by FastAPI, one connection will "steal" the events from the queue, leaving the other connection (the active tab) starved of updates. 
We must implement a 1-to-many broadcast system.

### Steps
1. In `backend/ml/trainer.py`:
   - Change `training_event_queue: asyncio.Queue | None = None` to `training_event_queues: list[asyncio.Queue] = []`.
   - Update `_send_event` to iterate over `training_event_queues` and call `_main_loop.call_soon_threadsafe(q.put_nowait, event)` for each queue.
   - Update `run_supervised_training` and `run_soc_adaptation` signatures from `event_queue` to `event_queues`.

2. In `backend/routers/training.py`:
   - Change `_event_queue` to `_event_queues: list[asyncio.Queue] = []`.
   - In `start_training`, clear `_event_queues` when a new training starts (or just rely on the existing queues). Pass `_event_queues` to the training thread.
   - In `stream_metrics`, when a new request comes in:
     - Create a local `q = asyncio.Queue(maxsize=200)`.
     - Append `q` to `_event_queues`.
     - Read from `q` in the `while True` loop.
     - Add a `finally:` block to remove `q` from `_event_queues` when the client disconnects.
     
## Task 2: Compile & Verify
Run `uv run python -m py_compile` to ensure correctness.
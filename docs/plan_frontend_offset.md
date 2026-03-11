# Plan: Add Offset Field to Frontend Dataset UI

## Objective
The user needs to be able to input an `offset` (starting position) when downloading a dataset from HuggingFace to the local backend.

## Scope
File: `app/fine-tune/datasets/page.tsx`

## Analysis
1.  **State Management:** Currently, the file uses `useState<number>(500)` for `downloadMax`. We need to add a new state `const [downloadOffset, setDownloadOffset] = useState<number>(0);`.
2.  **API Call:** The `handleDownload` function sends a payload to `/datasets/download` with `{ dataset_name, split, max_samples }`. We need to add `offset: downloadOffset` to this payload.
3.  **UI Elements:** In the "Descargar desde HuggingFace" section (around line 440), there's an input for `downloadMax`. We need to add a neighboring input field for the `offset`.

## Tasks

### 1. Update State
- Search for `useState<number>(500)` and add the offset state right below it.

### 2. Update API Call
- Search for `max_samples: downloadMax || null,` and append `offset: downloadOffset,`.

### 3. Update UI
- Search for the `<input type="number" ... value={downloadMax} />` block.
- Add a new `<div>` containing a `<label>` ("Empezar desde (offset):") and an `<input type="number" min={0} value={downloadOffset} onChange={(e) => setDownloadOffset(Number(e.target.value))} />`.

## Verification
Ensure the UI renders correctly (using Next.js build or TypeScript check) and the payload matches the backend schema.
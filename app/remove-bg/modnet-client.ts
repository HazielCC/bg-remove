import { pipeline } from '@huggingface/transformers';

export type Variant = 'auto' | 'fp32' | 'fp16' | 'uint8';
type ModelDType = 'fp32' | 'fp16' | 'uint8' | 'q8';

interface CacheItem {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  segmenter: any; 
  dtype: string;
  modelId: string;
  loadTime: string;
  cached?: boolean; // added this as it's used in page.tsx
}

const CACHE: Partial<Record<string, CacheItem>> = {};

async function detectModelDtype(variant: Variant): Promise<ModelDType> {
  if (variant === 'fp32') return 'fp32';
  if (variant === 'fp16') return 'fp16';
  if (variant === 'uint8') return 'uint8';

  // auto: q8 maps to *_quantized.onnx in transformers.js v3.
  return 'q8';
}

function buildDtypeCandidates(dtype: ModelDType): ModelDType[] {
  if (dtype === 'q8') return ['q8', 'uint8'];
  if (dtype === 'uint8') return ['uint8', 'q8'];
  return [dtype];
}

export async function createSegmenter(options?: { variant?: Variant; modelPath?: string }) {
  const variant = options?.variant ?? 'auto';
  const requestedDtype = await detectModelDtype(variant);
  const dtypeCandidates = buildDtypeCandidates(requestedDtype);
  
  const modelCandidates = options?.modelPath 
    ? [options.modelPath] 
    : ['/models/modnet', 'Xenova/modnet'];

  for (const modelId of modelCandidates) {
    for (const dtype of dtypeCandidates) {
      const cacheKey = `${modelId}:${dtype}`;
      if (CACHE[cacheKey]) {
        console.log(`[MODNet] ♻️ Using cached model | id=${modelId} | dtype=${dtype}`);
        return { ...CACHE[cacheKey], cached: true };
      }
    }
  }

  let lastError: unknown;
  for (const modelId of modelCandidates) {
    for (const dtype of dtypeCandidates) {
      try {
        console.log(`[MODNet] ⬇️ Loading model | id=${modelId} | dtype=${dtype}`);
        const t0 = performance.now();
        const segmenter = await pipeline('background-removal', modelId, { dtype });
        const loadTime = ((performance.now() - t0) / 1000).toFixed(1);
        console.log(`[MODNet] ✅ Model loaded in ${loadTime}s | id=${modelId} | dtype=${dtype}`);

        const cacheKey = `${modelId}:${dtype}`;
        CACHE[cacheKey] = { segmenter, dtype, modelId, loadTime };
        return { ...CACHE[cacheKey], cached: false };
      } catch (error) {
        lastError = error;
        console.warn(`[MODNet] Failed load | id=${modelId} | dtype=${dtype}`, error);
      }
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error('Failed to load MODNet pipeline with all dtype/model fallbacks');
}


import { pipeline } from '@huggingface/transformers';

type Variant = 'auto' | 'fp32' | 'fp16' | 'uint8';

const CACHE: Partial<Record<string, any>> = {};

async function detectModelDtype(variant: Variant): Promise<'fp32' | 'fp16' | 'uint8'> {
  if (variant === 'fp32') return 'fp32';
  if (variant === 'fp16') return 'fp16';
  if (variant === 'uint8') return 'uint8';

  // auto: default to uint8 for faster loading and compatibility with transformers.js
  return 'uint8';
}

export async function createSegmenter(options?: { variant?: Variant }) {
  const variant = options?.variant ?? 'auto';
  const dtype = await detectModelDtype(variant);
  const modelId = 'Xenova/modnet';

  const cacheKey = `${modelId}:${dtype}`;
  if (CACHE[cacheKey]) {
    console.log(`[MODNet] ♻️ Using cached model | id=${modelId} | dtype=${dtype}`);
    return { ...CACHE[cacheKey], cached: true };
  }

  console.log(`[MODNet] ⬇️ Loading model | id=${modelId} | dtype=${dtype}`);
  const t0 = performance.now();
  const segmenter = await pipeline('background-removal', modelId, { dtype });
  const loadTime = ((performance.now() - t0) / 1000).toFixed(1);
  console.log(`[MODNet] ✅ Model loaded in ${loadTime}s | id=${modelId} | dtype=${dtype}`);

  CACHE[cacheKey] = { segmenter, dtype, modelId, loadTime };
  return { ...CACHE[cacheKey], cached: false };
}


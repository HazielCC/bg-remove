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

function clearMemoryCacheForModel(modelPath: string) {
  for (const cacheKey of Object.keys(CACHE)) {
    if (cacheKey.startsWith(`${modelPath}:`)) {
      delete CACHE[cacheKey];
    }
  }
}

async function clearBrowserCacheForModel(modelPath: string) {
  if (typeof window === 'undefined' || typeof caches === 'undefined') {
    return;
  }

  try {
    const cache = await caches.open('transformers-cache');
    const keys = await cache.keys();
    const targetPath = new URL(modelPath, window.location.origin).pathname.replace(/\/$/, '');

    await Promise.all(
      keys.map(async (request) => {
        const requestPath = new URL(request.url).pathname.replace(/\/$/, '');
        if (requestPath === targetPath || requestPath.startsWith(`${targetPath}/`)) {
          await cache.delete(request);
        }
      }),
    );
  } catch (error) {
    console.warn(`[MODNet] Unable to clear browser cache for ${modelPath}`, error);
  }
}

async function detectModelDtype(variant: Variant): Promise<ModelDType> {
  if (variant === 'fp32') return 'fp32';
  if (variant === 'fp16') return 'fp16';
  if (variant === 'uint8') return 'uint8';

  // auto: q8 maps to *_quantized.onnx in transformers.js v3.
  return 'q8';
}

function buildDtypeCandidates(dtype: ModelDType): ModelDType[] {
  if (dtype === 'q8') return ['q8', 'uint8', 'fp16', 'fp32'];
  if (dtype === 'uint8') return ['uint8', 'q8', 'fp16', 'fp32'];
  if (dtype === 'fp16') return ['fp16', 'fp32', 'q8', 'uint8'];
  if (dtype === 'fp32') return ['fp32', 'fp16', 'q8', 'uint8'];
  return [dtype];
}

export async function createSegmenter(options?: {
  variant?: Variant;
  modelPath?: string;
  reload?: boolean;
}) {
  const variant = options?.variant ?? 'auto';
  const requestedDtype = await detectModelDtype(variant);
  const dtypeCandidates = buildDtypeCandidates(requestedDtype);
  
  const modelCandidates = options?.modelPath 
    ? [options.modelPath] 
    : ['/models/modnet', 'Xenova/modnet'];

  if (options?.reload && options.modelPath) {
    console.log(`[MODNet] Reload requested for ${options.modelPath}`);
    clearMemoryCacheForModel(options.modelPath);
    await clearBrowserCacheForModel(options.modelPath);
  }

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

import { pipeline } from '@huggingface/transformers';

type Variant = 'auto' | 'fp32' | 'fp16' | 'uint8';

const CACHE: Partial<Record<string, any>> = {};

async function fileExists(url: string) {
  try {
    const res = await fetch(url, { method: 'HEAD' });
    return res && res.ok;
  } catch (e) {
    return false;
  }
}

async function detectModelDtype(variant: Variant): Promise<'fp32' | 'fp16' | 'uint8'> {
  if (variant === 'fp32') return 'fp32';
  if (variant === 'fp16') return 'fp16';
  if (variant === 'uint8') return 'uint8';

  // auto-detect
  // prefer fp16 on WebGPU if weights exist
  const webgpuAvailable = typeof navigator !== 'undefined' && 'gpu' in navigator;
  const basePath = '/models/modnet/onnx';
  const hasFp16 = await fileExists(`${basePath}/model_fp16.onnx`);
  const hasQuant = await fileExists(`${basePath}/model_quantized.onnx`) || await fileExists(`${basePath}/model_uint8.onnx`);

  if (webgpuAvailable && hasFp16) return 'fp16';
  if (hasQuant) return 'uint8';
  return 'fp32';
}

export async function createSegmenter(options?: { variant?: Variant }) {
  const variant = options?.variant ?? 'auto';
  const dtype = await detectModelDtype(variant);
  const modelPath = '/models/modnet';

  const cacheKey = `${modelPath}:${dtype}`;
  if (CACHE[cacheKey]) return CACHE[cacheKey];

  const segmenter = await pipeline('background-removal', modelPath, { dtype });
  CACHE[cacheKey] = { segmenter, dtype };
  return CACHE[cacheKey];
}


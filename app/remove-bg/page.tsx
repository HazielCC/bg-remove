"use client";

import { RawImage } from '@huggingface/transformers';
import Link from 'next/link';
import { useCallback, useRef, useState } from 'react';
import { createSegmenter } from './modnet-client';

export default function RemoveBgPage() {
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [compositeUrl, setCompositeUrl] = useState<string | null>(null);
  const [variant, setVariant] = useState<'auto' | 'fp32' | 'fp16' | 'uint8'>('auto');
  const [modelInfo, setModelInfo] = useState<{ modelId: string; dtype: string; loadTime: string; cached: boolean } | null>(null);
  const [inferenceTime, setInferenceTime] = useState<string | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((f: File) => {
    const url = URL.createObjectURL(f);
    setFileUrl(url);
    setFile(f);
    setMaskUrl(null);
    setCompositeUrl(null);
    setModelInfo(null);
    setInferenceTime(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith('image/')) handleFile(f);
  }, [handleFile]);

  async function onRun() {
    if (!file) return;
    setLoading(true);
    setLoadingStep('Cargando modelo...');
    try {
      const result_seg = await createSegmenter({ variant });
      const { segmenter, dtype, modelId, loadTime, cached } = result_seg;
      setModelInfo({ modelId, dtype, loadTime: loadTime ?? '—', cached });

      setLoadingStep('Procesando imagen...');
      const raw = await RawImage.fromBlob(file);
      const t0 = performance.now();
      const output = await segmenter(raw);
      const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
      setInferenceTime(elapsed);
      console.log(`[MODNet] 🖼️ Inference done in ${elapsed}s`);

      const result = output[0];
      if (result.toBlob) {
        const blob = await result.toBlob();
        const url = URL.createObjectURL(blob);
        setMaskUrl(url);

        // Create composite (original with transparent background)
        setLoadingStep('Generando resultado...');
        const composite = await createComposite(file, blob);
        setCompositeUrl(composite);
      } else {
        await result.save('mask.png');
      }
    } catch (err) {
      console.error(err);
      alert('Error ejecutando MODNet. Revisa la consola.');
    } finally {
      setLoading(false);
      setLoadingStep('');
    }
  }

  async function createComposite(original: File, maskBlob: Blob): Promise<string> {
    const [imgBmp, maskBmp] = await Promise.all([
      createImageBitmap(original),
      createImageBitmap(maskBlob),
    ]);
    const canvas = document.createElement('canvas');
    canvas.width = imgBmp.width;
    canvas.height = imgBmp.height;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(imgBmp, 0, 0);
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = imgBmp.width;
    maskCanvas.height = imgBmp.height;
    const mCtx = maskCanvas.getContext('2d')!;
    mCtx.drawImage(maskBmp, 0, 0, imgBmp.width, imgBmp.height);
    const maskData = mCtx.getImageData(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < imgData.data.length; i += 4) {
      imgData.data[i + 3] = maskData.data[i]; // use R channel of mask as alpha
    }
    ctx.putImageData(imgData, 0, 0);
    return new Promise((resolve) => canvas.toBlob((b) => resolve(URL.createObjectURL(b!)), 'image/png'));
  }

  function reset() {
    setFile(null);
    setFileUrl(null);
    setMaskUrl(null);
    setCompositeUrl(null);
    setModelInfo(null);
    setInferenceTime(null);
    if (inputRef.current) inputRef.current.value = '';
  }

  return (
    <main className="min-h-screen bg-neutral-950 text-white">
      {/* Header */}
      <header className="border-b border-neutral-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/" className="text-neutral-500 hover:text-white transition-colors">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
          </Link>
          <div>
            <h1 className="text-lg font-semibold">Remover Fondo</h1>
            <p className="text-xs text-neutral-500">MODNet · Inferencia en navegador</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <label className="text-xs text-neutral-500">Variante:</label>
          <select
            className="bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            value={variant}
            onChange={(e) => setVariant(e.target.value as any)}
          >
            <option value="auto">Auto (cuantizado)</option>
            <option value="fp32">FP32 (completo)</option>
            <option value="fp16">FP16 (medio)</option>
            <option value="uint8">UINT8 (cuantizado)</option>
          </select>
        </div>
      </header>

      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Upload zone */}
        {!fileUrl ? (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current?.click()}
            className={`
              border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer transition-all
              ${dragActive
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-neutral-700 hover:border-neutral-500 hover:bg-neutral-900/50'}
            `}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFile(f);
              }}
            />
            <div className="text-5xl mb-4">📸</div>
            <p className="text-lg font-medium text-neutral-300">
              Arrastra una imagen aquí
            </p>
            <p className="text-sm text-neutral-500 mt-2">
              o haz clic para seleccionar un archivo
            </p>
            <p className="text-xs text-neutral-600 mt-4">
              PNG, JPG, WebP · Retratos recomendados
            </p>
          </div>
        ) : (
          <>
            {/* Action bar */}
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <button
                  onClick={reset}
                  className="text-sm text-neutral-400 hover:text-white transition-colors flex items-center gap-1.5"
                >
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                  Nueva imagen
                </button>
              </div>
              <div className="flex items-center gap-3">
                {compositeUrl && (
                  <a
                    href={compositeUrl}
                    download="sin-fondo.png"
                    className="text-sm bg-neutral-800 hover:bg-neutral-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    Sin fondo
                  </a>
                )}
                {maskUrl && (
                  <a
                    href={maskUrl}
                    download="mask.png"
                    className="text-sm bg-neutral-800 hover:bg-neutral-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
                  >
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    Máscara
                  </a>
                )}
                <button
                  onClick={onRun}
                  disabled={loading}
                  className={`
                    text-sm px-5 py-2 rounded-lg font-medium transition-all flex items-center gap-2
                    ${loading
                      ? 'bg-blue-600/50 text-blue-300 cursor-wait'
                      : 'bg-blue-600 hover:bg-blue-500 text-white'}
                  `}
                >
                  {loading ? (
                    <>
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      {loadingStep}
                    </>
                  ) : maskUrl ? (
                    <>
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                      </svg>
                      Re-procesar
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                        <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Remover fondo
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Image comparison */}
            <div className={`grid gap-4 ${compositeUrl ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1 max-w-lg mx-auto'}`}>
              {/* Original */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-neutral-500 uppercase tracking-wider">Original</span>
                </div>
                <div className="rounded-xl overflow-hidden border border-neutral-800 bg-neutral-900">
                  <img
                    src={fileUrl}
                    alt="original"
                    className="w-full h-auto"
                  />
                </div>
              </div>

              {/* Result */}
              {compositeUrl && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-neutral-500 uppercase tracking-wider">Sin fondo</span>
                  </div>
                  <div
                    className="rounded-xl overflow-hidden border border-neutral-800"
                    style={{
                      backgroundImage: 'url("data:image/svg+xml,%3Csvg width=\'20\' height=\'20\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Crect width=\'10\' height=\'10\' fill=\'%23222\'/%3E%3Crect x=\'10\' y=\'10\' width=\'10\' height=\'10\' fill=\'%23222\'/%3E%3Crect x=\'10\' width=\'10\' height=\'10\' fill=\'%23333\'/%3E%3Crect y=\'10\' width=\'10\' height=\'10\' fill=\'%23333\'/%3E%3C/svg%3E")',
                      backgroundSize: '20px 20px',
                    }}
                  >
                    <img
                      src={compositeUrl}
                      alt="sin fondo"
                      className="w-full h-auto"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Model info bar */}
            {modelInfo && (
              <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2 text-xs text-neutral-500 border-t border-neutral-800 pt-4">
                <span>
                  <span className="text-neutral-600">Modelo</span>{' '}
                  <span className="text-neutral-300 font-mono">{modelInfo.modelId}</span>
                </span>
                <span>
                  <span className="text-neutral-600">Tipo de dato</span>{' '}
                  <span className="text-neutral-300 font-mono">{modelInfo.dtype}</span>
                </span>
                <span>
                  <span className="text-neutral-600">Carga</span>{' '}
                  <span className="text-neutral-300">{modelInfo.cached ? '♻️ cache' : `⬇️ ${modelInfo.loadTime}s`}</span>
                </span>
                {inferenceTime && (
                  <span>
                    <span className="text-neutral-600">Inferencia</span>{' '}
                    <span className="text-neutral-300 font-mono">{inferenceTime}s</span>
                  </span>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </main>
  );
}

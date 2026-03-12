"use client";

import Image from 'next/image';
import Link from 'next/link';
import { useCallback, useEffect, useRef, useState } from 'react';

export default function LayeredDecompositionPage() {
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [downloadProgress, setDownloadProgress] = useState<number | null>(null);
  const [layers, setLayers] = useState<string[]>([]);
  
  // Model parameters
  const [prompt, setPrompt] = useState('A detailed image decomposed into semantic layers');
  const [layerNum, setLayerNum] = useState(4);
  const [steps, setSteps] = useState(50);
  const [seed, setSeed] = useState<number | ''>('');

  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Poll for download status
  useEffect(() => {
    let timer: NodeJS.Timeout;
    let active = true;

    async function poll() {
      if (!loading || !active) return;
      try {
        const res = await fetch('/api/layered/status');
        if (res.ok && active) {
          const status = await res.json();
          if (status.is_downloading) {
            setLoadingStep(status.message);
            setDownloadProgress(status.progress);
          } else {
            setDownloadProgress(null);
            setLoadingStep('Ejecutando Qwen-Image-Layered (esto puede tardar)...');
          }
        }
      } catch (e) {
        console.error("Status poll failed", e);
      }
      if (active && loading) {
        timer = setTimeout(poll, 1000);
      }
    }

    if (loading) {
      poll();
    }

    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [loading]);

  // Cleanup URLs on unmount
  useEffect(() => {
    return () => {
      if (fileUrl) URL.revokeObjectURL(fileUrl);
    };
  }, [fileUrl]);

  const handleFile = useCallback((f: File) => {
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    const url = URL.createObjectURL(f);
    setFileUrl(url);
    setFile(f);
    setLayers([]);
  }, [fileUrl]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const f = e.dataTransfer.files?.[0];
    if (f?.type?.startsWith('image/')) handleFile(f);
  }, [handleFile]);

  async function onDecompose() {
    if (!file) return;
    setLoading(true);
    setLoadingStep('Iniciando descomposición...');
    
    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('prompt', prompt);
      formData.append('layer_num', layerNum.toString());
      formData.append('num_inference_steps', steps.toString());
      if (seed !== '') formData.append('seed', seed.toString());

      setLoadingStep('Ejecutando Qwen-Image-Layered (esto puede tardar)...');
      const res = await fetch('/api/layered/decompose', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let errorMsg = 'Error en la descomposición';
        try {
          const err = await res.json();
          errorMsg = err.detail || errorMsg;
        } catch {
          errorMsg = `Error del servidor (${res.status}): ${res.statusText}`;
        }
        throw new Error(errorMsg);
      }

      const data = await res.json();
      setLayers(data.layers);
      console.log(`[Qwen] Decomposed into ${data.count} layers`);
    } catch (err: unknown) {
      console.error(err);
      const message = err instanceof Error ? err.message : 'Error desconocido';
      alert(`Error: ${message}`);
    } finally {
      setLoading(false);
      setLoadingStep('');
    }
  }

  function reset() {
    if (fileUrl) URL.revokeObjectURL(fileUrl);
    setFile(null);
    setFileUrl(null);
    setLayers([]);
    if (inputRef.current) inputRef.current.value = '';
  }

  return (
    <main className="min-h-screen bg-neutral-950 text-white pb-20">
      {/* Header */}
      <header className="border-b border-neutral-800 px-6 py-4 flex items-center justify-between sticky top-0 bg-neutral-950/80 backdrop-blur-md z-10">
        <div className="flex items-center gap-4">
          <Link href="/" className="text-neutral-500 hover:text-white transition-colors">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
            </svg>
          </Link>
          <div>
            <h1 className="text-lg font-semibold">Qwen Smart Layers</h1>
            <p className="text-xs text-neutral-500">Descomposición RGBA inteligente</p>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {!fileUrl ? (
          <div className="max-w-xl mx-auto">
            <button
              type="button"
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
              className={`
                w-full 
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
              <div className="text-5xl mb-4">🪄</div>
              <p className="text-lg font-medium text-neutral-300">
                Arrastra una imagen para descomponerla en capas
              </p>
              <p className="text-sm text-neutral-500 mt-2">
                Qwen-Image-Layered separará el fondo y los objetos automáticamente
              </p>
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
            {/* Sidebar Controls */}
            <div className="lg:col-span-1 space-y-6">
              <div className="bg-neutral-900 rounded-xl p-4 border border-neutral-800 space-y-4">
                <h2 className="text-sm font-semibold text-neutral-300 uppercase tracking-wider mb-2">Configuración</h2>
                
                <div>
                  <label className="block text-xs text-neutral-500 mb-1">Prompt descriptivo</label>
                  <textarea 
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 h-24 resize-none"
                  />
                </div>

                <div>
                  <label className="block text-xs text-neutral-500 mb-1">Número de capas: {layerNum}</label>
                  <input 
                    type="range" min="2" max="8" step="1"
                    value={layerNum}
                    onChange={(e) => setLayerNum(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-xs text-neutral-500 mb-1">Pasos de inferencia: {steps}</label>
                  <input 
                    type="range" min="20" max="100" step="5"
                    value={steps}
                    onChange={(e) => setSteps(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>

                <div>
                  <label className="block text-xs text-neutral-500 mb-1">Semilla (Seed - Opcional)</label>
                  <input 
                    type="number"
                    placeholder="Aleatoria si está vacío"
                    value={seed}
                    onChange={(e) => {
                      const val = parseInt(e.target.value);
                      setSeed(isNaN(val) ? '' : val);
                    }}
                    className="w-full bg-neutral-800 border border-neutral-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                </div>

                <button
                  onClick={onDecompose}
                  disabled={loading}
                  className={`
                    w-full py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2
                    ${loading
                      ? 'bg-blue-600/50 text-blue-300 cursor-wait'
                      : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20'}
                  `}
                >
                  {loading ? (
                    <>
                      <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Procesando...
                    </>
                  ) : 'Descomponer Capas'}
                </button>

                <button
                  onClick={reset}
                  disabled={loading}
                  className="w-full py-2 text-sm text-neutral-500 hover:text-white transition-colors"
                >
                  Cambiar imagen
                </button>
              </div>

              {loading && (
                <div className="bg-blue-900/20 border border-blue-800/50 rounded-xl p-4 space-y-3">
                  <p className="text-xs text-blue-300 text-center animate-pulse">
                    {loadingStep}
                  </p>
                  
                  {downloadProgress !== null && (
                    <div className="space-y-1">
                      <div className="flex justify-between text-[10px] text-blue-400">
                        <span>Descargando modelo...</span>
                        <span>{downloadProgress}%</span>
                      </div>
                      <div className="w-full bg-blue-900/30 rounded-full h-1.5 overflow-hidden">
                        <div 
                          className="bg-blue-500 h-full transition-all duration-500 ease-out"
                          style={{ width: `${downloadProgress}%` }}
                        />
                      </div>
                    </div>
                  )}

                  <p className="text-[10px] text-blue-400/70 text-center">
                    Nota: La primera ejecución descargará el modelo (~5GB)
                  </p>
                </div>
              )}
            </div>

            {/* Main Area */}
            <div className="lg:col-span-3 space-y-8">
              {/* Original Preview */}
              <div className="space-y-3">
                <h3 className="text-xs font-medium text-neutral-500 uppercase tracking-wider">Imagen Original</h3>
                <div className="rounded-2xl overflow-hidden border border-neutral-800 bg-neutral-900 max-w-2xl">
                  <Image
                    src={fileUrl}
                    alt="original"
                    width={0} height={0} sizes="100vw"
                    className="w-full h-auto"
                    unoptimized
                  />
                </div>
              </div>

              {/* Layers Results */}
              {layers.length > 0 && (
                <div className="space-y-6">
                  <h3 className="text-xs font-medium text-neutral-500 uppercase tracking-wider">Capas Generadas ({layers.length})</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {layers.map((layer, idx) => (
                      <div key={idx} className="space-y-2 group">
                        <div className="flex items-center justify-between px-1">
                          <span className="text-xs text-neutral-400">Capa {idx + 1}</span>
                          <a 
                            href={layer} 
                            download={`capa-${idx + 1}.png`}
                            className="text-[10px] bg-neutral-800 hover:bg-neutral-700 px-2 py-1 rounded transition-colors opacity-0 group-hover:opacity-100"
                          >
                            Descargar PNG
                          </a>
                        </div>
                        <div 
                          className="rounded-xl overflow-hidden border border-neutral-800 aspect-video relative"
                          style={{
                            backgroundImage: `url("data:image/svg+xml,%3Csvg width='20' height='20' xmlns='http://www.w3.org/2000/svg'%3E%3Crect width='10' height='10' fill='%23111'/%3E%3Crect x='10' y='10' width='10' height='10' fill='%23111'/%3E%3Crect x='10' width='10' height='10' fill='%23181818'/%3E%3Crect y='10' width='10' height='10' fill='%23181818'/%3E%3C/svg%3E")`,
                            backgroundSize: '20px 20px',
                          }}
                        >
                          <Image
                            src={layer}
                            alt={`layer-${idx}`}
                            fill
                            className="object-contain"
                            unoptimized
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

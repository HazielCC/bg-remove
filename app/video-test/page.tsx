"use client";

import { useState, useEffect } from "react";

interface ModelStatus {
  is_downloaded: boolean;
  is_downloading: boolean;
  downloaded_gb: number;
  detail: string;
}

export default function VideoTestPage() {
  const [prompt, setPrompt] = useState("");
  const [duration, setDuration] = useState(4);
  const [video, setVideo] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [checking, setChecking] = useState(true);

  // Approximate target size of Wan2.2-TI2V-5B in GB
  const EXPECTED_SIZE_GB = 34;

  const fetchStatus = async () => {
    try {
      const res = await fetch("/api/video/model-status");
      if (res.ok) {
        const data = await res.json();
        setModelStatus(data);
      }
    } catch (err) {
      console.error("No se pudo obtener el estado del modelo", err);
    } finally {
      setChecking(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(() => {
      fetchStatus();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const triggerDownload = async () => {
    try {
      await fetch("/api/video/download-model", { method: "POST" });
      fetchStatus();
    } catch (err) {
      console.error("Error iniciando descarga", err);
    }
  };

  const generate = async () => {
    if (!prompt.trim() || !modelStatus?.is_downloaded) return;

    setLoading(true);
    setError(null);
    setVideo(null);

    try {
      const res = await fetch("/api/video/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt, duration }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Error generating video");
      }

      const data = await res.json();
      setVideo(`http://localhost:8000${data.video_url}`);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Unknown error occurred");
      }
    } finally {
      setLoading(false);
    }
  };

  if (checking && !modelStatus) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-pulse text-gray-500">Comprobando estado del modelo...</div>
      </div>
    );
  }

  const isReady = modelStatus?.is_downloaded && !modelStatus?.is_downloading;

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 flex flex-col items-center py-12 px-4">
      <div className="max-w-2xl w-full bg-white rounded-2xl shadow p-8 space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 mb-2">Wan2.2 Video Sandbox</h1>
          <p className="text-gray-500">Prueba rápida del modelo Wan2.2-TI2V-5B local.</p>
        </div>

        {/* --- MODEL DOWNLOAD STATUS PANEL --- */}
        {!isReady && (
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-6 text-center space-y-4">
            <h2 className="text-xl font-semibold text-blue-900">Modelo no detectado localmente</h2>
            <p className="text-sm text-blue-700">
              {modelStatus?.detail || "Es necesario descargar el modelo Wan-AI/Wan2.2-TI2V-5B (~34 GB) antes de generar videos."}
            </p>

            {modelStatus?.is_downloading ? (
              <div className="w-full bg-blue-200 rounded-full h-4 mt-4 overflow-hidden relative">
                <div 
                  className="bg-blue-600 h-4 rounded-full transition-all duration-1000" 
                  style={{ width: `${Math.min((modelStatus.downloaded_gb / EXPECTED_SIZE_GB) * 100, 100)}%` }}
                />
                <div className="absolute inset-0 flex items-center justify-center text-[10px] font-bold text-white drop-shadow-md">
                  {modelStatus.downloaded_gb.toFixed(1)} GB / ~{EXPECTED_SIZE_GB} GB
                </div>
              </div>
            ) : (
              <button
                onClick={triggerDownload}
                className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
              >
                Iniciar Descarga
              </button>
            )}
            
            {modelStatus?.is_downloading && (
              <p className="text-xs text-blue-600 animate-pulse text-center">
                Descargando en segundo plano. Puedes minimizar esta ventana.
              </p>
            )}
          </div>
        )}

        {/* --- GENERATION FORM --- */}
        {isReady && (
          <div className="space-y-4 animate-in fade-in duration-500">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Prompt</label>
              <textarea
                className="w-full border border-gray-300 rounded-lg p-3 text-sm focus:ring-blue-500 focus:border-blue-500 outline-none"
                rows={4}
                placeholder="A cinematic shot of a futuristic cyberpunk city with neon lights..."
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                disabled={loading}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Duración (segundos)</label>
              <input
                type="number"
                min={1}
                max={10}
                className="w-full border border-gray-300 rounded-lg p-3 text-sm focus:ring-blue-500 focus:border-blue-500 outline-none"
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
                disabled={loading}
              />
            </div>

            <button
              onClick={generate}
              disabled={loading || !prompt.trim()}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Generando video (puede tardar minutos)...
                </>
              ) : (
                "Generar Video"
              )}
            </button>
          </div>
        )}

        {error && (
          <div className="p-4 bg-red-50 text-red-700 rounded-lg text-sm border border-red-200">
            <strong>Error:</strong> {error}
          </div>
        )}

        {video && (
          <div className="pt-6 border-t border-gray-100">
            <h2 className="text-lg font-medium text-gray-900 mb-4">Resultado:</h2>
            <video
              src={video}
              controls
              autoPlay
              loop
              className="w-full rounded-lg shadow-sm border border-gray-200 bg-black"
            />
          </div>
        )}
      </div>
    </div>
  );
}

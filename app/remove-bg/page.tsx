"use client";

import { RawImage } from '@huggingface/transformers';
import { useState } from 'react';
import { createSegmenter } from './modnet-client';

export default function RemoveBgPage() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [maskUrl, setMaskUrl] = useState<string | null>(null);
  const [variant, setVariant] = useState<'auto'|'fp32'|'fp16'|'uint8'>('auto');
  const [detectedDtype, setDetectedDtype] = useState<string | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);

  async function onRun() {
    if (!file) {
        alert('Debes subir un archivo de imagen antes de ejecutar el modelo');
        setLoading(false);
        return;
    }
    setLoading(true);
    try {
      const { segmenter, dtype } = await createSegmenter({ variant });
      setDetectedDtype(dtype);
      let output;
      if (file) {
        // If user uploaded a file, use RawImage.fromBlob(file) so pipeline receives bytes directly
        const raw = await RawImage.fromBlob(file);
        output = await segmenter(raw);
      } else {
        alert('No image selected');
        return;
      }
      // output[0] exposes `save` in transformers.js runtime to download to file, but
      // here we try to use `toBlob` if available.
      const result = output[0];
      if (result.toBlob) {
        const blob = await result.toBlob();
        const url = URL.createObjectURL(blob);
        setMaskUrl(url);
      } else {
        // Fallback: try save and instruct user to check downloads
        await result.save('mask.png');
        setMaskUrl(null);
      }
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(err);
      alert('Error ejecutando MODNet. Revisa la consola.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="p-6">
      <h1 className="text-2xl font-bold">Demo Remove BG (MODNet)</h1>
      <p className="mt-2">Sube un archivo de imagen (portrait recomendado) y pulsa Run.</p>

      <div className="mt-3">
        <label className="block mb-2">Or upload an image</label>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (!f) return;
            const url = URL.createObjectURL(f);
            setFileUrl(url);
            setImageUrl(url);
            setFile(f);
          }}
        />
        {fileUrl && (
          <div className="mt-3">
            <img src={fileUrl} className="max-w-xs border" alt="preview" />
          </div>
        )}
      </div>

      <div className="mt-3">
        <label className="text-sm mr-2">Model variant:</label>
        <select className="border p-2" value={variant} onChange={(e) => setVariant(e.target.value as any)}>
          <option value="auto">Auto</option>
          <option value="fp32">FP32</option>
          <option value="fp16">FP16</option>
          <option value="uint8">Quantized (uint8)</option>
        </select>
      </div>

      <div className="mt-4">
        <button
          className="bg-blue-600 text-white px-4 py-2 rounded"
          onClick={onRun}
          disabled={loading || !file}
        >
          {loading ? 'Running...' : 'Run'}
        </button>
      </div>

      {maskUrl && (
        <div className="mt-4">
          <h2 className="font-semibold">Mask result</h2>
          <img src={maskUrl} alt="mask" className="mt-2 max-w-full" />
          <div className="mt-2">
            <a href={maskUrl} download="mask.png" className="text-blue-600 underline">Download mask</a>
          </div>
        </div>
      )}
        {detectedDtype && (
          <div className="mt-4 text-sm text-gray-600">Model dtype used: {detectedDtype}</div>
        )}
    </main>
  );
}

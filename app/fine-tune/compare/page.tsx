"use client";

import { useEffect, useState } from "react";
import { apiFetch, apiPost, apiUpload } from "../lib/api";

interface ModelInfo {
    type: string;
    run?: string;
    name: string;
    path: string;
    size_mb: number;
}

interface CompareResult {
    filename: string;
    original: string;
    mask_a: string;
    mask_b: string;
}

export default function ComparePage() {
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [modelA, setModelA] = useState("");
    const [modelB, setModelB] = useState("");
    const [results, setResults] = useState<CompareResult[]>([]);
    const [loading, setLoading] = useState(false);

    // Custom image upload
    const [customImage, setCustomImage] = useState<File | null>(null);
    const [customResults, setCustomResults] = useState<{
        matte_a: string;
        matte_b: string;
        original: string;
    } | null>(null);

    useEffect(() => {
        apiFetch<ModelInfo[]>("/models/list").then(setModels).catch(console.error);
    }, []);

    const checkpoints = models.filter((m) => m.type === "checkpoint" || m.type === "pretrained");

    // ── Compare on dataset images ─────────────────────────
    const handleCompare = async () => {
        if (!modelA || !modelB) {
            alert("Selecciona ambos modelos");
            return;
        }
        setLoading(true);
        setResults([]);
        try {
            const data = await apiPost<{ results: CompareResult[] }>("/models/compare", {
                model_a: modelA,
                model_b: modelB,
            });
            setResults(data.results);
        } catch (e) {
            alert("Error de comparación: " + (e as Error).message);
        } finally {
            setLoading(false);
        }
    };

    // ── Compare on custom image ───────────────────────────
    const handleCustomCompare = async () => {
        if (!customImage || !modelA || !modelB) {
            alert("Sube una imagen y selecciona ambos modelos");
            return;
        }
        setLoading(true);
        setCustomResults(null);
        try {
            // Run inference with model A
            const formA = new FormData();
            formA.append("image", customImage);
            formA.append("checkpoint", modelA);
            const resA = await apiUpload<{ matte: string; result: string }>(
                "/inference/run",
                formA
            );

            // Run inference with model B
            const formB = new FormData();
            formB.append("image", customImage);
            formB.append("checkpoint", modelB);
            const resB = await apiUpload<{ matte: string; result: string }>(
                "/inference/run",
                formB
            );

            setCustomResults({
                original: URL.createObjectURL(customImage),
                matte_a: resA.matte,
                matte_b: resB.matte,
            });
        } catch (e) {
            alert("Error: " + (e as Error).message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="p-6 max-w-6xl">
            <h1 className="text-2xl font-bold mb-1">Comparación de Modelos</h1>
            <p className="text-sm text-neutral-500 mb-6">
                Compare baseline vs fine-tuned models side by side
            </p>

            {/* ── Model Selection ──────────────────────────── */}
            <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="border rounded-lg p-4 dark:border-neutral-700">
                    <label className="block text-sm font-medium mb-1">Modelo A</label>
                    <select
                        value={modelA}
                        onChange={(e) => setModelA(e.target.value)}
                        className="w-full border rounded-lg px-3 py-2 text-sm dark:bg-neutral-900 dark:border-neutral-700"
                    >
                        <option value="">-- Seleccionar modelo --</option>
                        {checkpoints.map((m) => (
                            <option key={m.path} value={m.path}>
                                {m.run ? `${m.run}/` : ""}
                                {m.name} ({m.size_mb} MB)
                            </option>
                        ))}
                    </select>
                </div>
                <div className="border rounded-lg p-4 dark:border-neutral-700">
                    <h3 className="text-sm font-semibold mb-2">Modelo B</h3>
                    <select
                        value={modelB}
                        onChange={(e) => setModelB(e.target.value)}
                        className="w-full border rounded-lg px-3 py-2 text-sm dark:bg-neutral-900 dark:border-neutral-700"
                    >
                        <option value="">-- Select model --</option>
                        {checkpoints.map((m) => (
                            <option key={m.path} value={m.path}>
                                {m.run ? `${m.run}/` : ""}
                                {m.name} ({m.size_mb} MB)
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {/* ── Actions ──────────────────────────────────── */}
            <div className="flex gap-3 mb-6">
                <button
                    onClick={handleCompare}
                    disabled={loading || !modelA || !modelB}
                    className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm disabled:opacity-50"
                >
                    {loading ? "Comparando..." : "Comparar en Dataset"}
                </button>

                <div className="flex items-center gap-2">
                    <input
                        type="file"
                        accept="image/*"
                        onChange={(e) => setCustomImage(e.target.files?.[0] || null)}
                        className="text-sm"
                    />
                    <button
                        onClick={handleCustomCompare}
                        disabled={loading || !modelA || !modelB || !customImage}
                        className="bg-green-600 text-white px-4 py-2 rounded-lg text-sm disabled:opacity-50"
                    >
                        Compare Custom Image
                    </button>
                </div>
            </div>

            {/* ── Custom Image Results ─────────────────────── */}
            {customResults && (
                <div className="mb-6 border rounded-lg p-4 dark:border-neutral-700">
                    <h3 className="text-sm font-semibold mb-3">Custom Image Comparison</h3>
                    <div className="grid grid-cols-3 gap-4">
                        <div>
                            <div className="text-xs text-neutral-500 mb-1">Original</div>
                            <img
                                src={customResults.original}
                                alt="original"
                                className="w-full rounded-lg border dark:border-neutral-700"
                            />
                        </div>
                        <div>
                            <div className="text-xs text-neutral-500 mb-1">Model A</div>
                            <img
                                src={customResults.matte_a}
                                alt="mask A"
                                className="w-full rounded-lg border dark:border-neutral-700 bg-black"
                            />
                        </div>
                        <div>
                            <div className="text-xs text-neutral-500 mb-1">Model B</div>
                            <img
                                src={customResults.matte_b}
                                alt="mask B"
                                className="w-full rounded-lg border dark:border-neutral-700 bg-black"
                            />
                        </div>
                    </div>
                </div>
            )}

            {/* ── Dataset Comparison Grid ──────────────────── */}
            {results.length > 0 && (
                <div className="border rounded-lg p-4 dark:border-neutral-700">
                    <h3 className="text-sm font-semibold mb-3">
                        Dataset Comparison ({results.length} images)
                    </h3>
                    <div className="space-y-4">
                        {results.map((r) => (
                            <div
                                key={r.filename}
                                className="grid grid-cols-3 gap-3 border-b dark:border-neutral-700 pb-4 last:border-0"
                            >
                                <div>
                                    <div className="text-xs text-neutral-500 mb-1">
                                        {r.filename}
                                    </div>
                                    <img
                                        src={r.original}
                                        alt={r.filename}
                                        className="w-full rounded border dark:border-neutral-700"
                                    />
                                </div>
                                <div>
                                    <div className="text-xs text-neutral-500 mb-1">Model A</div>
                                    <img
                                        src={r.mask_a}
                                        alt="A"
                                        className="w-full rounded border dark:border-neutral-700 bg-black"
                                    />
                                </div>
                                <div>
                                    <div className="text-xs text-neutral-500 mb-1">Model B</div>
                                    <img
                                        src={r.mask_b}
                                        alt="B"
                                        className="w-full rounded border dark:border-neutral-700 bg-black"
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* ── Empty state ──────────────────────────────── */}
            {results.length === 0 && !customResults && !loading && (
                <div className="text-center py-16 text-neutral-400">
                    <p className="text-lg mb-2">Select two models to compare</p>
                    <p className="text-sm">
                        Puedes comparar en imágenes del dataset o subir una imagen personalizada
                    </p>
                </div>
            )}
        </div>
    );
}

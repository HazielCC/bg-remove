"use client";

import { useEffect, useState } from "react";
import { apiFetch, apiPost } from "../lib/api";
import HelpTip from "../components/help-tip";

interface ModelInfo {
    type: string;
    run?: string;
    name: string;
    path: string;
    size_mb: number;
    modified?: number;
}

export default function ExportPage() {
    const [models, setModels] = useState<ModelInfo[]>([]);
    const [loading, setLoading] = useState(false);
    const [log, setLog] = useState<string[]>([]);

    // Selected model for operations
    const [selectedCkpt, setSelectedCkpt] = useState("");
    const [selectedOnnx, setSelectedOnnx] = useState("");
    const [exportSize, setExportSize] = useState(512);
    const [quantDtype, setQuantDtype] = useState<"fp16" | "uint8">("uint8");

    const fetchModels = () => {
        apiFetch<ModelInfo[]>("/models/list").then(setModels).catch(console.error);
    };

    useEffect(() => {
        fetchModels();
    }, []);

    const checkpoints = models.filter(
        (m) => m.type === "checkpoint" || m.type === "pretrained"
    );
    const onnxModels = models.filter((m) => m.type === "onnx");

    const addLog = (msg: string) => {
        setLog((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ${msg}`]);
    };

    const isNew = (timestamp?: number) => {
        if (!timestamp) return false;
        // Check if modified within last 30 minutes (1800 seconds)
        return Date.now() / 1000 - timestamp < 1800;
    };

    // ── Export to ONNX ────────────────────────────────────
    const handleExport = async () => {
        if (!selectedCkpt) {
            alert("Select a checkpoint first");
            return;
        }
        setLoading(true);
        addLog(`Exporting ${selectedCkpt} → ONNX (${exportSize}px)...`);
        try {
            const stem = selectedCkpt.split("/").pop()?.replace(".ckpt", "") || "model";
            const result = await apiPost<{ status: string; path: string; size_mb: number }>(
                `/models/${encodeURIComponent(stem)}/export-onnx`,
                { img_size: exportSize, path: selectedCkpt }
            );
            addLog(`Exported: ${result.path} (${result.size_mb} MB)`);
            fetchModels();
        } catch (e) {
            addLog(`ERROR: ${(e as Error).message}`);
        } finally {
            setLoading(false);
        }
    };

    // ── Quantize ──────────────────────────────────────────
    const handleQuantize = async () => {
        if (!selectedOnnx) {
            alert("Select an ONNX model first");
            return;
        }
        setLoading(true);
        addLog(`Quantizing ${selectedOnnx} → ${quantDtype}...`);
        try {
            const stem = selectedOnnx.split("/").pop()?.replace(".onnx", "") || "model";
            const result = await apiPost<{ status: string; path: string; size_mb: number }>(
                `/models/${encodeURIComponent(stem)}/quantize`,
                { dtype: quantDtype, path: selectedOnnx }
            );
            addLog(`Quantized: ${result.path} (${result.size_mb} MB)`);
            fetchModels();
        } catch (e) {
            addLog(`ERROR: ${(e as Error).message}`);
        } finally {
            setLoading(false);
        }
    };

    // ── Deploy to Frontend ────────────────────────────────
    const handleDeploy = async () => {
        if (!selectedOnnx) {
            alert("Select an ONNX model first");
            return;
        }
        setLoading(true);
        addLog(`Deploying ${selectedOnnx} → public/models/modnet/onnx/...`);
        try {
            const stem = selectedOnnx.split("/").pop()?.replace(".onnx", "") || "model";
            const result = await apiPost<{
                status: string;
                destination: string;
                filename: string;
            }>(`/models/${encodeURIComponent(stem)}/deploy`, {
                target_dir: "../public/models/modnet/onnx",
                path: selectedOnnx,
            });
            addLog(
                `Deployed! ${result.filename} → ${result.destination}`
            );
            addLog("Restart the dev server to pick up the new model.");
        } catch (e) {
            addLog(`ERROR: ${(e as Error).message}`);
        } finally {
            setLoading(false);
        }
    };

    // ── Delete ────────────────────────────────────────────
    const handleDelete = async (m: ModelInfo) => {
        if (m.type === "pretrained") {
            alert("No se pueden eliminar los modelos base.");
            return;
        }
        if (!confirm(`¿Estás seguro de que deseas eliminar ${m.name}? Esta acción no se puede deshacer.`)) {
            return;
        }

        try {
            await apiPost("/models/delete", { path: m.path });
            addLog(`Deleted: ${m.name}`);
            
            // Clear selection if deleted item was selected
            if (selectedCkpt === m.path) setSelectedCkpt("");
            if (selectedOnnx === m.path) setSelectedOnnx("");
            
            fetchModels();
        } catch (e) {
            addLog(`ERROR deleting ${m.name}: ${(e as Error).message}`);
        }
    };

    return (
        <div className="p-6 max-w-4xl">
            <h1 className="text-2xl font-bold mb-1">Export & Deploy</h1>
            <p className="text-sm text-neutral-500 mb-6">
                Export checkpoints to ONNX, quantize, and deploy to the frontend
            </p>

            <div className="space-y-6">
                {/* ── Step 1: Export ONNX ─────────────────────── */}
                <div className="border rounded-lg p-4 dark:border-neutral-700">
                    <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                        <span className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs">
                            1
                        </span>
                        Export Checkpoint → ONNX
                        <HelpTip text="Convierte un checkpoint de PyTorch a ONNX para inferencia portátil." />
                    </h3>
                    <div className="grid grid-cols-3 gap-3">
                        <div className="col-span-2">
                            <div className="text-xs text-neutral-500 mb-1 inline-flex items-center">
                                <span>Checkpoint</span>
                                <HelpTip text="Punto de partida entrenado. Selecciona exactamente el checkpoint que quieres exportar." />
                            </div>
                            <select
                                value={selectedCkpt}
                                onChange={(e) => setSelectedCkpt(e.target.value)}
                                className="w-full border rounded-lg px-3 py-2 text-sm dark:bg-neutral-900 dark:border-neutral-700"
                            >
                                <option value="">-- Select checkpoint --</option>
                                {checkpoints.map((m) => (
                                    <option key={m.path} value={m.path}>
                                        {m.run ? `${m.run}/` : ""}
                                        {m.name} ({m.size_mb} MB)
                                    </option>
                                ))}
                            </select>
                        </div>
                        <div>
                            <div className="text-xs text-neutral-500 mb-1 inline-flex items-center">
                                <span>Input Size</span>
                                <HelpTip text="Resolución base usada al exportar. Debe coincidir con lo esperado en evaluación/inferencia." />
                            </div>
                            <input
                                type="number"
                                value={exportSize}
                                onChange={(e) => setExportSize(Number(e.target.value))}
                                className="w-full border rounded-lg px-3 py-2 text-sm dark:bg-neutral-900 dark:border-neutral-700"
                            />
                        </div>
                    </div>
                    <button
                        onClick={handleExport}
                        disabled={loading || !selectedCkpt}
                        className="mt-3 bg-blue-600 text-white px-4 py-2 rounded-lg text-sm disabled:opacity-50"
                    >
                        {loading ? "Exporting..." : "Export to ONNX"}
                    </button>
                </div>

                {/* ── Step 2: Quantize ───────────────────────── */}
                <div className="border rounded-lg p-4 dark:border-neutral-700">
                    <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                        <span className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs">
                            2
                        </span>
                        Quantize ONNX
                        <HelpTip text="Reduce tamaño y costo computacional del ONNX para inferencia más rápida." />
                    </h3>
                    <div className="grid grid-cols-3 gap-3">
                        <div className="col-span-2">
                            <div className="text-xs text-neutral-500 mb-1 inline-flex items-center">
                                <span>ONNX Model</span>
                                <HelpTip text="Modelo ONNX origen sobre el que se aplicará cuantización." />
                            </div>
                            <select
                                value={selectedOnnx}
                                onChange={(e) => setSelectedOnnx(e.target.value)}
                                className="w-full border rounded-lg px-3 py-2 text-sm dark:bg-neutral-900 dark:border-neutral-700"
                            >
                                <option value="">-- Select ONNX model --</option>
                                {onnxModels.map((m) => (
                                    <option key={m.path} value={m.path}>
                                        {m.name} ({m.size_mb} MB)
                                    </option>
                                ))}
                            </select>
                        </div>
                        <div>
                            <div className="text-xs text-neutral-500 mb-1 inline-flex items-center">
                                <span>Target</span>
                                <HelpTip text="UINT8: más pequeño/rápido. FP16: mejor precisión relativa, tamaño intermedio." />
                            </div>
                            <select
                                value={quantDtype}
                                onChange={(e) =>
                                    setQuantDtype(e.target.value as "fp16" | "uint8")
                                }
                                className="w-full border rounded-lg px-3 py-2 text-sm dark:bg-neutral-900 dark:border-neutral-700"
                            >
                                <option value="uint8">UINT8 (smallest)</option>
                                <option value="fp16">FP16 (balanced)</option>
                            </select>
                        </div>
                    </div>
                    <button
                        onClick={handleQuantize}
                        disabled={loading || !selectedOnnx}
                        className="mt-3 bg-purple-600 text-white px-4 py-2 rounded-lg text-sm disabled:opacity-50"
                    >
                        {loading ? "Quantizing..." : "Quantize"}
                    </button>
                </div>

                {/* ── Step 3: Deploy ─────────────────────────── */}
                <div className="border rounded-lg p-4 dark:border-neutral-700">
                    <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                        <span className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center text-xs">
                            3
                        </span>
                        Deploy to Frontend
                        <HelpTip text="Copia el ONNX elegido al directorio público para usarlo en la demo del navegador." />
                    </h3>
                    <p className="text-xs text-neutral-500 mb-3">
                        Copy the selected ONNX model to{" "}
                        <code className="px-1 py-0.5 bg-neutral-100 dark:bg-neutral-800 rounded">
                            public/models/modnet/onnx/
                        </code>{" "}
                        for use in the browser inference demo.
                    </p>
                    <div className="mb-3">
                        <div className="text-xs text-neutral-500 mb-1 inline-flex items-center">
                            <span>ONNX Model to Deploy</span>
                            <HelpTip text="Modelo ONNX final que quedará disponible en `public/models/modnet/onnx`." />
                        </div>
                        <select
                            value={selectedOnnx}
                            onChange={(e) => setSelectedOnnx(e.target.value)}
                            className="w-full border rounded-lg px-3 py-2 text-sm dark:bg-neutral-900 dark:border-neutral-700"
                        >
                            <option value="">-- Select ONNX model --</option>
                            {onnxModels.map((m) => (
                                <option key={m.path} value={m.path}>
                                    {m.name} ({m.size_mb} MB)
                                </option>
                            ))}
                        </select>
                    </div>
                    <button
                        onClick={handleDeploy}
                        disabled={loading || !selectedOnnx}
                        className="bg-green-600 text-white px-4 py-2 rounded-lg text-sm disabled:opacity-50"
                    >
                        {loading ? "Deploying..." : "Deploy to Frontend"}
                    </button>
                </div>

                {/* ── Available Models ────────────────────────── */}
                <div className="border rounded-lg p-4 dark:border-neutral-700">
                    <h3 className="text-sm font-semibold mb-3">Todos los Modelos</h3>
                    {models.length === 0 ? (
                        <p className="text-sm text-neutral-500 dark:text-neutral-400">No se encontraron modelos</p>
                    ) : (
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="text-left text-xs text-neutral-500 border-b dark:border-neutral-700">
                                    <th className="pb-2">Name</th>
                                    <th className="pb-2">Type</th>
                                    <th className="pb-2">Run</th>
                                    <th className="pb-2 text-right">Size</th>
                                    <th className="pb-2 text-right w-16">Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {models.map((m) => (
                                    <tr
                                        key={m.path}
                                        className="border-b dark:border-neutral-800 hover:bg-neutral-50 dark:hover:bg-neutral-900/50"
                                    >
                                        <td className="py-2 font-mono text-xs">
                                            {m.name}
                                            {isNew(m.modified) && (
                                                <span className="ml-2 text-[10px] bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded-full font-sans font-medium">New!</span>
                                            )}
                                        </td>
                                        <td className="py-2">
                                            <span
                                                className={`text-xs px-2 py-0.5 rounded-full ${m.type === "onnx"
                                                    ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400"
                                                    : m.type === "pretrained"
                                                        ? "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400"
                                                        : "bg-neutral-100 dark:bg-neutral-800 text-neutral-600"
                                                    }`}
                                            >
                                                {m.type}
                                            </span>
                                        </td>
                                        <td className="py-2 text-neutral-500">{m.run || "—"}</td>
                                        <td className="py-2 text-right">{m.size_mb} MB</td>
                                        <td className="py-2 text-right">
                                            {m.type !== "pretrained" && (
                                                <button
                                                    onClick={() => handleDelete(m)}
                                                    className="text-neutral-400 hover:text-red-500 p-1 rounded hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                                                    title="Delete"
                                                >
                                                    <svg
                                                        xmlns="http://www.w3.org/2000/svg"
                                                        viewBox="0 0 20 20"
                                                        fill="currentColor"
                                                        className="w-4 h-4"
                                                    >
                                                        <path
                                                            fillRule="evenodd"
                                                            d="M8.75 1A2.75 2.75 0 006 3.75v.443c-.795.077-1.584.176-2.365.298a.75.75 0 10.23 1.482l.149-.022.841 10.518A2.75 2.75 0 007.596 19h4.807a2.75 2.75 0 002.742-2.53l.841-10.52.149.023a.75.75 0 00.23-1.482A41.03 41.03 0 0014 4.193V3.75A2.75 2.75 0 0011.25 1h-2.5zM10 4c.84 0 1.673.025 2.5.075V3.75c0-.69-.56-1.25-1.25-1.25h-2.5c-.69 0-1.25.56-1.25 1.25v.325C8.327 4.025 9.16 4 10 4zM8.58 7.72a.75.75 0 00-1.5.06l.3 7.5a.75.75 0 101.5-.06l-.3-7.5zm4.34.06a.75.75 0 10-1.5-.06l-.3 7.5a.75.75 0 101.5.06l.3-7.5z"
                                                            clipRule="evenodd"
                                                        />
                                                    </svg>
                                                </button>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>

                {/* ── Operation Log ──────────────────────────── */}
                {log.length > 0 && (
                    <div className="border rounded-lg dark:border-neutral-700">
                        <div className="px-4 py-2 border-b dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800 flex justify-between items-center">
                            <h3 className="text-sm font-semibold">Log</h3>
                            <button
                                onClick={() => setLog([])}
                                className="text-xs text-neutral-500 hover:text-neutral-700 dark:hover:text-neutral-300 transition-colors"
                            >
                                Clear Logs
                            </button>
                        </div>
                        <div className="p-3 font-mono text-xs space-y-0.5 max-h-40 overflow-y-auto bg-neutral-900 text-neutral-300 rounded-b-lg">
                            {log.map((line, i) => (
                                <div
                                    key={i}
                                    className={
                                        line.includes("ERROR") ? "text-red-400" : ""
                                    }
                                >
                                    {line}
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch, apiPost, createMetricsStream } from "../lib/api";
import HelpTip from "../components/help-tip";

interface EpochMetric {
    epoch: number;
    total_loss: number;
    semantic_loss?: number;
    detail_loss?: number;
    matte_loss?: number;
    val_loss?: number;
    lr?: number;
    elapsed_seconds?: number;
    eta_seconds?: number;
}

interface BatchProgress {
    epoch: number;
    total_epochs: number;
    batch: number;
    total_batches: number;
    batch_loss?: number;
    elapsed_seconds?: number;
}

interface TrainingStatus {
    status: string;
    epoch: number;
    total_epochs: number;
    total_loss: number;
    semantic_loss: number;
    detail_loss: number;
    matte_loss: number;
    val_loss: number;
    lr: number;
    samples_processed: number;
    elapsed_seconds: number;
    eta_seconds: number;
    best_val_loss: number;
    error_message: string;
}

interface DatasetInfo {
    id: string;
    name: string;
    images_count: number;
    alphas_count: number;
}

interface BenchmarkModelRow {
    rank?: number;
    checkpoint: string;
    run?: string;
    name: string;
    status: "ok" | "error";
    n_images?: number;
    avg_metrics?: {
        sad?: number;
        mse?: number;
        gradient_error?: number;
        connectivity_error?: number;
    };
    avg_inference_ms?: number | null;
    overall_score?: number;
    error?: string;
}

interface BenchmarkResponse {
    dataset_id: string;
    n_images: number;
    ranked_models: BenchmarkModelRow[];
    failed_models: BenchmarkModelRow[];
    metric_baselines?: {
        sad?: number;
        mse?: number;
        gradient_error?: number;
        connectivity_error?: number;
    };
    score_formula: string;
}

function makeEmptyStatus(): TrainingStatus {
    return {
        status: "idle",
        epoch: 0,
        total_epochs: 0,
        total_loss: 0,
        semantic_loss: 0,
        detail_loss: 0,
        matte_loss: 0,
        val_loss: 0,
        lr: 0,
        samples_processed: 0,
        elapsed_seconds: 0,
        eta_seconds: 0,
        best_val_loss: Infinity,
        error_message: "",
    };
}

export default function MonitorPage() {
    const [status, setStatus] = useState<TrainingStatus | null>(null);
    const [metrics, setMetrics] = useState<EpochMetric[]>([]);
    const [batchProgress, setBatchProgress] = useState<BatchProgress | null>(null);
    const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
    const [benchmarkDatasetId, setBenchmarkDatasetId] = useState("");
    const [benchmarkMaxImages, setBenchmarkMaxImages] = useState(40);
    const [benchmarkImgSize, setBenchmarkImgSize] = useState(512);
    const [benchmarkLoading, setBenchmarkLoading] = useState(false);
    const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResponse | null>(
        null
    );
    const [logs, setLogs] = useState<string[]>([]);
    const [connected, setConnected] = useState(false);
    const sourceRef = useRef<EventSource | null>(null);
    const logsEndRef = useRef<HTMLDivElement>(null);

    const appendLog = useCallback((line: string) => {
        setLogs((prev) => [...prev.slice(-249), line]);
    }, []);

    // Fetch initial status
    useEffect(() => {
        apiFetch<TrainingStatus>("/training/status")
            .then(setStatus)
            .catch(console.error);
    }, []);

    useEffect(() => {
        apiFetch<DatasetInfo[]>("/datasets/local")
            .then((rows) => {
                setDatasets(rows);
                if (rows.length > 0) {
                    setBenchmarkDatasetId((prev) => prev || rows[0].id);
                }
            })
            .catch(console.error);
    }, []);

    // Connect to SSE
    const connect = useCallback((mode: "manual" | "auto" = "manual") => {
        if (sourceRef.current) {
            sourceRef.current.close();
            sourceRef.current = null;
        }

        const source = createMetricsStream(
            (data) => {
                const type = data.type as string;

                if (type === "status") {
                    const streamStatus = String(data.status ?? "idle");
                    setStatus((prev) => ({
                        ...(prev ?? makeEmptyStatus()),
                        status: streamStatus,
                    }));
                    return;
                }

                if (type === "batch_progress") {
                    const b = data as unknown as BatchProgress;
                    setBatchProgress(b);
                    setStatus((prev) => ({
                        ...(prev ?? makeEmptyStatus()),
                        status: "running",
                        epoch: b.epoch,
                        total_epochs: b.total_epochs,
                        total_loss: b.batch_loss ?? prev?.total_loss ?? 0,
                        elapsed_seconds: b.elapsed_seconds ?? prev?.elapsed_seconds ?? 0,
                    }));

                    if (b.batch === 1 || b.batch === b.total_batches) {
                        appendLog(
                            `[Epoch ${b.epoch}] batch ${b.batch}/${b.total_batches} loss=${(b.batch_loss ?? 0).toFixed(4)}`
                        );
                    }
                    return;
                }

                if (type === "epoch_end") {
                    const m = data as unknown as EpochMetric;
                    setBatchProgress(null);
                    setMetrics((prev) => [...prev, m]);
                    appendLog(
                        `[Epoch ${m.epoch}] loss=${(m.total_loss ?? 0).toFixed(4)} val=${(m.val_loss ?? 0).toFixed(4)} lr=${(m.lr ?? 0).toFixed(6)}`
                    );
                    setStatus((prev) => ({
                        ...(prev ?? makeEmptyStatus()),
                        status: "running",
                        epoch: m.epoch,
                        total_loss: m.total_loss ?? 0,
                        semantic_loss: m.semantic_loss ?? 0,
                        detail_loss: m.detail_loss ?? 0,
                        matte_loss: m.matte_loss ?? 0,
                        val_loss: m.val_loss ?? 0,
                        lr: m.lr ?? 0,
                        elapsed_seconds: m.elapsed_seconds ?? 0,
                        eta_seconds: m.eta_seconds ?? 0,
                    }));
                    return;
                }

                if (type === "heartbeat") {
                    const heartbeatStatus = String(data.status ?? "idle");
                    setStatus((prev) => ({
                        ...(prev ?? makeEmptyStatus()),
                        status: heartbeatStatus,
                        epoch: Number(data.epoch ?? prev?.epoch ?? 0),
                    }));
                    return;
                }

                if (type === "finished") {
                    setBatchProgress(null);
                    appendLog(`Training finished! Best val loss: ${data.best_val_loss}`);
                    setStatus((prev) => ({
                        ...(prev ?? makeEmptyStatus()),
                        status: "finished",
                    }));
                    return;
                }

                if (type === "error") {
                    setBatchProgress(null);
                    appendLog(`ERROR: ${data.message}`);
                    setStatus((prev) => ({
                        ...(prev ?? makeEmptyStatus()),
                        status: "error",
                        error_message: String(data.message ?? ""),
                    }));
                    return;
                }

                if (type === "stopped") {
                    setBatchProgress(null);
                    appendLog("Training stopped by user");
                    setStatus((prev) => ({
                        ...(prev ?? makeEmptyStatus()),
                        status: "stopped",
                    }));
                }
            },
            () => {
                setConnected(false);
            }
        );

        source.onopen = () => {
            setConnected(true);
            if (mode === "manual") {
                appendLog("SSE connected");
            }
        };

        sourceRef.current = source;
    }, [appendLog]);

    useEffect(() => {
        connect("auto");
        return () => {
            if (sourceRef.current) {
                sourceRef.current.close();
                sourceRef.current = null;
            }
        };
    }, [connect]);

    // Auto-scroll logs
    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [logs]);

    // Stop training
    const handleStop = async () => {
        try {
            await apiPost("/training/stop", {});
            appendLog("Stop requested...");
        } catch (e) {
            console.error(e);
        }
    };

    const handleBenchmark = async () => {
        if (!benchmarkDatasetId) {
            appendLog("ERROR: Selecciona un dataset para benchmark");
            return;
        }
        setBenchmarkLoading(true);
        appendLog(
            `Benchmark started | dataset=${benchmarkDatasetId} | img_size=${benchmarkImgSize} | max_images=${benchmarkMaxImages}`
        );
        try {
            const result = await apiPost<BenchmarkResponse>("/models/benchmark", {
                dataset_id: benchmarkDatasetId,
                max_images: benchmarkMaxImages,
                img_size: benchmarkImgSize,
            });
            setBenchmarkResult(result);
            appendLog(
                `Benchmark finished | ranked=${result.ranked_models.length} | failed=${result.failed_models.length}`
            );
        } catch (e) {
            appendLog(`ERROR: benchmark failed: ${(e as Error).message}`);
        } finally {
            setBenchmarkLoading(false);
        }
    };

    const formatTime = (seconds: number) => {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}m ${s}s`;
    };

    const fmtMetric = (value?: number | null, decimals = 4) =>
        typeof value === "number" && Number.isFinite(value)
            ? value.toFixed(decimals)
            : "—";

    const maxLoss = metrics.length
        ? Math.max(...metrics.map((m) => m.total_loss || 0)) * 1.1
        : 1;
    const epochProgressPct =
        status && status.total_epochs > 0
            ? Math.min((status.epoch / status.total_epochs) * 100, 100)
            : 0;
    const batchProgressPct =
        batchProgress && batchProgress.total_batches > 0
            ? Math.min((batchProgress.batch / batchProgress.total_batches) * 100, 100)
            : 0;

    return (
        <div className="p-6 max-w-6xl">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h1 className="text-2xl font-bold">Training Monitor</h1>
                    <p className="text-sm text-neutral-500">
                        Real-time training metrics via SSE
                    </p>
                    <div className="mt-1 flex items-center gap-3 text-xs text-neutral-500">
                        <span className="inline-flex items-center">
                            Live Stream
                            <HelpTip text="El monitor se conecta automáticamente al stream SSE para recibir métricas en vivo." />
                        </span>
                        <span className="inline-flex items-center">
                            Stop
                            <HelpTip text="Solicita detener el entrenamiento actual. El cambio puede tardar unos segundos." />
                        </span>
                    </div>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={() => connect("manual")}
                        disabled={connected}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm disabled:opacity-50"
                    >
                        {connected ? "Connected" : "Reconnect"}
                    </button>
                    <button
                        onClick={handleStop}
                        className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm"
                    >
                        Stop
                    </button>
                </div>
            </div>

            {/* ── Status Cards ─────────────────────────────── */}
            {status && (
                <div className="grid grid-cols-5 gap-3 mb-6">
                    <StatusCard
                        label="Status"
                        value={status.status}
                        color={
                            status.status === "running"
                                ? "text-green-600"
                                : status.status === "error"
                                    ? "text-red-600"
                                    : "text-neutral-600"
                        }
                    />
                    <StatusCard
                        label="Epoch"
                        value={`${status.epoch} / ${status.total_epochs}`}
                    />
                    <StatusCard
                        label="Loss"
                        value={status.total_loss.toFixed(4)}
                        tooltip="Pérdida total del último epoch reportado."
                    />
                    <StatusCard
                        label="Val Loss"
                        value={status.val_loss.toFixed(4)}
                        tooltip="Pérdida en validación. Mejor para comparar calidad entre epochs."
                    />
                    <StatusCard
                        label="ETA"
                        value={formatTime(status.eta_seconds)}
                        tooltip="Tiempo estimado restante según la velocidad de epochs previos."
                    />
                </div>
            )}

            {status && status.total_epochs > 0 && (
                <div className="mb-6 space-y-3">
                    <div>
                        <div className="flex items-center justify-between text-xs text-neutral-500 mb-1">
                            <span>Progreso por época</span>
                            <span>{status.epoch}/{status.total_epochs} ({epochProgressPct.toFixed(0)}%)</span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-neutral-200 dark:bg-neutral-700 overflow-hidden">
                            <div
                                className="h-2 rounded-full bg-blue-500 transition-all"
                                style={{ width: `${Math.max(epochProgressPct, 0)}%` }}
                            />
                        </div>
                    </div>
                    {batchProgress && (
                        <div>
                            <div className="flex items-center justify-between text-xs text-neutral-500 mb-1">
                                <span>Progreso del batch actual (Epoch {batchProgress.epoch})</span>
                                <span>
                                    {batchProgress.batch}/{batchProgress.total_batches} ({batchProgressPct.toFixed(0)}%)
                                </span>
                            </div>
                            <div className="h-2 w-full rounded-full bg-neutral-200 dark:bg-neutral-700 overflow-hidden">
                                <div
                                    className="h-2 rounded-full bg-emerald-500 transition-all"
                                    style={{ width: `${Math.max(batchProgressPct, 0)}%` }}
                                />
                            </div>
                        </div>
                    )}
                </div>
            )}

            <div className="mb-6 border rounded-lg p-4 dark:border-neutral-700">
                <div className="flex items-center justify-between gap-3 mb-3">
                    <div>
                        <h3 className="text-sm font-semibold inline-flex items-center">
                            Benchmark de Checkpoints
                            <HelpTip text="Evalúa todos los checkpoints con métricas objetivas sobre un dataset local etiquetado (images+alphas)." />
                        </h3>
                        <p className="text-xs text-neutral-500 mt-1">
                            Score menor = mejor. Basado en SAD, MSE, gradient y connectivity.
                        </p>
                    </div>
                    <button
                        onClick={handleBenchmark}
                        disabled={benchmarkLoading || !benchmarkDatasetId}
                        className="px-4 py-2 bg-emerald-600 text-white rounded-lg text-sm disabled:opacity-50"
                    >
                        {benchmarkLoading ? "Benchmark..." : "Run Benchmark"}
                    </button>
                </div>

                <div className="grid grid-cols-3 gap-3 mb-4">
                    <div>
                        <label className="text-xs text-neutral-500 mb-1 block">Dataset</label>
                        <select
                            value={benchmarkDatasetId}
                            onChange={(e) => setBenchmarkDatasetId(e.target.value)}
                            className="w-full border rounded-lg px-3 py-2 text-sm dark:bg-neutral-900 dark:border-neutral-700"
                        >
                            <option value="">-- Selecciona dataset --</option>
                            {datasets.map((d) => (
                                <option key={d.id} value={d.id}>
                                    {d.name} ({d.images_count} imgs / {d.alphas_count} alpha)
                                </option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="text-xs text-neutral-500 mb-1 block">Max Images</label>
                        <input
                            type="number"
                            min={1}
                            value={benchmarkMaxImages}
                            onChange={(e) =>
                                setBenchmarkMaxImages(Math.max(1, Number(e.target.value) || 1))
                            }
                            className="w-full border rounded-lg px-3 py-2 text-sm dark:bg-neutral-900 dark:border-neutral-700"
                        />
                    </div>
                    <div>
                        <label className="text-xs text-neutral-500 mb-1 block">Image Size</label>
                        <input
                            type="number"
                            min={32}
                            value={benchmarkImgSize}
                            onChange={(e) =>
                                setBenchmarkImgSize(Math.max(32, Number(e.target.value) || 32))
                            }
                            className="w-full border rounded-lg px-3 py-2 text-sm dark:bg-neutral-900 dark:border-neutral-700"
                        />
                    </div>
                </div>

                {benchmarkResult && (
                    <div className="space-y-3">
                        <div className="text-xs text-neutral-500">
                            Dataset: <span className="font-mono">{benchmarkResult.dataset_id}</span>
                            {" · "}
                            Samples: {benchmarkResult.n_images}
                            {" · "}
                            Formula: <span className="font-mono">{benchmarkResult.score_formula}</span>
                        </div>

                        <div className="overflow-x-auto">
                            <table className="min-w-full text-xs border-collapse">
                                <thead>
                                    <tr className="text-left border-b dark:border-neutral-700">
                                        <th className="py-2 pr-3">Rank</th>
                                        <th className="py-2 pr-3">Model</th>
                                        <th className="py-2 pr-3">SAD</th>
                                        <th className="py-2 pr-3">MSE</th>
                                        <th className="py-2 pr-3">Grad</th>
                                        <th className="py-2 pr-3">Conn</th>
                                        <th className="py-2 pr-3">ms/img</th>
                                        <th className="py-2 pr-0">Score</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {benchmarkResult.ranked_models.map((row) => (
                                        <tr
                                            key={row.checkpoint}
                                            className={`border-b dark:border-neutral-800 ${row.rank === 1 ? "bg-emerald-50 dark:bg-emerald-900/20" : ""}`}
                                        >
                                            <td className="py-2 pr-3 font-semibold">
                                                {row.rank}
                                            </td>
                                            <td className="py-2 pr-3">
                                                <div className="font-medium">{row.name}</div>
                                                <div className="text-neutral-500">{row.run ?? "—"}</div>
                                            </td>
                                            <td className="py-2 pr-3">
                                                {fmtMetric(row.avg_metrics?.sad)}
                                            </td>
                                            <td className="py-2 pr-3">
                                                {fmtMetric(row.avg_metrics?.mse, 6)}
                                            </td>
                                            <td className="py-2 pr-3">
                                                {fmtMetric(row.avg_metrics?.gradient_error)}
                                            </td>
                                            <td className="py-2 pr-3">
                                                {fmtMetric(row.avg_metrics?.connectivity_error)}
                                            </td>
                                            <td className="py-2 pr-3">
                                                {fmtMetric(row.avg_inference_ms, 2)}
                                            </td>
                                            <td className="py-2 pr-0 font-semibold">
                                                {fmtMetric(row.overall_score, 4)}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {benchmarkResult.failed_models.length > 0 && (
                            <div className="text-xs text-amber-600 dark:text-amber-400">
                                Checkpoints con error: {benchmarkResult.failed_models.length}
                            </div>
                        )}
                    </div>
                )}
            </div>

            <div className="grid grid-cols-2 gap-6">
                {/* ── Loss Chart (simple bar chart) ────────── */}
                <div className="border rounded-lg p-4 dark:border-neutral-700">
                    <h3 className="text-sm font-semibold mb-3">Pérdida por Época</h3>
                    {metrics.length === 0 ? (
                        <div className="h-48 flex items-center justify-center text-sm text-neutral-500 dark:text-neutral-400">
                            Waiting for data...
                        </div>
                    ) : (
                        <div className="h-48 flex items-end gap-[2px]">
                            {metrics.map((m) => {
                                const h = ((m.total_loss || 0) / maxLoss) * 100;
                                return (
                                    <div
                                        key={m.epoch}
                                        className="flex-1 group relative"
                                    >
                                        <div
                                            className="bg-blue-500 rounded-t-sm transition-all hover:bg-blue-400"
                                            style={{ height: `${Math.max(h, 2)}%` }}
                                        />
                                        {/* Tooltip */}
                                        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-10">
                                            <div className="bg-black text-white text-[10px] px-2 py-1 rounded whitespace-nowrap">
                                                E{m.epoch}: {(m.total_loss ?? 0).toFixed(4)}
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                    <div className="flex justify-between text-[10px] text-neutral-500 dark:text-neutral-400 mt-1">
                        <span>Epoch 1</span>
                        <span>Epoch {metrics.length || "?"}</span>
                    </div>
                </div>

                {/* ── Loss Breakdown ─────────────────────────── */}
                <div className="border rounded-lg p-4 dark:border-neutral-700">
                    <h3 className="text-sm font-semibold mb-3">Pérdida por Época</h3>
                    {metrics.length === 0 ? (
                        <div className="h-48 flex items-center justify-center text-sm text-neutral-500 dark:text-neutral-400">
                            Esperando datos...
                        </div>
                    ) : (
                        <div className="space-y-2 h-48 overflow-y-auto">
                            {metrics.slice(-10).map((m) => (
                                <div key={m.epoch} className="flex items-center gap-2 text-xs">
                                    <span className="w-8 text-neutral-500">E{m.epoch}</span>
                                    <BarSegment
                                        label="S"
                                        value={m.semantic_loss ?? 0}
                                        max={maxLoss}
                                        color="bg-purple-500"
                                    />
                                    <BarSegment
                                        label="D"
                                        value={m.detail_loss ?? 0}
                                        max={maxLoss}
                                        color="bg-orange-500"
                                    />
                                    <BarSegment
                                        label="M"
                                        value={m.matte_loss ?? 0}
                                        max={maxLoss}
                                        color="bg-green-500"
                                    />
                                </div>
                            ))}
                        </div>
                    )}
                    <div className="flex gap-4 mt-2 text-[10px]">
                        <span className="flex items-center gap-1">
                            <span className="w-2 h-2 bg-purple-500 rounded-full" /> Semantic
                        </span>
                        <span className="flex items-center gap-1">
                            <span className="w-2 h-2 bg-orange-500 rounded-full" /> Detail
                        </span>
                        <span className="flex items-center gap-1">
                            <span className="w-2 h-2 bg-green-500 rounded-full" /> Matte
                        </span>
                    </div>
                </div>
            </div>

            {/* ── Learning Rate + Validation ───────────────── */}
            {status && (
                <div className="grid grid-cols-3 gap-3 mt-4">
                    <StatusCard
                        label="Learning Rate"
                        value={status.lr.toExponential(2)}
                        tooltip="Learning rate actual aplicado por el scheduler."
                    />
                    <StatusCard
                        label="Best Val Loss"
                        value={status.best_val_loss != null && status.best_val_loss !== Infinity ? status.best_val_loss.toFixed(4) : "—"}
                        tooltip="Mejor valor de validación observado en la corrida."
                    />
                    <StatusCard
                        label="Elapsed"
                        value={formatTime(status.elapsed_seconds)}
                        tooltip="Tiempo total transcurrido desde el inicio del entrenamiento."
                    />
                </div>
            )}

            {/* ── Console Log ──────────────────────────────── */}
            <div className="mt-6 border rounded-lg dark:border-neutral-700">
                <div className="flex items-center justify-between px-4 py-2 border-b dark:border-neutral-700 bg-neutral-50 dark:bg-neutral-800">
                    <h3 className="text-sm font-semibold">Console</h3>
                    <span
                        className={`text-xs px-2 py-0.5 rounded-full ${connected
                            ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                            : "bg-neutral-100 text-neutral-500 dark:bg-neutral-700"
                            }`}
                    >
                        {connected ? "Live" : "Disconnected"}
                    </span>
                </div>
                <div className="h-48 overflow-y-auto p-3 font-mono text-xs space-y-0.5 bg-neutral-900 text-neutral-300">
                    {logs.length === 0 && (
                        <span className="text-neutral-600">
                            Waiting for training events...
                        </span>
                    )}
                    {logs.map((line, i) => (
                        <div
                            key={i}
                            className={
                                line.includes("ERROR")
                                    ? "text-red-400"
                                    : line.includes("finished")
                                        ? "text-green-400"
                                        : ""
                            }
                        >
                            {line}
                        </div>
                    ))}
                    <div ref={logsEndRef} />
                </div>
            </div>
        </div>
    );
}

// ── Sub-components ──────────────────────────────────────
function StatusCard({
    label,
    value,
    color,
    tooltip,
}: {
    label: string;
    value: string;
    color?: string;
    tooltip?: string;
}) {
    return (
        <div className="bg-neutral-50 dark:bg-neutral-800 border dark:border-neutral-700 rounded-lg p-3 text-center">
            <div className={`text-lg font-bold ${color || ""}`}>{value}</div>
            <div className="mt-0.5 inline-flex items-center justify-center text-[10px] text-neutral-500">
                {label}
                {tooltip && <HelpTip text={tooltip} />}
            </div>
        </div>
    );
}

function BarSegment({
    label,
    value,
    max,
    color,
}: {
    label: string;
    value: number;
    max: number;
    color: string;
}) {
    const pct = Math.min((value / max) * 100, 100);
    return (
        <div className="flex-1 flex items-center gap-1">
            <span className="w-3 text-neutral-500">{label}</span>
            <div className="flex-1 bg-neutral-200 dark:bg-neutral-700 rounded-full h-2">
                <div
                    className={`${color} h-2 rounded-full transition-all`}
                    style={{ width: `${Math.max(pct, 1)}%` }}
                />
            </div>
            <span className="w-12 text-right text-neutral-500">
                {value.toFixed(3)}
            </span>
        </div>
    );
}

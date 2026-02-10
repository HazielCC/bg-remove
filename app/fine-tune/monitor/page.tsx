"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch, apiPost, createMetricsStream } from "../lib/api";

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

export default function MonitorPage() {
    const [status, setStatus] = useState<TrainingStatus | null>(null);
    const [metrics, setMetrics] = useState<EpochMetric[]>([]);
    const [logs, setLogs] = useState<string[]>([]);
    const [connected, setConnected] = useState(false);
    const sourceRef = useRef<EventSource | null>(null);
    const logsEndRef = useRef<HTMLDivElement>(null);

    // Fetch initial status
    useEffect(() => {
        apiFetch<TrainingStatus>("/training/status").then(setStatus).catch(console.error);
    }, []);

    // Connect to SSE
    const connect = useCallback(() => {
        if (sourceRef.current) {
            sourceRef.current.close();
        }

        const source = createMetricsStream(
            (data) => {
                const type = data.type as string;

                if (type === "epoch_end") {
                    const m = data as unknown as EpochMetric;
                    setMetrics((prev) => [...prev, m]);
                    setLogs((prev) => [
                        ...prev,
                        `[Epoch ${m.epoch}] loss=${(m.total_loss ?? 0).toFixed(4)} val=${(m.val_loss ?? 0).toFixed(4)} lr=${(m.lr ?? 0).toFixed(6)}`,
                    ]);
                    // Update status
                    setStatus((prev) =>
                        prev
                            ? {
                                ...prev,
                                epoch: m.epoch,
                                total_loss: m.total_loss ?? 0,
                                semantic_loss: m.semantic_loss ?? 0,
                                detail_loss: m.detail_loss ?? 0,
                                matte_loss: m.matte_loss ?? 0,
                                val_loss: m.val_loss ?? 0,
                                lr: m.lr ?? 0,
                                elapsed_seconds: m.elapsed_seconds ?? 0,
                                eta_seconds: m.eta_seconds ?? 0,
                                status: "running",
                            }
                            : prev
                    );
                } else if (type === "heartbeat") {
                    // just keep alive
                } else if (type === "finished") {
                    setLogs((prev) => [...prev, `Training finished! Best val loss: ${data.best_val_loss}`]);
                    setStatus((prev) => (prev ? { ...prev, status: "finished" } : prev));
                } else if (type === "error") {
                    setLogs((prev) => [...prev, `ERROR: ${data.message}`]);
                    setStatus((prev) => (prev ? { ...prev, status: "error" } : prev));
                } else if (type === "stopped") {
                    setLogs((prev) => [...prev, "Training stopped by user"]);
                    setStatus((prev) => (prev ? { ...prev, status: "stopped" } : prev));
                }
            },
            () => {
                setConnected(false);
            }
        );

        sourceRef.current = source;
        setConnected(true);

        return () => {
            source.close();
            setConnected(false);
        };
    }, []);

    // Auto-scroll logs
    useEffect(() => {
        logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [logs]);

    // Stop training
    const handleStop = async () => {
        try {
            await apiPost("/training/stop", {});
            setLogs((prev) => [...prev, "Stop requested..."]);
        } catch (e) {
            console.error(e);
        }
    };

    const formatTime = (seconds: number) => {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}m ${s}s`;
    };

    const maxLoss = metrics.length
        ? Math.max(...metrics.map((m) => m.total_loss || 0)) * 1.1
        : 1;

    return (
        <div className="p-6 max-w-6xl">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h1 className="text-2xl font-bold">Training Monitor</h1>
                    <p className="text-sm text-neutral-500">
                        Real-time training metrics via SSE
                    </p>
                </div>
                <div className="flex gap-2">
                    <button
                        onClick={connect}
                        disabled={connected}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm disabled:opacity-50"
                    >
                        {connected ? "Connected" : "Connect"}
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
                    />
                    <StatusCard
                        label="Val Loss"
                        value={status.val_loss.toFixed(4)}
                    />
                    <StatusCard
                        label="ETA"
                        value={formatTime(status.eta_seconds)}
                    />
                </div>
            )}

            <div className="grid grid-cols-2 gap-6">
                {/* ── Loss Chart (simple bar chart) ────────── */}
                <div className="border rounded-lg p-4 dark:border-neutral-700">
                    <h3 className="text-sm font-semibold mb-3">Loss per Epoch</h3>
                    {metrics.length === 0 ? (
                        <div className="h-48 flex items-center justify-center text-sm text-neutral-400">
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
                    <div className="flex justify-between text-[10px] text-neutral-400 mt-1">
                        <span>Epoch 1</span>
                        <span>Epoch {metrics.length || "?"}</span>
                    </div>
                </div>

                {/* ── Loss Breakdown ─────────────────────────── */}
                <div className="border rounded-lg p-4 dark:border-neutral-700">
                    <h3 className="text-sm font-semibold mb-3">Loss Breakdown</h3>
                    {metrics.length === 0 ? (
                        <div className="h-48 flex items-center justify-center text-sm text-neutral-400">
                            Waiting for data...
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
                    <StatusCard label="Learning Rate" value={status.lr.toExponential(2)} />
                    <StatusCard label="Best Val Loss" value={status.best_val_loss != null && status.best_val_loss !== Infinity ? status.best_val_loss.toFixed(4) : "—"} />
                    <StatusCard label="Elapsed" value={formatTime(status.elapsed_seconds)} />
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
                            Click &ldquo;Connect&rdquo; to start receiving metrics...
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
}: {
    label: string;
    value: string;
    color?: string;
}) {
    return (
        <div className="bg-neutral-50 dark:bg-neutral-800 border dark:border-neutral-700 rounded-lg p-3 text-center">
            <div className={`text-lg font-bold ${color || ""}`}>{value}</div>
            <div className="text-[10px] text-neutral-500 mt-0.5">{label}</div>
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

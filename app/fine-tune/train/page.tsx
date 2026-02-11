"use client";

import { useEffect, useState } from "react";
import { apiFetch, apiPost } from "../lib/api";

interface DatasetInfo {
    id: string;
    name: string;
    images_count: number;
    alphas_count: number;
    curated: boolean;
}

interface Checkpoint {
    run: string;
    name: string;
    path: string;
    size_mb: number;
}

export default function TrainPage() {
    const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
    const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
    const [starting, setStarting] = useState(false);
    const [result, setResult] = useState<string | null>(null);

    // Form state
    const [form, setForm] = useState({
        dataset_id: "",
        stage: "supervised",
        epochs: 40,
        lr: 0.01,
        batch_size: 4,
        img_size: 512,
        pretrained_ckpt: "",
        run_name: `run_${Date.now()}`,
        semantic_loss_weight: 10.0,
        detail_loss_weight: 10.0,
        matte_loss_weight: 1.0,
        soc_lr: 0.00001,
        soc_epochs: 10,
        val_split: 0.1,
        save_every: 5,
        backgrounds_dir: "",
    });

    useEffect(() => {
        apiFetch<DatasetInfo[]>("/datasets/local").then(setDatasets).catch(console.error);
        apiFetch<Checkpoint[]>("/training/checkpoints").then(setCheckpoints).catch(console.error);
    }, []);

    const update = (key: string, value: string | number) => {
        setForm((prev) => ({ ...prev, [key]: value }));
    };

    const handleStart = async () => {
        if (!form.dataset_id) {
            alert("Select a dataset first");
            return;
        }
        setStarting(true);
        setResult(null);
        try {
            const res = await apiPost<{ status: string; run_name: string }>(
                "/training/start",
                {
                    ...form,
                    pretrained_ckpt: form.pretrained_ckpt || null,
                    backgrounds_dir: form.backgrounds_dir || null,
                }
            );
            setResult(`Training started: ${res.run_name} (${res.status})`);
        } catch (e) {
            setResult("Error: " + (e as Error).message);
        } finally {
            setStarting(false);
        }
    };

    const isSoc = form.stage === "soc";

    return (
        <div className="p-6 max-w-3xl">
            <h1 className="text-2xl font-bold mb-1">Configuración de Entrenamiento</h1>
            <p className="text-secondary mb-6">
                Configura y lanza el fine-tuning de MODNet
            </p>

            <div className="space-y-6">
                {/* ── Dataset ──────────────────────────────────── */}
                <Section title="Dataset">
                    <label className="block text-sm font-medium mb-1">Dataset</label>
                    <select
                        value={form.dataset_id}
                        onChange={(e) => update("dataset_id", e.target.value)}
                        className="w-full border rounded-lg px-3 py-2 text-sm dark:bg-neutral-900 dark:border-neutral-700"
                    >
                        <option value="">-- Selecciona un dataset local --</option>
                        {datasets.map((ds) => (
                            <option key={ds.id} value={ds.id}>
                                {ds.name} ({ds.images_count} images)
                            </option>
                        ))}
                    </select>
                </Section>

                {/* ── Stage ────────────────────────────────────── */}
                <Section title="Etapa de Entrenamiento">
                    <div className="flex gap-3">
                        {[
                            {
                                val: "supervised",
                                label: "Supervisado",
                                desc: "Requiere imagen + alpha GT. Usa pérdidas basadas en trimap.",
                            },
                            {
                                val: "soc",
                                label: "SOC Adaptation",
                                desc: "Auto-supervisado. No requiere etiquetas. Usa consistencia entre ramas.",
                            },
                        ].map((opt) => (
                            <label
                                key={opt.val}
                                className={`flex-1 border rounded-lg p-3 cursor-pointer transition-colors ${form.stage === opt.val
                                    ? "border-accent bg-success dark:bg-blue-900/20"
                                    : "dark:border-neutral-700 hover:bg-secondary dark:hover:bg-neutral-800"
                                    }`}
                            >
                                <input
                                    type="radio"
                                    name="stage"
                                    value={opt.val}
                                    checked={form.stage === opt.val}
                                    onChange={(e) => update("stage", e.target.value)}
                                    className="sr-only"
                                />
                                <div className="font-medium text-sm">{opt.label}</div>
                                <div className="text-xs text-secondary mt-0.5">
                                    {opt.desc}
                                </div>
                            </label>
                        ))}
                    </div>
                </Section>

                {/* ── Hyperparameters ─────────────────────────── */}
                <Section title="Hiperparámetros">
                    <div className="grid grid-cols-2 gap-4">
                        <Field
                            label={isSoc ? "SOC Epochs" : "Epochs"}
                            type="number"
                            value={isSoc ? form.soc_epochs : form.epochs}
                            onChange={(v) => update(isSoc ? "soc_epochs" : "epochs", Number(v))}
                        />
                        <Field
                            label="Learning Rate"
                            type="number"
                            value={isSoc ? form.soc_lr : form.lr}
                            onChange={(v) => update(isSoc ? "soc_lr" : "lr", Number(v))}
                            step="0.0001"
                        />
                        <Field
                            label="Batch Size"
                            type="number"
                            value={form.batch_size}
                            onChange={(v) => update("batch_size", Number(v))}
                            hint="Use 2-4 for MPS"
                        />
                        <Field
                            label="Image Size"
                            type="number"
                            value={form.img_size}
                            onChange={(v) => update("img_size", Number(v))}
                            hint="512 recommended"
                        />
                        <Field
                            label="Validation Split"
                            type="number"
                            value={form.val_split}
                            onChange={(v) => update("val_split", Number(v))}
                            step="0.05"
                            hint="0.1 = 10%"
                        />
                        <Field
                            label="Save Every N Epochs"
                            type="number"
                            value={form.save_every}
                            onChange={(v) => update("save_every", Number(v))}
                        />
                    </div>
                </Section>

                {/* ── Loss Weights (supervised only) ──────────── */}
                {!isSoc && (
                    <Section title="Pesos de Pérdida">
                        <div className="grid grid-cols-3 gap-4">
                            <Field
                                label="Semantic"
                                type="number"
                                value={form.semantic_loss_weight}
                                onChange={(v) => update("semantic_loss_weight", Number(v))}
                            />
                            <Field
                                label="Detail"
                                type="number"
                                value={form.detail_loss_weight}
                                onChange={(v) => update("detail_loss_weight", Number(v))}
                            />
                            <Field
                                label="Matte"
                                type="number"
                                value={form.matte_loss_weight}
                                onChange={(v) => update("matte_loss_weight", Number(v))}
                            />
                        </div>
                    </Section>
                )}

                {/* ── Model ────────────────────────────────────── */}
                <Section title="Modelo Base">
                    <label className="block text-sm font-medium mb-1">
                        Pretrained Checkpoint (optional)
                    </label>
                    <select
                        value={form.pretrained_ckpt}
                        onChange={(e) => update("pretrained_ckpt", e.target.value)}
                        className="w-full border rounded-lg px-3 py-2 text-sm dark:bg-neutral-900 dark:border-neutral-700"
                    >
                        <option value="">
                            Ninguno (usar backbone ImageNet)
                        </option>
                        {checkpoints.map((ck) => (
                            <option key={ck.path} value={ck.path}>
                                {ck.run}/{ck.name} ({ck.size_mb} MB)
                            </option>
                        ))}
                    </select>
                </Section>

                {/* ── Run Name ─────────────────────────────────── */}
                <Section title="Run">
                    <Field
                        label="Nombre de Ejecución"
                        value={form.run_name}
                        onChange={(v) => update("run_name", v)}
                        hint="Identificador único para esta ejecución de entrenamiento"
                    />
                </Section>

                {/* ── Start Button ─────────────────────────────── */}
                <div className="flex items-center gap-4 pt-2">
                    <button
                        onClick={handleStart}
                        disabled={starting || !form.dataset_id}
                        className="bg-primary text-white px-6 py-2.5 rounded-lg font-medium disabled:opacity-50 hover:bg-primary transition-colors"
                    >
                        {starting ? "Iniciando..." : "Iniciar Entrenamiento"}
                    </button>

                    {result && (
                        <span
                            className={`text-sm ${result.startsWith("Error")
                                ? "text-error"
                                : "text-success"
                                }`}
                        >
                            {result}
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
}

// ── Reusable components ─────────────────────────────────
function Section({
    title,
    children,
}: {
    title: string;
    children: React.ReactNode;
}) {
    return (
        <div className="border rounded-lg p-4 dark:border-neutral-700">
            <h3 className="text-sm font-semibold text-neutral-700 dark:text-neutral-300 mb-3">
                {title}
            </h3>
            {children}
        </div>
    );
}

function Field({
    label,
    value,
    onChange,
    type = "text",
    hint,
    step,
}: {
    label: string;
    value: string | number;
    onChange: (v: string) => void;
    type?: string;
    hint?: string;
    step?: string;
}) {
    return (
        <div>
            <label className="block text-xs font-medium text-neutral-600 dark:text-neutral-400 mb-1">
                {label}
            </label>
            <input
                type={type}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                step={step}
                className="w-full border rounded-lg px-3 py-1.5 text-sm dark:bg-neutral-900 dark:border-neutral-700"
            />
            {hint && (
                <p className="text-[10px] text-muted dark:text-neutral-400 mt-0.5">{hint}</p>
            )}
        </div>
    );
}

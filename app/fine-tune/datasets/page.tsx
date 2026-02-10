"use client";

import { useCallback, useEffect, useState } from "react";
import { apiDelete, apiFetch, apiPost } from "../lib/api";

// ── Types ───────────────────────────────────────────────
interface DatasetInfo {
    id: string;
    name: string;
    path: string;
    images_count: number;
    alphas_count: number;
    curated: boolean;
    metadata: Record<string, unknown>;
}

interface HFDataset {
    id: string;
    author: string;
    downloads: number;
    likes: number;
    tags: string[];
}

interface SuggestedDataset {
    id: string;
    description: string;
    downloads: number;
    format: string;
    recommended: boolean;
}

interface ValidationResult {
    valid: boolean;
    file_count: number;
    data_format: string | null;
    message: string;
}

interface Sample {
    filename: string;
    image: string;
    alpha: string | null;
    width: number;
    height: number;
}

interface CurateResult {
    total: number;
    valid: number;
    too_small: number;
    broken_alpha: number;
    no_alpha: number;
    all_white_alpha: number;
    all_black_alpha: number;
    issues: { file: string; issue: string; detail?: string }[];
}

interface DatasetStats {
    total_images: number;
    avg_width: number;
    avg_height: number;
    min_width: number;
    min_height: number;
    max_width: number;
    max_height: number;
    avg_file_size_kb: number;
    total_size_mb: number;
}

// ── Component ───────────────────────────────────────────
export default function DatasetsPage() {
    const [tab, setTab] = useState<"local" | "search">("local");
    const [localDatasets, setLocalDatasets] = useState<DatasetInfo[]>([]);
    const [searchQuery, setSearchQuery] = useState("matting human");
    const [searchResults, setSearchResults] = useState<HFDataset[]>([]);
    const [suggestedDatasets, setSuggestedDatasets] = useState<SuggestedDataset[]>([]);
    const [loading, setLoading] = useState(false);
    const [downloading, setDownloading] = useState<string | null>(null);
    const [downloadMax, setDownloadMax] = useState<number>(500);
    const [downloadError, setDownloadError] = useState<string | null>(null);
    const [validating, setValidating] = useState<string | null>(null);

    // Preview / stats
    const [previewId, setPreviewId] = useState<string | null>(null);
    const [previewSamples, setPreviewSamples] = useState<Sample[]>([]);
    const [stats, setStats] = useState<DatasetStats | null>(null);
    const [curateResult, setCurateResult] = useState<CurateResult | null>(null);

    const fetchLocal = useCallback(async () => {
        try {
            const data = await apiFetch<DatasetInfo[]>("/datasets/local");
            setLocalDatasets(data);
        } catch (e) {
            console.error("Failed to fetch local datasets:", e);
        }
    }, []);

    useEffect(() => {
        fetchLocal();
        // Fetch suggested datasets
        apiFetch<SuggestedDataset[]>("/datasets/suggested")
            .then(setSuggestedDatasets)
            .catch(console.error);
    }, [fetchLocal]);

    // ── Search HF ─────────────────────────────────────────
    const handleSearch = async () => {
        setLoading(true);
        try {
            const results = await apiFetch<HFDataset[]>(
                `/datasets/search?q=${encodeURIComponent(searchQuery)}`
            );
            setSearchResults(results);
        } catch (e) {
            alert("Search error: " + (e as Error).message);
        } finally {
            setLoading(false);
        }
    };

    // ── Download ──────────────────────────────────────────
    const handleDownload = async (datasetName: string) => {
        setDownloadError(null);
        // Step 1: validate
        setValidating(datasetName);
        try {
            const check = await apiPost<ValidationResult>("/datasets/validate", {
                dataset_name: datasetName,
            });
            if (!check.valid) {
                setDownloadError(check.message);
                setValidating(null);
                return;
            }
        } catch {
            // If validation endpoint not available, proceed anyway
        }
        setValidating(null);

        // Step 2: download
        setDownloading(datasetName);
        try {
            const result = await apiPost<{ status: string; images_count?: number; alphas_count?: number; detail?: string }>(
                "/datasets/download",
                {
                    dataset_name: datasetName,
                    split: "train",
                    max_samples: downloadMax || null,
                }
            );
            if (result.status === "already_exists") {
                setDownloadError("El dataset ya existe localmente. Elimínalo primero para re-descargarlo.");
            } else {
                setDownloadError(null);
                alert(`Descargadas ${result.images_count ?? 0} imágenes y ${result.alphas_count ?? 0} alphas.`);
            }
            fetchLocal();
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : String(e);
            setDownloadError(msg);
        } finally {
            setDownloading(null);
        }
    };

    // ── Preview ───────────────────────────────────────────
    const handlePreview = async (id: string) => {
        setPreviewId(id);
        setCurateResult(null);
        setStats(null);
        try {
            const [samples, st] = await Promise.all([
                apiFetch<Sample[]>(`/datasets/${id}/preview?n=12`),
                apiFetch<DatasetStats>(`/datasets/${id}/stats`),
            ]);
            setPreviewSamples(samples);
            setStats(st);
        } catch (e) {
            console.error(e);
        }
    };

    // ── Curate ────────────────────────────────────────────
    const handleCurate = async (id: string) => {
        setLoading(true);
        try {
            const result = await apiPost<CurateResult>(`/datasets/${id}/curate`, {
                min_resolution: 256,
                check_alpha_range: true,
            });
            setCurateResult(result);
            fetchLocal();
        } catch (e) {
            alert("Curate error: " + (e as Error).message);
        } finally {
            setLoading(false);
        }
    };

    // ── Delete ────────────────────────────────────────────
    const handleDelete = async (id: string) => {
        if (!confirm(`Delete dataset "${id}"?`)) return;
        try {
            await apiDelete(`/datasets/${id}`);
            fetchLocal();
            if (previewId === id) {
                setPreviewId(null);
                setPreviewSamples([]);
            }
        } catch (e) {
            alert("Delete error: " + (e as Error).message);
        }
    };

    return (
        <div className="p-6 max-w-6xl">
            <h1 className="text-2xl font-bold mb-1">Dataset Manager</h1>
            <p className="text-sm text-neutral-500 mb-6">
                Search, download and curate matting datasets from HuggingFace
            </p>

            {/* Tabs */}
            <div className="flex gap-2 mb-6">
                <button
                    onClick={() => setTab("local")}
                    className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${tab === "local"
                        ? "bg-blue-600 text-white"
                        : "bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400"
                        }`}
                >
                    Datasets Locales ({localDatasets.length})
                </button>
                <button
                    onClick={() => setTab("search")}
                    className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${tab === "search"
                        ? "bg-blue-600 text-white"
                        : "bg-neutral-100 dark:bg-neutral-800 text-neutral-600 dark:text-neutral-400"
                        }`}
                >
                    Buscar en HuggingFace
                </button>
            </div>

            {/* ── Search Tab ───────────────────────────────── */}
            {tab === "search" && (
                <div className="space-y-4">
                    {/* Suggested datasets */}
                    {suggestedDatasets.length > 0 && (
                        <div className="bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
                            <h3 className="text-sm font-medium text-blue-800 dark:text-blue-300 mb-2">
                                📦 Datasets recomendados (verificados)
                            </h3>
                            <div className="space-y-2">
                                {suggestedDatasets.map((ds) => (
                                    <div
                                        key={ds.id}
                                        className="flex items-center justify-between bg-white dark:bg-neutral-900 rounded-lg px-3 py-2 border border-blue-100 dark:border-blue-900"
                                    >
                                        <div className="flex-1">
                                            <div className="flex items-center gap-2">
                                                <a
                                                    href={`https://huggingface.co/datasets/${ds.id}`}
                                                    target="_blank"
                                                    rel="noopener"
                                                    className="text-sm font-medium text-blue-600 hover:underline"
                                                >
                                                    {ds.id}
                                                </a>
                                                {ds.recommended && (
                                                    <span className="text-[10px] px-1.5 py-0.5 bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300 rounded-full">
                                                        Recomendado
                                                    </span>
                                                )}
                                                <span className="text-[10px] px-1.5 py-0.5 bg-neutral-100 dark:bg-neutral-800 rounded">
                                                    {ds.format}
                                                </span>
                                            </div>
                                            <p className="text-xs text-neutral-500 mt-0.5">
                                                {ds.description}
                                            </p>
                                        </div>
                                        <button
                                            onClick={() => handleDownload(ds.id)}
                                            disabled={downloading === ds.id || validating === ds.id}
                                            className="bg-green-600 text-white px-3 py-1 rounded text-xs disabled:opacity-50 ml-3"
                                        >
                                            {validating === ds.id
                                                ? "Validando..."
                                                : downloading === ds.id
                                                    ? "Descargando..."
                                                    : "Descargar"}
                                        </button>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Error banner */}
                    {downloadError && (
                        <div className="bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-lg p-3 flex items-start gap-2">
                            <span className="text-red-500 text-lg leading-none">⚠</span>
                            <div className="flex-1">
                                <p className="text-sm text-red-700 dark:text-red-300">{downloadError}</p>
                            </div>
                            <button
                                onClick={() => setDownloadError(null)}
                                className="text-red-400 hover:text-red-600 text-xs"
                            >
                                ✕
                            </button>
                        </div>
                    )}

                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                            placeholder="Search datasets (e.g., matting human portrait)..."
                            className="flex-1 border rounded-lg px-3 py-2 text-sm dark:bg-neutral-900 dark:border-neutral-700"
                        />
                        <div className="flex items-center gap-1">
                            <label className="text-xs text-neutral-500">Max:</label>
                            <input
                                type="number"
                                value={downloadMax}
                                onChange={(e) => setDownloadMax(Number(e.target.value))}
                                className="w-20 border rounded-lg px-2 py-2 text-sm dark:bg-neutral-900 dark:border-neutral-700"
                                min={10}
                                step={100}
                            />
                        </div>
                        <button
                            onClick={handleSearch}
                            disabled={loading}
                            className="bg-blue-600 text-white px-4 py-2 rounded-lg text-sm disabled:opacity-50"
                        >
                            {loading ? "Searching..." : "Search"}
                        </button>
                    </div>

                    {searchResults.length > 0 && (
                        <div className="border rounded-lg overflow-hidden dark:border-neutral-700">
                            <table className="w-full text-sm">
                                <thead className="bg-neutral-50 dark:bg-neutral-800">
                                    <tr>
                                        <th className="text-left p-3">Dataset</th>
                                        <th className="text-right p-3">Downloads</th>
                                        <th className="text-right p-3">Likes</th>
                                        <th className="text-right p-3">Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {searchResults.map((ds) => (
                                        <tr
                                            key={ds.id}
                                            className="border-t dark:border-neutral-700"
                                        >
                                            <td className="p-3">
                                                <a
                                                    href={`https://huggingface.co/datasets/${ds.id}`}
                                                    target="_blank"
                                                    rel="noopener"
                                                    className="text-blue-600 hover:underline"
                                                >
                                                    {ds.id}
                                                </a>
                                                <div className="flex gap-1 mt-1 flex-wrap">
                                                    {ds.tags.slice(0, 5).map((t) => (
                                                        <span
                                                            key={t}
                                                            className="text-[10px] px-1.5 py-0.5 bg-neutral-100 dark:bg-neutral-800 rounded"
                                                        >
                                                            {t}
                                                        </span>
                                                    ))}
                                                </div>
                                            </td>
                                            <td className="p-3 text-right text-neutral-500">
                                                {ds.downloads.toLocaleString()}
                                            </td>
                                            <td className="p-3 text-right text-neutral-500">
                                                {ds.likes}
                                            </td>
                                            <td className="p-3 text-right">
                                                <button
                                                    onClick={() => handleDownload(ds.id)}
                                                    disabled={downloading === ds.id || validating === ds.id}
                                                    className="bg-green-600 text-white px-3 py-1 rounded text-xs disabled:opacity-50"
                                                >
                                                    {validating === ds.id
                                                        ? "Validando..."
                                                        : downloading === ds.id
                                                            ? "Descargando..."
                                                            : "Descargar"}
                                                </button>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            )}

            {/* ── Local Datasets Tab ───────────────────────── */}
            {tab === "local" && (
                <div className="space-y-4">
                    {localDatasets.length === 0 ? (
                        <div className="text-center py-12 text-neutral-400">
                            <p className="text-lg mb-2">No hay datasets descargados</p>
                            <p className="text-sm">
                                Ve a &quot;Buscar en HuggingFace&quot; para buscar y descargar datasets.
                                <br />
                                Recomendamos empezar con <strong>Voxel51/DUTS</strong>.
                            </p>
                        </div>
                    ) : (
                        <div className="grid gap-3">
                            {localDatasets.map((ds) => (
                                <div
                                    key={ds.id}
                                    className={`border rounded-lg p-4 transition-colors dark:border-neutral-700 ${previewId === ds.id
                                        ? "border-blue-400 dark:border-blue-600"
                                        : ""
                                        }`}
                                >
                                    <div className="flex items-start justify-between">
                                        <div>
                                            <h3 className="font-medium">{ds.name}</h3>
                                            <p className="text-xs text-neutral-500 mt-0.5">
                                                {ds.images_count} images · {ds.alphas_count} alphas
                                                {ds.curated && (
                                                    <span className="ml-2 text-green-600">
                                                        ✓ Curated
                                                    </span>
                                                )}
                                            </p>
                                        </div>
                                        <div className="flex gap-2">
                                            <button
                                                onClick={() => handlePreview(ds.id)}
                                                className="text-xs px-3 py-1 border rounded hover:bg-neutral-50 dark:hover:bg-neutral-800 dark:border-neutral-600"
                                            >
                                                Preview
                                            </button>
                                            <button
                                                onClick={() => handleCurate(ds.id)}
                                                disabled={loading}
                                                className="text-xs px-3 py-1 border rounded hover:bg-neutral-50 dark:hover:bg-neutral-800 dark:border-neutral-600"
                                            >
                                                Curate
                                            </button>
                                            <button
                                                onClick={() => handleDelete(ds.id)}
                                                className="text-xs px-3 py-1 border border-red-300 text-red-600 rounded hover:bg-red-50 dark:border-red-800 dark:hover:bg-red-900/20"
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* ── Preview Panel ──────────────────────────── */}
                    {previewId && (
                        <div className="mt-6 border rounded-lg p-4 dark:border-neutral-700">
                            <h3 className="font-medium mb-3">
                                Preview: {previewId.replace("__", "/")}
                            </h3>

                            {/* Stats */}
                            {stats && (
                                <div className="grid grid-cols-4 gap-3 mb-4">
                                    {[
                                        { label: "Total", value: stats.total_images },
                                        {
                                            label: "Avg Size",
                                            value: `${stats.avg_width}×${stats.avg_height}`,
                                        },
                                        {
                                            label: "Total Size",
                                            value: `${stats.total_size_mb} MB`,
                                        },
                                        {
                                            label: "Avg File",
                                            value: `${stats.avg_file_size_kb} KB`,
                                        },
                                    ].map((s) => (
                                        <div
                                            key={s.label}
                                            className="bg-neutral-50 dark:bg-neutral-800 p-3 rounded-lg text-center"
                                        >
                                            <div className="text-lg font-bold">{s.value}</div>
                                            <div className="text-xs text-neutral-500">{s.label}</div>
                                        </div>
                                    ))}
                                </div>
                            )}

                            {/* Sample Grid */}
                            <div className="grid grid-cols-4 gap-3">
                                {previewSamples.map((s) => (
                                    <div
                                        key={s.filename}
                                        className="border rounded-lg overflow-hidden dark:border-neutral-700"
                                    >
                                        <div className="grid grid-cols-2">
                                            <img
                                                src={s.image}
                                                alt={s.filename}
                                                className="w-full aspect-square object-cover"
                                            />
                                            {s.alpha ? (
                                                <img
                                                    src={s.alpha}
                                                    alt="alpha"
                                                    className="w-full aspect-square object-cover bg-black"
                                                />
                                            ) : (
                                                <div className="w-full aspect-square bg-neutral-200 dark:bg-neutral-800 flex items-center justify-center text-xs text-neutral-400">
                                                    No alpha
                                                </div>
                                            )}
                                        </div>
                                        <div className="p-1.5 text-[10px] text-neutral-500 truncate">
                                            {s.filename}
                                        </div>
                                    </div>
                                ))}
                            </div>

                            {/* Curate Results */}
                            {curateResult && (
                                <div className="mt-4 bg-neutral-50 dark:bg-neutral-800 rounded-lg p-4">
                                    <h4 className="font-medium mb-2">Curation Report</h4>
                                    <div className="grid grid-cols-3 gap-2 text-sm">
                                        <div>
                                            <span className="text-green-600 font-bold">
                                                {curateResult.valid}
                                            </span>{" "}
                                            valid
                                        </div>
                                        <div>
                                            <span className="text-yellow-600 font-bold">
                                                {curateResult.too_small}
                                            </span>{" "}
                                            too small
                                        </div>
                                        <div>
                                            <span className="text-red-600 font-bold">
                                                {curateResult.no_alpha}
                                            </span>{" "}
                                            no alpha
                                        </div>
                                        <div>
                                            <span className="text-red-600 font-bold">
                                                {curateResult.broken_alpha}
                                            </span>{" "}
                                            broken
                                        </div>
                                        <div>
                                            <span className="text-yellow-600 font-bold">
                                                {curateResult.all_black_alpha}
                                            </span>{" "}
                                            all black
                                        </div>
                                        <div>
                                            <span className="text-yellow-600 font-bold">
                                                {curateResult.all_white_alpha}
                                            </span>{" "}
                                            all white
                                        </div>
                                    </div>
                                    {curateResult.issues.length > 0 && (
                                        <details className="mt-3">
                                            <summary className="text-xs cursor-pointer text-neutral-500">
                                                {curateResult.issues.length} issues (click to expand)
                                            </summary>
                                            <ul className="mt-2 space-y-1 max-h-40 overflow-y-auto text-xs">
                                                {curateResult.issues.map((iss, i) => (
                                                    <li key={i} className="text-neutral-600 dark:text-neutral-400">
                                                        <span className="font-mono">{iss.file}</span> —{" "}
                                                        {iss.issue}
                                                        {iss.detail && ` (${iss.detail})`}
                                                    </li>
                                                ))}
                                            </ul>
                                        </details>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

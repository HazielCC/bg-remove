/**
 * Shared utilities for communicating with the fine-tuning backend (FastAPI).
 *
 * The Next.js rewrites proxy /api/* → http://localhost:8000/api/*
 * so we can call /api/... directly without CORS issues.
 */

const API_BASE = "/api";

export async function apiFetch<T = unknown>(
    path: string,
    options?: RequestInit
): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`, {
        headers: { "Content-Type": "application/json", ...options?.headers },
        ...options,
    });
    if (!res.ok) {
        const body = await res.text();
        throw new Error(`API ${res.status}: ${body}`);
    }
    return res.json();
}

export async function apiPost<T = unknown>(
    path: string,
    body: unknown
): Promise<T> {
    return apiFetch<T>(path, {
        method: "POST",
        body: JSON.stringify(body),
    });
}

export async function apiDelete<T = unknown>(path: string): Promise<T> {
    return apiFetch<T>(path, { method: "DELETE" });
}

/**
 * Create an EventSource (SSE) connection for training metrics.
 */
export function createMetricsStream(
    onMessage: (data: Record<string, unknown>) => void,
    onError?: (err: Event) => void
): EventSource {
    const source = new EventSource(`${API_BASE}/training/stream`);
    source.onmessage = (e) => {
        try {
            const data = JSON.parse(e.data);
            onMessage(data);
        } catch {
            // skip
        }
    };
    if (onError) {
        source.onerror = onError;
    }
    return source;
}

/**
 * Upload a file via FormData to an API endpoint.
 */
export async function apiUpload<T = unknown>(
    path: string,
    formData: FormData
): Promise<T> {
    const res = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        body: formData,
    });
    if (!res.ok) {
        const body = await res.text();
        throw new Error(`Upload ${res.status}: ${body}`);
    }
    return res.json();
}

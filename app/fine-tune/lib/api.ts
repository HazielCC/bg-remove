/**
 * Shared utilities for communicating with the fine-tuning backend (FastAPI).
 *
 * The Next.js rewrites proxy /api/* → http://localhost:8000/api/*
 * so we can call /api/... directly without CORS issues.
 */

const API_BASE = "/api";

const RETRYABLE_PROXY_TOKENS = ["econnrefused", "failed to proxy"];

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryableProxyError(status: number, body: string): boolean {
    if (status === 502 || status === 503 || status === 504) {
        return true;
    }
    if (status !== 500) {
        return false;
    }
    const normalized = body.toLowerCase();
    return RETRYABLE_PROXY_TOKENS.some((token) => normalized.includes(token));
}

export async function apiFetch<T = unknown>(
    path: string,
    options?: RequestInit & {
        retries?: number;
        retryDelayMs?: number;
    }
): Promise<T> {
    const method = (options?.method ?? "GET").toUpperCase();
    const retries =
        options?.retries ?? (method === "GET" ? 3 : 0);
    const retryDelayMs = options?.retryDelayMs ?? 500;

    const fetchOptions: RequestInit = { ...(options ?? {}) };
    delete (fetchOptions as Record<string, unknown>).retries;
    delete (fetchOptions as Record<string, unknown>).retryDelayMs;

    let lastError: unknown = null;

    for (let attempt = 0; attempt <= retries; attempt++) {
        let res: Response;
        try {
            res = await fetch(`${API_BASE}${path}`, {
                headers: { "Content-Type": "application/json", ...fetchOptions.headers },
                ...fetchOptions,
            });
        } catch (error) {
            lastError = error;
            if (attempt < retries) {
                await sleep(retryDelayMs * (attempt + 1));
                continue;
            }
            throw error;
        }

        if (!res.ok) {
            const body = await res.text();
            if (attempt < retries && isRetryableProxyError(res.status, body)) {
                await sleep(retryDelayMs * (attempt + 1));
                continue;
            }
            throw new Error(`API ${res.status}: ${body}`);
        }

        return res.json();
    }

    throw lastError instanceof Error ? lastError : new Error("Unknown API error");
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

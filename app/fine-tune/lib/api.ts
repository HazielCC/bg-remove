/**
 * Shared utilities for communicating with the fine-tuning backend (FastAPI).
 *
 * The Next.js rewrites proxy /api/* → http://localhost:8000/api/*
 * so we can call /api/... directly without CORS issues.
 */

const API_BASE = "/api";

const RETRYABLE_PROXY_TOKENS = ["econnrefused", "failed to proxy"];
const GET_DEDUPE_TTL_MS = 800;
const inflightGetRequests = new Map<string, Promise<unknown>>();
const recentGetResponses = new Map<string, { expiresAt: number; data: unknown }>();

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

function normalizeHeaders(headers?: HeadersInit): string {
    if (!headers) {
        return "";
    }
    if (headers instanceof Headers) {
        return JSON.stringify(Array.from(headers.entries()).sort());
    }
    if (Array.isArray(headers)) {
        return JSON.stringify([...headers].sort((a, b) => a[0].localeCompare(b[0])));
    }
    return JSON.stringify(
        Object.entries(headers)
            .map(([k, v]) => [k.toLowerCase(), v])
            .sort((a, b) => a[0].localeCompare(b[0]))
    );
}

export async function apiFetch<T = unknown>(
    path: string,
    options?: RequestInit & {
        retries?: number;
        retryDelayMs?: number;
    }
): Promise<T> {
    const method = (options?.method ?? "GET").toUpperCase();
    const isGet = method === "GET";
    const retries =
        options?.retries ?? (method === "GET" ? 3 : 0);
    const retryDelayMs = options?.retryDelayMs ?? 500;

    const fetchOptions: RequestInit = { ...(options ?? {}) };
    delete (fetchOptions as Record<string, unknown>).retries;
    delete (fetchOptions as Record<string, unknown>).retryDelayMs;

    const requestKey = isGet
        ? `${path}|${normalizeHeaders(fetchOptions.headers)}`
        : "";

    // Invalidate short GET cache on state-changing requests to avoid stale UI.
    if (!isGet) {
        recentGetResponses.clear();
    }

    if (isGet) {
        const now = Date.now();
        const recent = recentGetResponses.get(requestKey);
        if (recent && recent.expiresAt > now) {
            return recent.data as T;
        }
        if (recent && recent.expiresAt <= now) {
            recentGetResponses.delete(requestKey);
        }

        const inflight = inflightGetRequests.get(requestKey);
        if (inflight) {
            return inflight as Promise<T>;
        }
    }

    let lastError: unknown = null;

    const requestPromise = (async () => {
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

            const data = await res.json();
            if (isGet) {
                recentGetResponses.set(requestKey, {
                    expiresAt: Date.now() + GET_DEDUPE_TTL_MS,
                    data,
                });
            }
            return data as T;
        }

        throw lastError instanceof Error ? lastError : new Error("Unknown API error");
    })();

    if (!isGet) {
        return requestPromise;
    }

    inflightGetRequests.set(requestKey, requestPromise as Promise<unknown>);
    try {
        return await requestPromise;
    } finally {
        inflightGetRequests.delete(requestKey);
    }
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

const requestedPort = new URLSearchParams(window.location.search).get('servicePort')
const servicePort = requestedPort && /^\d{1,5}$/.test(requestedPort) ? requestedPort : '18765'

export const API_BASE = `http://127.0.0.1:${servicePort}`
export const WS_URL = `ws://127.0.0.1:${servicePort}/ws/events`

export class ApiError extends Error {
  code: string
  details: Record<string, unknown>

  constructor(message: string, code = 'REQUEST_FAILED', details: Record<string, unknown> = {}) {
    super(message)
    this.code = code
    this.details = details
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) }
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const error = body.error ?? body.detail ?? {}
    throw new ApiError(error.message ?? (typeof error === 'string' ? error : `请求失败 (${response.status})`), error.code, error)
  }
  return body as T
}

export function post<T>(path: string, body?: unknown): Promise<T> {
  return api<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })
}

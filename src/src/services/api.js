/**
 * api.js — thin wrapper around the FastAPI backend.
 *
 * Base URL is configurable via VITE_API_URL env var (defaults to same origin,
 * which works with Vite's proxy for local dev).
 */

const BASE = import.meta.env.VITE_API_URL ?? ''

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json()
}

async function get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`${res.status}`)
  return res.json()
}

// ── Day 1 stub + Day 2 real ─────────────────────────────────
export async function submitTriage(payload) {
  return post('/triage', payload)
}

// ── Day 3 endpoints ──────────────────────────────────────────
export async function fetchQueue() {
  return get('/queue')
}

export async function fetchBeds() {
  return get('/beds')
}

export async function checkHealth() {
  return get('/health')
}

import { useState, useEffect, useCallback } from 'react'
import { fetchQueue, fetchBeds } from '../services/api.js'
import { esiMeta, mockQueueData, mockBedsData } from '../services/helpers.js'
import './Dashboard.css'

export default function Dashboard() {
  const [queue, setQueue] = useState([])
  const [beds, setBeds]   = useState([])
  const [selected, setSelected] = useState(null)
  const [loading, setLoading]   = useState(true)
  const [lastRefresh, setLastRefresh] = useState(null)

  const load = useCallback(async () => {
    try {
      const [q, b] = await Promise.all([fetchQueue(), fetchBeds()])
      setQueue(q)
      setBeds(b)
    } catch {
      // Fallback to mock data if backend not yet running (Day 1 / Day 2)
      setQueue(mockQueueData())
      setBeds(mockBedsData())
    }
    setLastRefresh(new Date())
    setLoading(false)
  }, [])

  useEffect(() => {
    load()
    const iv = setInterval(load, 30_000)
    return () => clearInterval(iv)
  }, [load])

  const criticalCount = queue.filter(p => p.esi_level <= 2).length
  const available = beds.filter(b => b.status === 'available').length

  return (
    <div className="dash">
      {/* ── Top status bar ── */}
      <div className="dash-statusbar">
        <div className="dash-stat">
          <span className="dash-stat-val">{queue.length}</span>
          <span className="dash-stat-lbl">Waiting</span>
        </div>
        <div className="dash-stat dash-stat--crit">
          <span className="dash-stat-val" style={{ color: 'var(--esi1)' }}>{criticalCount}</span>
          <span className="dash-stat-lbl">ESI 1–2</span>
        </div>
        <div className="dash-stat">
          <span className="dash-stat-val" style={{ color: 'var(--green)' }}>{available}</span>
          <span className="dash-stat-lbl">Beds free</span>
        </div>
        <div className="dash-stat">
          <span className="dash-stat-val">{beds.length}</span>
          <span className="dash-stat-lbl">Beds total</span>
        </div>
        <div className="dash-refresh">
          {lastRefresh && (
            <span className="dash-refresh-txt">
              Updated {lastRefresh.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
            </span>
          )}
          <button className="dash-refresh-btn" onClick={load} disabled={loading}>
            {loading ? '…' : '↻ Refresh'}
          </button>
        </div>
      </div>

      {/* ── Main 3-column layout ── */}
      <div className="dash-body">

        {/* Column 1 — Patient Queue */}
        <section className="dash-col dash-col--queue">
          <h2 className="dash-col-title">Patient Queue <span className="dash-col-badge">{queue.length}</span></h2>
          <div className="dash-queue-list">
            {loading && queue.length === 0 && (
              <div className="dash-loading">Loading patients…</div>
            )}
            {queue.map(patient => (
              <PatientRow
                key={patient.id}
                patient={patient}
                isSelected={selected?.id === patient.id}
                onClick={() => setSelected(selected?.id === patient.id ? null : patient)}
              />
            ))}
          </div>
        </section>

        {/* Column 2 — Bed Map */}
        <section className="dash-col dash-col--beds">
          <h2 className="dash-col-title">Bed Map</h2>
          <BedMap beds={beds} />
        </section>

        {/* Column 3 — Detail / SHAP panel */}
        <section className="dash-col dash-col--detail">
          {selected ? (
            <DetailPanel patient={selected} onClose={() => setSelected(null)} />
          ) : (
            <div className="dash-detail-empty">
              <div className="dash-empty-icon">⟵</div>
              <p>Select a patient to view<br/>SHAP risk breakdown</p>
            </div>
          )}
        </section>

      </div>
    </div>
  )
}

/* ── PatientRow ──────────────────────────────────────────── */
function PatientRow({ patient, isSelected, onClick }) {
  const { esi_level, name, age, gender, wait, confidence } = patient
  const meta = esiMeta(esi_level)
  const confPct = Math.round((confidence ?? 0) * 100)

  return (
    <div
      className={`prow${isSelected ? ' selected' : ''}`}
      onClick={onClick}
      style={{ '--row-color': meta.color }}
    >
      <div className="prow-esi" style={{ background: meta.bg, borderColor: meta.border, color: meta.color }}>
        {esi_level}
      </div>
      <div className="prow-info">
        <div className="prow-name">{name}</div>
        <div className="prow-meta">{age}{gender} · {wait}m wait</div>
      </div>
      <div className="prow-right">
        <div className="prow-conf">{confPct}%</div>
        <div className="prow-conf-track">
          <div
            className="prow-conf-fill"
            style={{ width: `${confPct}%`, background: meta.color }}
          />
        </div>
      </div>
    </div>
  )
}

/* ── BedMap ──────────────────────────────────────────────── */
function BedMap({ beds }) {
  const zones = [...new Set(beds.map(b => b.zone))]
  const statusColor = {
    available: 'var(--esi4)',
    occupied:  'var(--esi1)',
    pending:   'var(--esi3)',
    cleaning:  'var(--tx3)',
  }

  return (
    <div className="bedmap">
      <div className="bedmap-legend">
        {['available', 'occupied', 'pending'].map(s => (
          <div key={s} className="bedmap-legend-item">
            <span className="bedmap-legend-dot" style={{ background: statusColor[s] }} />
            {s}
          </div>
        ))}
      </div>

      {zones.map(zone => (
        <div key={zone} className="bedmap-zone">
          <div className="bedmap-zone-label">{zone}</div>
          <div className="bedmap-grid">
            {beds.filter(b => b.zone === zone).map(bed => (
              <div
                key={bed.id}
                className="bedmap-cell"
                title={`${bed.code} — ${bed.status}`}
                style={{
                  background: statusColor[bed.status] + '18',
                  borderColor: statusColor[bed.status] + '50',
                  color: statusColor[bed.status],
                }}
              >
                <span className="bedmap-cell-code">{bed.code}</span>
                <span className="bedmap-cell-status">{bed.status === 'available' ? '✓' : bed.status === 'occupied' ? '●' : '◐'}</span>
              </div>
            ))}
          </div>
        </div>
      ))}

      <div className="bedmap-summary">
        {beds.filter(b => b.status === 'available').length} available ·{' '}
        {beds.filter(b => b.status === 'occupied').length} occupied ·{' '}
        {beds.filter(b => b.status === 'pending').length} pending
      </div>
    </div>
  )
}

/* ── DetailPanel (SHAP) ──────────────────────────────────── */
function DetailPanel({ patient, onClose }) {
  const meta = esiMeta(patient.esi_level)
  const confPct = Math.round((patient.confidence ?? 0) * 100)
  const factors = patient.top_risk_factors ?? []
  const barWeights = [1, 0.72, 0.51]

  return (
    <div className="dp">
      <div className="dp-header">
        <div>
          <div className="dp-name">{patient.name}</div>
          <div className="dp-sub">{patient.age}{patient.gender} · {patient.wait}m in queue</div>
        </div>
        <button className="dp-close" onClick={onClose}>✕</button>
      </div>

      {/* ESI badge */}
      <div
        className="dp-esi-badge"
        style={{ background: meta.bg, borderColor: meta.border, color: meta.color }}
      >
        <span className="dp-esi-num">{patient.esi_level}</span>
        <div>
          <div className="dp-esi-short">{meta.short}</div>
          <div className="dp-esi-long">{meta.label}</div>
        </div>
      </div>

      {/* Confidence */}
      <div className="dp-section">
        <div className="dp-section-title">Confidence</div>
        <div className="dp-conf-row">
          <div className="dp-conf-track">
            <div
              className="dp-conf-fill"
              style={{
                width: `${confPct}%`,
                background: confPct >= 70 ? 'var(--green)' : 'var(--amber)',
              }}
            />
          </div>
          <span
            className="dp-conf-val"
            style={{ color: confPct >= 70 ? 'var(--green)' : 'var(--amber)' }}
          >
            {confPct}%
          </span>
        </div>
      </div>

      {/* SHAP factors */}
      {factors.length > 0 && (
        <div className="dp-section">
          <div className="dp-section-title">SHAP Risk Factors</div>
          <div className="dp-shap-list">
            {factors.map((f, i) => (
              <div key={i} className="dp-shap-item">
                <div className="dp-shap-header">
                  <span className="dp-shap-rank">#{i + 1}</span>
                  <span className="dp-shap-label">{f}</span>
                </div>
                <div className="dp-shap-bar-track">
                  <div
                    className="dp-shap-bar-fill"
                    style={{
                      width: `${Math.round(barWeights[i] * 100)}%`,
                      background: meta.color,
                      opacity: 0.8 - i * 0.15,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="dp-shap-note">
            SHAP values show each feature's contribution to the ESI prediction relative to baseline model output.
          </p>
        </div>
      )}

      {/* Recommended action */}
      <div className="dp-section">
        <div className="dp-section-title">Recommended Action</div>
        <div className="dp-action-box" style={{ borderColor: meta.border }}>
          <div className="dp-action-dot" style={{ background: meta.color }} />
          <p className="dp-action-text">{getGuidance(patient.esi_level)}</p>
        </div>
      </div>
    </div>
  )
}

function getGuidance(level) {
  const map = {
    1: 'Immediate resuscitation. Activate trauma/code team NOW.',
    2: 'Emergent — physician within 15 minutes. Continuous monitoring.',
    3: 'Urgent — reassess every 30 min. IV access recommended.',
    4: 'Less urgent — one resource needed. Can wait 60–120 minutes.',
    5: 'Non-urgent — routine care. Consider GP referral.',
  }
  return map[level] ?? '—'
}

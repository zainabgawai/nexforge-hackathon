import { esiMeta } from '../services/helpers.js'
import './ResultPanel.css'

export default function ResultPanel({ result, onClear }) {
  const { esi_level, confidence, top_risk_factors = [], model_version } = result
  const meta = esiMeta(esi_level)

  const confidencePct = Math.round((confidence ?? 0) * 100)
  const isHighConf = confidencePct >= 70

  // Normalise top_risk_factors — backend returns strings
  const shapFactors = (top_risk_factors ?? []).slice(0, 3)

  // Bar widths: first = 100%, scale others proportionally
  // Since we just have strings, give them mock weights
  const barWeights = [1, 0.72, 0.51]

  return (
    <div className="rp-wrap">
      <div className="rp-inner">

        {/* ESI badge */}
        <div className="rp-esi-ring" style={{ '--esi-color': meta.color, '--esi-bg': meta.bg, '--esi-border': meta.border }}>
          <div className="rp-esi-circle">
            <span className="rp-esi-num">{esi_level}</span>
            <span className="rp-esi-label">{meta.label}</span>
          </div>
          <svg className="rp-ring-svg" viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="54" fill="none" stroke="var(--bg4)" strokeWidth="6" />
            <circle
              cx="60" cy="60" r="54" fill="none"
              stroke={meta.color} strokeWidth="6"
              strokeLinecap="round"
              strokeDasharray={`${2 * Math.PI * 54 * ((6 - esi_level) / 5)} ${2 * Math.PI * 54}`}
              transform="rotate(-90 60 60)"
              style={{ transition: 'stroke-dasharray 0.8s ease' }}
            />
          </svg>
        </div>

        {/* Priority label */}
        <div className="rp-priority" style={{ color: meta.color, background: meta.bg, borderColor: meta.border }}>
          <span className="rp-priority-dot" style={{ background: meta.color }} />
          {meta.short} — {meta.label}
        </div>

        {/* Confidence meter */}
        <div className="rp-conf-section">
          <div className="rp-conf-header">
            <span className="rp-conf-title">Model confidence</span>
            <span className="rp-conf-val" style={{ color: isHighConf ? 'var(--green)' : 'var(--amber)' }}>
              {confidencePct}%
            </span>
          </div>
          <div className="rp-conf-track">
            <div
              className="rp-conf-fill"
              style={{
                width: `${confidencePct}%`,
                background: isHighConf ? 'var(--green)' : 'var(--amber)',
                transition: 'width 0.8s ease',
              }}
            />
          </div>
          {!isHighConf && (
            <p className="rp-conf-note">Low confidence — clinical override recommended</p>
          )}
        </div>

        {/* SHAP risk factors */}
        {shapFactors.length > 0 && (
          <div className="rp-shap-section">
            <h3 className="rp-section-title">Top Risk Factors</h3>
            <div className="rp-shap-list">
              {shapFactors.map((factor, i) => (
                <div key={i} className="rp-shap-row">
                  <div className="rp-shap-rank">{i + 1}</div>
                  <div className="rp-shap-content">
                    <div className="rp-shap-name">{factor}</div>
                    <div className="rp-shap-bar-track">
                      <div
                        className="rp-shap-bar-fill"
                        style={{
                          width: `${Math.round(barWeights[i] * 100)}%`,
                          background: meta.color,
                          opacity: 0.7 - i * 0.12,
                          transition: `width 0.6s ease ${i * 0.15}s`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Quick guidance */}
        <div className="rp-guidance">
          <h3 className="rp-section-title">Recommended Action</h3>
          <p className="rp-guidance-text">{getGuidance(esi_level)}</p>
        </div>

        {/* Footer */}
        <div className="rp-footer">
          <span className="rp-model-ver">model {model_version}</span>
          <button className="rp-clear-btn" onClick={onClear}>New patient →</button>
        </div>
      </div>
    </div>
  )
}

function getGuidance(level) {
  const map = {
    1: 'Immediate resuscitation. Activate trauma/code team NOW. Do not leave unattended.',
    2: 'Emergent — physician assessment within 15 minutes. Continuous monitoring required.',
    3: 'Urgent — reassess every 30 min. Two or more resources needed. IV access recommended.',
    4: 'Less urgent — one resource needed. Can safely wait 60–120 minutes.',
    5: 'Non-urgent — routine care. Consider discharge with GP follow-up.',
  }
  return map[level] ?? '—'
}

import { useState } from 'react'
import { submitTriage } from '../services/api.js'
import { COMPLAINT_LABELS } from '../services/helpers.js'
import ResultPanel from './ResultPanel.jsx'
import './TriageForm.css'

const INITIAL = {
  age: '',
  gender: 'M',
  vitals: {
    heart_rate: '',
    systolic_bp: '',
    diastolic_bp: '',
    resp_rate: '',
    spo2: '',
    temperature: '',
  },
  comorbidities: {
    has_diabetes: 0,
    has_hypertension: 0,
    has_heart_disease: 0,
    has_sepsis: 0,
    has_resp_failure: 0,
  },
  complaint_cat: 0,
}

const VITAL_META = [
  { key: 'heart_rate', label: 'Heart Rate', unit: 'bpm', placeholder: '60–100' },
  { key: 'systolic_bp', label: 'Systolic BP', unit: 'mmHg', placeholder: '100–140' },
  { key: 'diastolic_bp', label: 'Diastolic BP', unit: 'mmHg', placeholder: '60–90' },
  { key: 'resp_rate', label: 'Resp Rate', unit: '/min', placeholder: '12–20' },
  { key: 'spo2', label: 'SpO₂', unit: '%', placeholder: '95–100' },
  { key: 'temperature', label: 'Temperature', unit: '°C', placeholder: '36–38' },
]

const COMORBIDITY_META = [
  { key: 'has_diabetes', label: 'Diabetes' },
  { key: 'has_hypertension', label: 'Hypertension' },
  { key: 'has_heart_disease', label: 'Heart Disease' },
  { key: 'has_sepsis', label: 'Sepsis' },
  { key: 'has_resp_failure', label: 'Resp. Failure' },
]

export default function TriageForm() {
  const [form, setForm] = useState(INITIAL)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // ── field helpers ────────────────────────────────────────
  function setVital(key, val) {
    setForm(f => ({ ...f, vitals: { ...f.vitals, [key]: val } }))
  }
  function toggleComorbidity(key) {
    setForm(f => ({
      ...f,
      comorbidities: { ...f.comorbidities, [key]: f.comorbidities[key] ? 0 : 1 },
    }))
  }

  // ── submit ───────────────────────────────────────────────
  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    setResult(null)
    setLoading(true)

    const payload = {
      age: Number(form.age),
      gender: form.gender,
      vitals: Object.fromEntries(
        Object.entries(form.vitals).map(([k, v]) => [k, v === '' ? null : Number(v)])
      ),
      comorbidities: form.comorbidities,
      complaint_cat: Number(form.complaint_cat),
    }

    try {
      const data = await submitTriage(payload)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setForm(INITIAL)
    setResult(null)
    setError(null)
  }

  return (
    <div className="triage-form-page">
      <div className="tf-left">
        {/* Header */}
        <div className="tf-header">
          <div className="tf-pulse" />
          <div>
            <h1 className="tf-title">New Patient Assessment</h1>
            <p className="tf-sub">Enter vitals and demographics to receive an ESI prediction.</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="tf-form">

          {/* Demographics */}
          <fieldset className="tf-section">
            <legend className="tf-legend">Demographics</legend>
            <div className="tf-demo-row">
              <div className="tf-field">
                <label className="tf-label">Age</label>
                <div className="tf-input-wrap">
                  <input
                    type="number" min="0" max="120" required
                    className="tf-input"
                    placeholder="e.g. 45"
                    value={form.age}
                    onChange={e => setForm(f => ({ ...f, age: e.target.value }))}
                  />
                  <span className="tf-unit">yrs</span>
                </div>
              </div>
              <div className="tf-field">
                <label className="tf-label">Gender</label>
                <div className="tf-gender-toggle">
                  {['M', 'F'].map(g => (
                    <button
                      key={g} type="button"
                      className={`tf-gender-btn${form.gender === g ? ' active' : ''}`}
                      onClick={() => setForm(f => ({ ...f, gender: g }))}
                    >
                      {g === 'M' ? 'Male' : 'Female'}
                    </button>
                  ))}
                </div>
              </div>
              <div className="tf-field tf-field--wide">
                <label className="tf-label">Chief Complaint</label>
                <select
                  className="tf-select"
                  value={form.complaint_cat}
                  onChange={e => setForm(f => ({ ...f, complaint_cat: e.target.value }))}
                >
                  {Object.entries(COMPLAINT_LABELS).map(([k, v]) => (
                    <option key={k} value={k}>{v}</option>
                  ))}
                </select>
              </div>
            </div>
          </fieldset>

          {/* Vitals */}
          <fieldset className="tf-section">
            <legend className="tf-legend">Vitals</legend>
            <div className="tf-vitals-grid">
              {VITAL_META.map(({ key, label, unit, placeholder }) => {
                const val = form.vitals[key]
                return (
                  <div key={key} className="tf-vital-card">
                    <label className="tf-vital-label">{label}</label>
                    <div className="tf-vital-input-row">
                      <input
                        type="number" step="any"
                        className="tf-vital-input"
                        placeholder={placeholder}
                        value={val}
                        onChange={e => setVital(key, e.target.value)}
                      />
                      <span className="tf-vital-unit">{unit}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </fieldset>

          {/* Comorbidities */}
          <fieldset className="tf-section">
            <legend className="tf-legend">Comorbidities</legend>
            <div className="tf-comorbidity-grid">
              {COMORBIDITY_META.map(({ key, label }) => (
                <button
                  key={key} type="button"
                  className={`tf-comorbidity-btn${form.comorbidities[key] ? ' active' : ''}`}
                  onClick={() => toggleComorbidity(key)}
                >
                  <span className="tf-comorbidity-dot" />
                  {label}
                </button>
              ))}
            </div>
          </fieldset>

          {error && (
            <div className="tf-error">
              <span>⚠</span> {error}
            </div>
          )}

          {/* Actions */}
          <div className="tf-actions">
            <button type="button" className="tf-btn-reset" onClick={handleReset}>
              Clear
            </button>
            <button type="submit" className="tf-btn-submit" disabled={loading || !form.age}>
              {loading ? (
                <>
                  <span className="tf-spinner" />
                  Analysing…
                </>
              ) : (
                <>
                  <span className="tf-btn-icon">→</span>
                  Run Triage Assessment
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* Result panel */}
      <div className="tf-right">
        {result ? (
          <ResultPanel result={result} onClear={() => setResult(null)} />
        ) : (
          <div className="tf-right-empty">
            <div className="tf-empty-icon">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2">
                <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5.586a1 1 0 0 1 .707.293l5.414 5.414A1 1 0 0 1 19 9.414V19a2 2 0 0 1-2 2z" />
              </svg>
            </div>
            <p className="tf-empty-text">Complete the form to<br />receive an ESI assessment</p>
          </div>
        )}
      </div>
    </div>
  )
}

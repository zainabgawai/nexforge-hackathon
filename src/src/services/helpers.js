export const ESI_META = {
  1: { label: 'Resuscitation', short: 'ESI-1', color: 'var(--esi1)', bg: 'var(--esi1bg)', border: 'var(--esi1border)' },
  2: { label: 'Emergent',      short: 'ESI-2', color: 'var(--esi2)', bg: 'var(--esi2bg)', border: 'var(--esi2border)' },
  3: { label: 'Urgent',        short: 'ESI-3', color: 'var(--esi3)', bg: 'var(--esi3bg)', border: 'var(--esi3border)' },
  4: { label: 'Less Urgent',   short: 'ESI-4', color: 'var(--esi4)', bg: 'var(--esi4bg)', border: 'var(--esi4border)' },
  5: { label: 'Non-Urgent',    short: 'ESI-5', color: 'var(--esi5)', bg: 'var(--esi5bg)', border: 'var(--esi5border)' },
}

export function esiMeta(level) {
  return ESI_META[level] ?? ESI_META[3]
}

export const COMPLAINT_LABELS = {
  0: 'Other',
  1: 'Cardiac',
  2: 'Infection',
  3: 'Respiratory',
  4: 'Neurological',
  5: 'Abdominal',
}

// ── MOCK DATA for offline / Day 1 ────────────────────────────
export function mockQueueData() {
  return [
    { id: 1, name: 'Al-Rashid, M.', age: 67, gender: 'M', wait: 4,  esi_level: 1, confidence: 0.91, complaint_cat: 1, top_risk_factors: ['Heart rate critically elevated', 'Low systolic BP', 'Cardiac complaint'] },
    { id: 2, name: 'Okonkwo, F.',   age: 34, gender: 'F', wait: 12, esi_level: 2, confidence: 0.84, complaint_cat: 4, top_risk_factors: ['Neurological complaint', 'Elevated temperature', 'Age risk'] },
    { id: 3, name: 'Sharma, K.',    age: 51, gender: 'F', wait: 18, esi_level: 2, confidence: 0.79, complaint_cat: 3, top_risk_factors: ['Low SpO2', 'Elevated resp rate', 'Respiratory complaint'] },
    { id: 4, name: 'Bergmann, T.',  age: 28, gender: 'M', wait: 31, esi_level: 3, confidence: 0.72, complaint_cat: 5, top_risk_factors: ['Abdominal complaint', 'Mild fever', 'Shock index 0.74'] },
    { id: 5, name: 'Delacroix, R.', age: 78, gender: 'F', wait: 45, esi_level: 3, confidence: 0.68, complaint_cat: 0, top_risk_factors: ['Age ≥ 75', 'Hypertension flag', 'Elevated HR'] },
    { id: 6, name: 'Petrov, A.',    age: 22, gender: 'M', wait: 62, esi_level: 4, confidence: 0.81, complaint_cat: 2, top_risk_factors: ['Low severity infection', 'Stable vitals', 'Young age'] },
    { id: 7, name: 'Nakamura, Y.',  age: 45, gender: 'F', wait: 71, esi_level: 5, confidence: 0.88, complaint_cat: 0, top_risk_factors: ['All vitals stable', 'No comorbidities', 'Non-urgent complaint'] },
  ]
}

export function mockBedsData() {
  const zones = ['Resus', 'Majors', 'Minors']
  return Array.from({ length: 18 }, (_, i) => {
    const zone = i < 3 ? 'Resus' : i < 10 ? 'Majors' : 'Minors'
    const roll = Math.random()
    const status = roll < 0.45 ? 'occupied' : roll < 0.62 ? 'pending' : 'available'
    return { id: i + 1, code: `${zone[0]}${i + 1}`, zone, status }
  })
}

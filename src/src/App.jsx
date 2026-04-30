import { useState } from 'react'
import TriageForm from './components/TriageForm.jsx'
import Dashboard from './components/Dashboard.jsx'
import './styles/App.css'

export default function App() {
  const [view, setView] = useState('form')

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-logo">
          <div className="app-logo-pulse" />
          <span className="app-logo-text">ER Triage System</span>
          <span className="app-logo-tag">ESI Predictor</span>
        </div>

        <nav className="app-nav">
          <button
            className={`app-nav-btn${view === 'form' ? ' active' : ''}`}
            onClick={() => setView('form')}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 5v14M5 12h14"/>
            </svg>
            New Patient
          </button>
          <button
            className={`app-nav-btn${view === 'dashboard' ? ' active' : ''}`}
            onClick={() => setView('dashboard')}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
              <rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
            </svg>
            Dashboard
          </button>
        </nav>

        <div className="app-header-right">
          <span className="app-status-dot" />
          <span className="app-status-txt">MIMIC-III · XGBoost + SHAP</span>
        </div>
      </header>

      <main className="app-main">
        {view === 'form'      && <TriageForm />}
        {view === 'dashboard' && <Dashboard />}
      </main>
    </div>
  )
}

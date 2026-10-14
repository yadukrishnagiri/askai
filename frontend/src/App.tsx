import { useEffect, useState, type FormEvent, type ReactNode } from 'react'
import { Link, NavLink, Route, Routes, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { ArrowLeft, ArrowRight, Check, ChevronRight, ClipboardList, FileText, Link2, LoaderCircle, Menu, Plus, Send, ShieldCheck, Sparkles, Target, UploadCloud, X } from 'lucide-react'
import { api } from './api'
import type { Dashboard, InterviewConfig, RecruiterSession, Session } from './types'

function getError(error: unknown) { return error instanceof Error ? error.message : 'Something went wrong.' }
function absolute(path: string) { return `${window.location.origin}${path}` }

function Shell({ children, active = 'practice' }: { children: ReactNode; active?: 'practice' | 'create' | 'reports' }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const [demo, setDemo] = useState(false)
  useEffect(() => { api.health().then(result => setDemo(result.demo_mode)).catch(() => {}) }, [])
  return <div className="app-shell">
    <aside className={`sidebar ${menuOpen ? 'sidebar-open' : ''}`}>
      <div className="brand"><span className="brand-mark">✦</span> askai <button className="mobile-menu" aria-label="Close menu" onClick={() => setMenuOpen(false)}><X size={20} /></button></div>
      <div className="nav-label">WORKSPACE</div>
      <nav aria-label="Main navigation">
        <NavLink className={active === 'practice' ? 'nav-item active' : 'nav-item'} to="/" onClick={() => setMenuOpen(false)}><Sparkles size={18} /> Practice interview</NavLink>
        <NavLink className={active === 'create' ? 'nav-item active' : 'nav-item'} to="/recruiter/new" onClick={() => setMenuOpen(false)}><Plus size={18} /> Create interview</NavLink>
      </nav>
      <div className="sidebar-bottom"><span className="sidebar-icon"><ShieldCheck size={18} /></span><span>Thoughtful interviews.<br />Better conversations.</span></div>
    </aside>
    <div className="mobile-topbar"><Link to="/" className="mobile-brand"><span>✦</span> askai</Link><button aria-label="Open menu" onClick={() => setMenuOpen(true)}><Menu size={23} /></button></div>
    {menuOpen && <button className="menu-backdrop" aria-label="Close menu" onClick={() => setMenuOpen(false)} />}
    <main className="main-content">{demo && <div className="demo-banner">Demo mode is on. AI analysis and scores are illustrative until Gemini is configured.</div>}{children}</main>
  </div>
}

function PageIntro({ eyebrow, title, subtitle }: { eyebrow?: string; title: string; subtitle: string }) {
  return <header className="page-intro">{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h1>{title}</h1><p>{subtitle}</p></header>
}

function ErrorMessage({ message }: { message: string }) { return <div role="alert" className="error-message">{message}</div> }
function Button({ children, disabled = false, type = 'button', onClick, variant = 'primary' }: { children: ReactNode; disabled?: boolean; type?: 'button' | 'submit'; onClick?: () => void; variant?: 'primary' | 'secondary' }) {
  return <button type={type} disabled={disabled} onClick={onClick} className={`button button-${variant}`}>{children}</button>
}

function Home() {
  const navigate = useNavigate()
  const [role, setRole] = useState('')
  const roles = ['Product Designer', 'Frontend Engineer', 'Data Analyst']
  return <Shell><div className="content-wrap home-wrap">
    <PageIntro eyebrow="AI INTERVIEW STUDIO" title="Practice with purpose." subtitle="A focused interview built around your experience and the role you want." />
    <div className="home-grid">
      <section className="card role-card"><div className="step-chip">STEP 01 / 03</div><h2>Tell us what you are aiming for</h2><p>We will use your role and CV to shape the conversation.</p>
        <label className="field-label" htmlFor="role">TARGET ROLE</label><input id="role" value={role} onChange={event => setRole(event.target.value)} placeholder="e.g. Product Designer" maxLength={120} />
        <div className="field-label suggested-label">SUGGESTED ROLES</div><div className="chip-row">{roles.map(value => <button key={value} className="choice-chip" onClick={() => setRole(value)}>{value}</button>)}</div>
        <Button disabled={role.trim().length < 2} onClick={() => navigate(`/start?role=${encodeURIComponent(role.trim())}`)}>Continue to CV <ArrowRight size={17} /></Button>
      </section>
      <aside className="insight-card"><div className="insight-icon"><Target size={25} /></div><h3>Built around your story</h3><p>Questions connect your CV, target role, and the details you share as the interview unfolds.</p><div className="insight-divider" /><div className="insight-stat"><strong>01</strong><span>Real experience first</span></div><div className="insight-stat"><strong>02</strong><span>Relevant follow-ups</span></div><div className="insight-stat"><strong>03</strong><span>Clear coaching at the end</span></div></aside>
    </div>
    <div className="subtle-footer">Creating interviews for others? <Link to="/recruiter/new">Open recruiter studio <ChevronRight size={14} /></Link></div>
  </div></Shell>
}

function Start() {
  const { configurationToken } = useParams()
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const [role, setRole] = useState(params.get('role') ?? '')
  const [candidateName, setCandidateName] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [fetching, setFetching] = useState(Boolean(configurationToken))
  const [error, setError] = useState('')
  const [resumeToken, setResumeToken] = useState<string | null>(null)
  useEffect(() => {
    if (!configurationToken) return
    api.configuration(configurationToken).then(result => setRole(result.settings.role)).catch(error => setError(getError(error))).finally(() => setFetching(false))
  }, [configurationToken])
  useEffect(() => {
    const key = `askai:session:${configurationToken || role}`
    const stored = localStorage.getItem(key)
    if (stored) setResumeToken(stored)
  }, [configurationToken, role])
  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!file) { setError('Choose a PDF CV to continue.'); return }
    const data = new FormData()
    data.append('candidate_name', candidateName)
    data.append('role', role)
    data.append('configuration_token', configurationToken ?? '')
    data.append('cv', file)
    setError(''); setLoading(true)
    try {
      const session = await api.startSession(data)
      localStorage.setItem(`askai:session:${configurationToken || role}`, session.token)
      navigate(`/session/${session.token}`)
    } catch (failure) { setError(getError(failure)) } finally { setLoading(false) }
  }
  return <Shell><div className="content-wrap narrow-wrap"><Link className="back-link" to="/"><ArrowLeft size={16} /> Back to role selection</Link>
    <PageIntro eyebrow="STEP 02 / 03" title="Make it personal." subtitle="Upload your PDF CV so the questions reflect your real projects and skills." />
    {fetching ? <div className="card loading-card"><LoaderCircle className="spin" /> Loading interview…</div> : <form className="card upload-card" onSubmit={submit}>
      <div className="card-header"><div className="card-icon"><FileText size={22} /></div><div><h2>Your details and CV</h2><p>We need a name and a text-based PDF to begin.</p></div></div>
      {error && <ErrorMessage message={error} />}
      <div className="form-grid"><div><label className="field-label" htmlFor="candidate">YOUR NAME</label><input id="candidate" value={candidateName} onChange={event => setCandidateName(event.target.value)} placeholder="e.g. Alex Morgan" required minLength={2} maxLength={100} /></div><div><label className="field-label" htmlFor="target-role">TARGET ROLE</label><input id="target-role" value={role} onChange={event => setRole(event.target.value)} disabled={Boolean(configurationToken)} required minLength={2} maxLength={120} /></div></div>
      <label className="upload-zone" htmlFor="cv"><UploadCloud size={33} /><strong>{file ? file.name : 'Choose a PDF CV'}</strong><span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB selected` : 'Select a text-based PDF · up to 10 MB'}</span><input id="cv" type="file" accept="application/pdf,.pdf" onChange={event => setFile(event.target.files?.[0] ?? null)} required /></label>
      <p className="privacy-note"><ShieldCheck size={16} /> The PDF is deleted after text extraction. Your analysis and interview results are saved locally.</p>
      <div className="form-actions"><Button type="submit" disabled={loading || !file || !role.trim() || candidateName.trim().length < 2}>{loading ? <><LoaderCircle className="spin" size={17} /> Preparing interview…</> : <>Start interview <ArrowRight size={17} /></>}</Button></div>
      {resumeToken && <p className="resume-link">Already started? <Link to={`/session/${resumeToken}`}>Resume your interview <ChevronRight size={15} /></Link></p>}
    </form>}
  </div></Shell>
}

function SessionPage() {
  const { accessToken = '' } = useParams()
  const [session, setSession] = useState<Session | null>(null)
  const [answer, setAnswer] = useState('')
  const [started, setStarted] = useState(sessionStorage.getItem(`askai:begun:${accessToken}`) === 'yes')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => { api.session(accessToken).then(setSession).catch(failure => setError(getError(failure))) }, [accessToken])
  async function submit(event: FormEvent) {
    event.preventDefault()
    if (!answer.trim()) return
    setLoading(true); setError('')
    try { const next = await api.answer(accessToken, answer.trim()); setSession(next); setAnswer(''); window.scrollTo({ top: 0, behavior: 'smooth' }) }
    catch (failure) { setError(getError(failure)) }
    finally { setLoading(false) }
  }
  if (!session) return <Shell><div className="content-wrap narrow-wrap">{error ? <ErrorMessage message={error} /> : <div className="card loading-card"><LoaderCircle className="spin" /> Loading your interview…</div>}</div></Shell>
  if (session.status === 'complete') return <CandidateResults session={session} />
  if (!started && session.answered_count === 0) return <Shell><div className="content-wrap narrow-wrap"><PageIntro eyebrow="CV ANALYSIS COMPLETE" title="Ready when you are." subtitle={`Your ${session.role} interview is prepared from your CV.`} />
    <div className="card analysis-card"><div className="card-header"><div className="card-icon"><Sparkles size={22} /></div><div><h2>Your experience, in focus</h2><p>{session.cv_analysis.summary}</p></div></div><div className="analysis-section"><span className="field-label">IDENTIFIED SKILLS</span><div className="chip-row">{Object.keys(session.cv_analysis.technical_skills).slice(0, 8).map(skill => <span className="choice-chip static" key={skill}>{skill}</span>)}</div></div><div className="analysis-section"><span className="field-label">INTERVIEW FORMAT</span><p>{session.question_count} questions · starts at {session.difficulty} difficulty · feedback at the end</p></div><Button onClick={() => { sessionStorage.setItem(`askai:begun:${accessToken}`, 'yes'); setStarted(true) }}>Begin interview <ArrowRight size={17} /></Button></div>
  </div></Shell>
  return <Shell><div className="content-wrap interview-wrap"><div className="interview-heading"><div><span className="eyebrow">LIVE INTERVIEW</span><h1>Interview in progress</h1><p>{session.role} <span className="dot">·</span> Question {session.answered_count + 1} of {session.question_count}</p></div><div className="progress-count">{Math.round(session.answered_count / session.question_count * 100)}% complete</div></div>
    <div className="progress-track"><div style={{ width: `${session.answered_count / session.question_count * 100}%` }} /></div>
    <div className="interview-grid"><section className="card question-card"><div className="question-meta"><span className="step-chip">{session.current_question?.difficulty.toUpperCase()}</span><span>{session.current_question?.kind === 'followup' ? 'Follow-up question' : session.current_question?.topic}</span></div><h2>{session.current_question?.text}</h2><form onSubmit={submit}><label className="field-label" htmlFor="answer">YOUR ANSWER</label><textarea id="answer" value={answer} onChange={event => setAnswer(event.target.value)} placeholder="Describe your approach, the decisions you made, and the outcome…" rows={8} maxLength={12000} required /><div className="answer-footer"><p>Take your time. Feedback appears when the interview ends.</p><Button type="submit" disabled={loading || !answer.trim()}>{loading ? <><LoaderCircle className="spin" size={17} /> Reviewing…</> : <>Submit answer <Send size={16} /></>}</Button></div></form>{error && <ErrorMessage message={error} />}</section>
      <aside className="side-panel"><h3>Interview guide</h3><div className="guide-item"><span>01</span><div><strong>Use a real example</strong><p>Choose work you can describe in detail.</p></div></div><div className="guide-item"><span>02</span><div><strong>Explain your decisions</strong><p>Share the tradeoffs behind your approach.</p></div></div><div className="guide-item"><span>03</span><div><strong>Finish with impact</strong><p>Explain what changed and how you know.</p></div></div></aside></div>
    {session.turns.length > 0 && <div className="answered-list"><h3>Previously answered</h3>{session.turns.map(turn => <div key={turn.position}><span>Q{turn.position}</span><p>{turn.question.text}</p><Check size={17} /></div>)}</div>}
  </div></Shell>
}

function CandidateResults({ session }: { session: Session }) {
  return <Shell><div className="content-wrap results-wrap"><PageIntro eyebrow="INTERVIEW COMPLETE" title="Your interview, unpacked." subtitle="Clear strengths and practical next steps for your next conversation." />
    <div className="results-grid"><div className="card score-card"><span className="field-label">OVERALL SCORE</span><div className="score-number">{session.score}<span>/ 10</span></div><div className="score-caption">{session.role} interview</div><div className="score-bar"><span style={{ width: `${(session.score ?? 0) * 10}%` }} /></div><p>Your result reflects the configured rubric across {session.question_count} questions.</p></div><div className="card report-card"><h2>Your interview summary</h2><p className="lead-copy">{session.report?.summary}</p><div className="report-columns"><div><h3>What went well</h3><ul>{session.report?.strengths.map(item => <li key={item}>{item}</li>)}</ul></div><div><h3>Build on this</h3><ul>{session.report?.improvements.map(item => <li key={item}>{item}</li>)}</ul></div></div></div></div>
    <section className="card next-card"><div className="card-header"><div className="card-icon"><Target size={22} /></div><div><h2>Your next steps</h2><p>Turn this interview into stronger future answers.</p></div></div><ol>{session.report?.next_steps.map(item => <li key={item}>{item}</li>)}</ol></section>
    <section className="card answer-review"><h2>Question review</h2>{session.turns.map(turn => <details key={turn.position}><summary><span>Question {turn.position}</span>{turn.question.text}<ChevronRight size={18} /></summary><div className="review-body"><p><strong>Your answer</strong><br />{turn.answer}</p><p><strong>Feedback</strong><br />{turn.evaluation?.feedback}</p><div className="mini-scores"><span>Technical {turn.evaluation?.technical_accuracy}/10</span><span>Communication {turn.evaluation?.communication}/10</span><span>Evidence {turn.evaluation?.evidence}/10</span></div></div></details>)}</section>
    <div className="page-end-action"><Link className="button button-secondary" to="/">Practice another role <ArrowRight size={17} /></Link></div>
  </div></Shell>
}

const defaultSettings: InterviewConfig = { role: '', description: '', skills: [], starting_difficulty: 'basic', question_count: 8, technical_weight: 50, communication_weight: 30, evidence_weight: 20, custom_questions: [] }

function RecruiterCreate() {
  const navigate = useNavigate()
  const [settings, setSettings] = useState<InterviewConfig>(defaultSettings)
  const [skillsText, setSkillsText] = useState('')
  const [questionsText, setQuestionsText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  function update<K extends keyof InterviewConfig>(key: K, value: InterviewConfig[K]) { setSettings(previous => ({ ...previous, [key]: value })) }
  async function submit(event: FormEvent) {
    event.preventDefault(); setError('')
    const payload = { ...settings, skills: skillsText.split(',').map(value => value.trim()).filter(Boolean), custom_questions: questionsText.split('\n').map(value => value.trim()).filter(Boolean) }
    setLoading(true)
    try { const created = await api.createConfiguration(payload); navigate(created.dashboard_url) }
    catch (failure) { setError(getError(failure)) }
    finally { setLoading(false) }
  }
  return <Shell active="create"><div className="content-wrap recruiter-wrap"><PageIntro eyebrow="RECRUITER STUDIO" title="Create an interview." subtitle="Set the role, focus areas, and assessment shape. Then share one candidate link." />
    <form className="card config-card" onSubmit={submit}>{error && <ErrorMessage message={error} />}
      <div className="form-section"><div className="section-heading"><span>01</span><div><h2>Role and focus</h2><p>Give the interviewer a clear brief.</p></div></div><div className="form-grid"><div><label className="field-label" htmlFor="config-role">ROLE TITLE</label><input id="config-role" value={settings.role} onChange={event => update('role', event.target.value)} placeholder="e.g. Senior Product Designer" required minLength={2} maxLength={120} /></div><div><label className="field-label" htmlFor="count">QUESTION COUNT</label><select id="count" value={settings.question_count} onChange={event => update('question_count', Number(event.target.value))}>{[3, 5, 8, 10, 12, 15].map(count => <option key={count} value={count}>{count} questions</option>)}</select></div></div><label className="field-label" htmlFor="description">ROLE DESCRIPTION</label><textarea id="description" value={settings.description} onChange={event => update('description', event.target.value)} placeholder="Responsibilities, context, and what success looks like…" rows={3} maxLength={1500} /><label className="field-label" htmlFor="skills">TARGET SKILLS</label><input id="skills" value={skillsText} onChange={event => setSkillsText(event.target.value)} placeholder="Research, systems thinking, collaboration" /><p className="field-help">Separate skills with commas. Up to 10.</p></div>
      <div className="form-section"><div className="section-heading"><span>02</span><div><h2>Interview settings</h2><p>Choose the challenge level and how answers are scored.</p></div></div><div className="form-grid"><div><label className="field-label" htmlFor="difficulty">STARTING DIFFICULTY</label><select id="difficulty" value={settings.starting_difficulty} onChange={event => update('starting_difficulty', event.target.value as InterviewConfig['starting_difficulty'])}><option value="basic">Basic</option><option value="intermediate">Intermediate</option><option value="advanced">Advanced</option></select></div><div className="weight-summary">The three scoring weights must total <strong>100%</strong>.</div></div><div className="weight-grid"><div><label className="field-label" htmlFor="technical">TECHNICAL</label><input id="technical" type="number" min="0" max="100" value={settings.technical_weight} onChange={event => update('technical_weight', Number(event.target.value))} /></div><div><label className="field-label" htmlFor="communication">COMMUNICATION</label><input id="communication" type="number" min="0" max="100" value={settings.communication_weight} onChange={event => update('communication_weight', Number(event.target.value))} /></div><div><label className="field-label" htmlFor="evidence">EVIDENCE</label><input id="evidence" type="number" min="0" max="100" value={settings.evidence_weight} onChange={event => update('evidence_weight', Number(event.target.value))} /></div></div><label className="field-label" htmlFor="custom">OPTIONAL CUSTOM QUESTIONS</label><textarea id="custom" value={questionsText} onChange={event => setQuestionsText(event.target.value)} placeholder="One question per line, up to five" rows={3} /><p className="field-help">Custom questions are mixed into the adaptive interview.</p></div>
      <div className="config-footer"><p><Link2 size={17} /> You will receive separate candidate and private dashboard links.</p><Button type="submit" disabled={loading || settings.role.trim().length < 2}>{loading ? <><LoaderCircle className="spin" size={17} /> Creating…</> : <>Create interview <ArrowRight size={17} /></>}</Button></div>
    </form>
  </div></Shell>
}

function DashboardPage() {
  const { dashboardToken = '' } = useParams()
  const [data, setData] = useState<Dashboard | null>(null)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState<'candidate' | 'dashboard' | null>(null)
  useEffect(() => { api.dashboard(dashboardToken).then(setData).catch(failure => setError(getError(failure))) }, [dashboardToken])
  async function copy(kind: 'candidate' | 'dashboard', path: string) { await navigator.clipboard.writeText(absolute(path)); setCopied(kind); setTimeout(() => setCopied(null), 2200) }
  return <Shell active="reports"><div className="content-wrap dashboard-wrap">{error ? <ErrorMessage message={error} /> : !data ? <div className="card loading-card"><LoaderCircle className="spin" /> Loading dashboard…</div> : <>
    <PageIntro eyebrow="PRIVATE RECRUITER DASHBOARD" title="Interview results." subtitle={`${data.settings.role} · ${data.settings.question_count} questions · ${data.sessions.length} candidates`} />
    <div className="dashboard-summary"><div className="summary-tile"><span>COMPLETED</span><strong>{data.sessions.filter(row => row.status === 'complete').length}</strong><p>Interview reports</p></div><div className="summary-tile"><span>IN PROGRESS</span><strong>{data.sessions.filter(row => row.status === 'active').length}</strong><p>Candidate sessions</p></div><div className="summary-tile"><span>QUESTION COUNT</span><strong>{data.settings.question_count}</strong><p>Per interview</p></div></div>
    <div className="card link-card"><div><span className="field-label">CANDIDATE LINK</span><h2>Ready to share</h2><p>Anyone with this link can upload a CV and take the configured interview.</p><code>{absolute(data.candidate_url)}</code></div><Button onClick={() => copy('candidate', data.candidate_url)} variant="secondary">{copied === 'candidate' ? <><Check size={17} /> Copied</> : <><Link2 size={17} /> Copy link</>}</Button></div>
    <div className="private-link"><ShieldCheck size={16} /> Keep this dashboard URL private. <button onClick={() => copy('dashboard', window.location.pathname)}>{copied === 'dashboard' ? 'Copied' : 'Copy dashboard link'}</button></div>
    <section className="card table-card"><div className="table-header"><div><h2>Completed interviews</h2><p>Open a report to review evidence and rubric scores.</p></div><span>{data.sessions.length} total</span></div>{data.sessions.length ? <div className="table-scroll"><table><thead><tr><th>Candidate</th><th>Progress</th><th>Last updated</th><th>Score</th><th /></tr></thead><tbody>{data.sessions.map(row => <tr key={row.id}><td><strong>{row.candidate_name}</strong><small>{row.role}</small></td><td><span className={`status status-${row.status}`}>{row.status === 'complete' ? 'Completed' : `${row.answered_count}/${row.question_count} answered`}</span></td><td>{new Date(row.updated_at).toLocaleString()}</td><td><strong>{row.score === null ? '—' : `${row.score}/10`}</strong></td><td><Link aria-label={`View ${row.candidate_name}'s report`} to={`/dashboard/${dashboardToken}/report/${row.id}`}><ArrowRight size={18} /></Link></td></tr>)}</tbody></table></div> : <div className="empty-state"><ClipboardList size={30} /><h3>No interviews yet</h3><p>Share the candidate link to collect your first report.</p></div>}</section>
  </>}</div></Shell>
}

function RecruiterReportPage() {
  const { dashboardToken = '', sessionId = '' } = useParams()
  const [data, setData] = useState<RecruiterSession | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { api.recruiterSession(dashboardToken, sessionId).then(setData).catch(failure => setError(getError(failure))) }, [dashboardToken, sessionId])
  return <Shell active="reports"><div className="content-wrap results-wrap"><Link className="back-link" to={`/dashboard/${dashboardToken}`}><ArrowLeft size={16} /> Back to dashboard</Link>{error ? <ErrorMessage message={error} /> : !data ? <div className="card loading-card"><LoaderCircle className="spin" /> Loading report…</div> : <>
    <PageIntro eyebrow="CANDIDATE ASSESSMENT" title={data.candidate_name} subtitle={`${data.role} · ${data.status === 'complete' ? 'Interview complete' : 'Interview in progress'}`} />
    {data.report ? <><div className="results-grid"><div className="card score-card"><span className="field-label">OVERALL SCORE</span><div className="score-number">{data.score}<span>/ 10</span></div><div className="score-caption">Configured rubric</div><div className="score-bar"><span style={{ width: `${(data.score ?? 0) * 10}%` }} /></div></div><div className="card report-card"><h2>Assessment summary</h2><p className="lead-copy">{data.report.summary}</p><div className="report-columns"><div><h3>Strengths</h3><ul>{data.report.strengths.map(value => <li key={value}>{value}</li>)}</ul></div><div><h3>Areas to probe</h3><ul>{data.report.concerns.map(value => <li key={value}>{value}</li>)}</ul></div></div></div></div><div className="card next-card"><h2>Evidence from the interview</h2><ul>{data.report.evidence.map(value => <li key={value}>{value}</li>)}</ul></div></> : <div className="card next-card"><h2>Interview in progress</h2><p>The assessment will appear after the candidate completes all questions.</p></div>}
    <section className="card answer-review"><h2>Questions and answers</h2>{data.turns.map(turn => <details key={turn.position}><summary><span>Question {turn.position}</span>{turn.question.text}<ChevronRight size={18} /></summary><div className="review-body"><p><strong>Candidate answer</strong><br />{turn.answer}</p><p><strong>Evaluation</strong><br />{turn.evaluation?.feedback}</p><div className="mini-scores"><span>Technical {turn.evaluation?.technical_accuracy}/10</span><span>Communication {turn.evaluation?.communication}/10</span><span>Evidence {turn.evaluation?.evidence}/10</span></div></div></details>)}</section>
  </>}</div></Shell>
}

function NotFound() { return <Shell><div className="content-wrap narrow-wrap"><PageIntro title="Page not found." subtitle="That link may be incomplete or unavailable." /><Link className="button button-primary" to="/">Return home <ArrowRight size={16} /></Link></div></Shell> }

export default function App() { return <Routes><Route path="/" element={<Home />} /><Route path="/start" element={<Start />} /><Route path="/interview/:configurationToken" element={<Start />} /><Route path="/session/:accessToken" element={<SessionPage />} /><Route path="/recruiter/new" element={<RecruiterCreate />} /><Route path="/dashboard/:dashboardToken" element={<DashboardPage />} /><Route path="/dashboard/:dashboardToken/report/:sessionId" element={<RecruiterReportPage />} /><Route path="*" element={<NotFound />} /></Routes> }

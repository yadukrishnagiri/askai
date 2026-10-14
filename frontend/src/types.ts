export type Difficulty = 'basic' | 'intermediate' | 'advanced'

export interface InterviewConfig {
  role: string
  description: string
  skills: string[]
  starting_difficulty: Difficulty
  question_count: number
  technical_weight: number
  communication_weight: number
  evidence_weight: number
  custom_questions: string[]
}

export interface Question {
  text: string
  topic: string
  kind: 'opening' | 'followup' | 'coverage' | 'custom'
  difficulty: Difficulty
}

export interface Evaluation {
  technical_accuracy: number
  communication: number
  evidence: number
  feedback: string
  adjust_difficulty: string
  followup_worthwhile: boolean
  suggested_followup: string
}

export interface Turn {
  position: number
  question: Question
  answer: string
  evaluation?: Evaluation
}

export interface CandidateReport {
  summary: string
  strengths: string[]
  improvements: string[]
  next_steps: string[]
}

export interface Session {
  token: string
  candidate_name: string
  role: string
  status: 'active' | 'complete'
  question_count: number
  answered_count: number
  difficulty: Difficulty
  cv_analysis: {
    summary: string
    technical_skills: Record<string, number>
    years_experience: string
    key_achievements: string[]
    areas_for_improvement: string[]
  }
  current_question: Question | null
  turns: Turn[]
  score?: number
  report?: CandidateReport
}

export interface DashboardRow {
  id: string
  candidate_name: string
  role: string
  status: string
  updated_at: string
  answered_count: number
  question_count: number
  score: number | null
}

export interface Dashboard {
  settings: InterviewConfig
  candidate_url: string
  sessions: DashboardRow[]
}

export interface RecruiterReport {
  summary: string
  strengths: string[]
  concerns: string[]
  evidence: string[]
}

export interface RecruiterSession {
  candidate_name: string
  role: string
  status: string
  score: number | null
  report: RecruiterReport | null
  turns: Turn[]
  cv_analysis: Session['cv_analysis']
}

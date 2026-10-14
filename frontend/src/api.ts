import type { Dashboard, InterviewConfig, RecruiterSession, Session } from './types'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api${path}`, options)
  } catch {
    throw new Error('Could not reach the AskAI server. Check that the backend is running.')
  }
  if (!response.ok) {
    let detail = 'Something went wrong. Please try again.'
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') detail = body.detail
      else if (Array.isArray(body.detail) && body.detail[0]?.msg) detail = body.detail[0].msg
    } catch { /* Response was not JSON. */ }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string; demo_mode: boolean; gemini_ready: boolean }>('/health'),
  createConfiguration: (settings: InterviewConfig) => request<{ settings: InterviewConfig; candidate_url: string; dashboard_url: string }>('/configurations', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settings),
  }),
  configuration: (token: string) => request<{ settings: InterviewConfig }>(`/configurations/${encodeURIComponent(token)}`),
  startSession: (data: FormData) => request<Session>('/sessions', { method: 'POST', body: data }),
  session: (token: string) => request<Session>(`/sessions/${encodeURIComponent(token)}`),
  answer: (token: string, answer: string) => request<Session>(`/sessions/${encodeURIComponent(token)}/answers`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ answer }),
  }),
  dashboard: (token: string) => request<Dashboard>(`/dashboard/${encodeURIComponent(token)}`),
  recruiterSession: (token: string, id: string) => request<RecruiterSession>(`/dashboard/${encodeURIComponent(token)}/sessions/${encodeURIComponent(id)}`),
}

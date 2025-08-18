// Simple frontend API client for backend music generation integration
// Step 1: Wire generation UI to real backend /api/v1/music/generate/full-song endpoint.

export interface FullSongBackendResponse {
  project_id: string;
  status: string;
  message: string;
  estimated_completion?: string;
  pipeline_stages?: string[];
  websocket_url?: string;
  collaboration_enabled?: boolean;
  premium_features?: Record<string, any>;
}

const API_BASE: string = (import.meta.env.VITE_API_BASE as string) || '';

// Local storage keys
const TOKEN_KEY = 'dev_jwt_token';
const TOKEN_TS_KEY = 'dev_jwt_token_ts';

function tokenFresh(ts: string | null): boolean {
  if (!ts) return false;
  const saved = parseInt(ts, 10);
  if (isNaN(saved)) return false;
  // treat tokens older than 50 minutes as stale
  return Date.now() - saved < 50 * 60 * 1000;
}

export async function ensureDevToken(): Promise<string> {
  let token = localStorage.getItem(TOKEN_KEY);
  const ts = localStorage.getItem(TOKEN_TS_KEY);
  if (token && tokenFresh(ts)) return token;
  const res = await fetch(`${API_BASE}/api/v1/dev/token`);
  if (!res.ok) throw new Error(`Failed to get dev token (${res.status})`);
  const json = await res.json();
  token = json.token;
  if (!token) throw new Error('Token missing in dev token response');
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(TOKEN_TS_KEY, Date.now().toString());
  return token;
}

export interface GenerateFullSongParams {
  projectName: string;
  lyricsPrompt: string;
  genrePrompt: string;
}

export async function generateFullSongBackend(params: GenerateFullSongParams): Promise<FullSongBackendResponse> {
  const token = await ensureDevToken();
  const body = {
    project_name: params.projectName,
    style_config: {
      genre: 'pop',
      energy: 0.7,
      mood: 'uplifting',
      tempo: 120,
      key: 'C',
      time_signature: '4/4'
    },
    lyrics_request: params.lyricsPrompt ? {
      theme: params.lyricsPrompt.slice(0, 60) || 'music inspiration',
      style: 'pop',
      mood: 'uplifting',
      language: 'english',
      explicit: false
    } : undefined,
    advanced_options: {}
  };
  const res = await fetch(`${API_BASE}/api/v1/music/generate/full-song`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Backend generation failed (${res.status}): ${text}`);
  }
  return res.json();
}

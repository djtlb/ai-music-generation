import { useEffect, useState } from 'react';

export interface BackendHealth {
  status: string;
  services?: Record<string,string>;
  version?: string;
  timestamp?: string;
  error?: string;
}

const API_BASE = (import.meta.env.VITE_API_BASE as string) || '';

export function useBackendHealth(pollMs = 5000) {
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const fetchHealth = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`);
        const json = await res.json();
        if (!cancelled) {
          setHealth(json);
          setError(null);
          setLoading(false);
        }
      } catch (e: any) {
        if (!cancelled) {
          setError(e.message || 'health fetch failed');
          setLoading(false);
        }
      } finally {
        if (!cancelled) {
          timer = window.setTimeout(fetchHealth, pollMs);
        }
      }
    };
    fetchHealth();
    return () => { cancelled = true; if (timer) window.clearTimeout(timer); };
  }, [pollMs]);

  return { health, loading, error };
}

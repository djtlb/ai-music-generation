import { useEffect, useRef } from 'react';

interface WSOptions {
  userId: string | null;
  onEvent?: (evt: any) => void;
  enabled?: boolean;
  apiBase?: string;
}

export function useProjectWebSocket({ userId, onEvent, enabled = true, apiBase = '' }: WSOptions) {
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!enabled || !userId) return;
    const base = apiBase || (import.meta.env.VITE_API_BASE as string) || '';
    const urlBase = base.replace(/^(http)/, 'ws');
    const ws = new WebSocket(`${urlBase}/ws/${userId}`);
    wsRef.current = ws;
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'subscribe', events: ['stage.completed','project.completed','project.failed'] }));
    };
    ws.onmessage = (msg) => {
      try { const data = JSON.parse(msg.data); onEvent && onEvent(data); } catch { /* ignore */ }
    };
    ws.onerror = () => { /* silent */ };
    ws.onclose = () => { wsRef.current = null; };
    return () => { ws.close(); };
  }, [userId, enabled, apiBase, onEvent]);

  return { socket: wsRef.current };
}

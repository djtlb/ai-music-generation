/**
 * API Service for connecting to FastAPI backend
 * This service handles all communication between the React frontend and Python backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

// Types for API requests/responses
export interface StyleConfig {
  genre: string;
  subgenre?: string;
  energy: number;
  mood: string;
  tempo: number;
  key: string;
  timeSignature: string;
}

export interface FullSongRequest {
  project_name: string;
  style_config: StyleConfig;
  lyrics_request?: {
    theme: string;
    style: string;
    mood: string;
    language: string;
    explicit: boolean;
  };
  advanced_options?: Record<string, any>;
  collaboration_enabled?: boolean;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface ProjectStatus {
  project_id: string;
  status: 'processing' | 'completed' | 'failed';
  progress?: number;
  stages?: {
    lyrics?: any;
    arrangement?: any;
    composition?: any;
    sound_design?: any;
    mix_master?: any;
  };
  estimated_completion?: string;
  error?: string;
}

class APIService {
  private apiKey: string | null = null;
  private baseURL: string;

  constructor() {
    this.baseURL = API_BASE_URL;
    this.apiKey = localStorage.getItem('aimusic_api_key') || 'demo_key_12345';
  }

  private async makeRequest<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const url = `${this.baseURL}${endpoint}`;
      
      const defaultHeaders: Record<string, string> = {
        'Content-Type': 'application/json',
      };

      if (this.apiKey) {
        defaultHeaders['Authorization'] = `Bearer ${this.apiKey}`;
      }

      const response = await fetch(url, {
        ...options,
        headers: {
          ...defaultHeaders,
          ...options.headers,
        },
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      return {
        success: true,
        data,
      };
    } catch (error) {
      console.error('API Request failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      };
    }
  }

  // Authentication
  async setApiKey(apiKey: string): Promise<void> {
    this.apiKey = apiKey;
    localStorage.setItem('aimusic_api_key', apiKey);
  }

  // Health check
  async healthCheck(): Promise<ApiResponse> {
    return this.makeRequest('/health');
  }

  // Music Generation Endpoints
  async generateFullSong(request: FullSongRequest): Promise<ApiResponse<{
    project_id: string;
    status: string;
    message: string;
    estimated_completion: string;
    pipeline_stages: string[];
    websocket_url: string;
    premium_features: Record<string, boolean>;
  }>> {
    return this.makeRequest('/music/generate/full-song', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async generateLyrics(request: {
    theme: string;
    style: string;
    mood: string;
    language: string;
    explicit: boolean;
  }): Promise<ApiResponse<{ task_id: string; status: string; message: string }>> {
    return this.makeRequest('/music/generate/lyrics', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async generateArrangement(request: {
    style_config: StyleConfig;
    duration: number;
    complexity: string;
    instruments: string[];
  }): Promise<ApiResponse<{ task_id: string; status: string; message: string }>> {
    return this.makeRequest('/music/generate/arrangement', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Project Management
  async getProjectStatus(projectId: string): Promise<ApiResponse<ProjectStatus>> {
    return this.makeRequest(`/projects/${projectId}/status`);
  }

  async listProjects(): Promise<ApiResponse<Array<{
    id: string;
    name: string;
    status: string;
    created_at: string;
    style_config: StyleConfig;
  }>>> {
    return this.makeRequest('/projects/list');
  }

  // Task Status
  async getTaskStatus(taskId: string): Promise<ApiResponse<{
    id: string;
    type: string;
    status: string;
    progress: number;
    result?: any;
    error?: string;
  }>> {
    return this.makeRequest(`/music/task/${taskId}/status`);
  }

  // Download
  async downloadFile(fileId: string): Promise<string> {
    return `${this.baseURL}/music/download/${fileId}`;
  }

  // Style Presets
  async getStylePresets(): Promise<ApiResponse<{
    genres: Array<{ id: string; name: string; description: string }>;
    moods: string[];
    keys: string[];
    time_signatures: string[];
  }>> {
    return this.makeRequest('/music/styles/presets');
  }

  // Analytics
  async getGenerationStats(): Promise<ApiResponse<{
    total_songs: number;
    monthly_usage: number;
    favorite_genre: string;
    total_playtime: number;
    subscription_tier: string;
  }>> {
    return this.makeRequest('/music/analytics/generation-stats');
  }

  async getSystemStats(): Promise<ApiResponse<{
    total_songs_generated: number;
    active_users: number;
    revenue_today: number;
    server_uptime: string;
    ai_model_performance: Record<string, any>;
  }>> {
    return this.makeRequest('/system/stats');
  }

  // File Upload
  async uploadReferenceAudio(file: File): Promise<ApiResponse<{
    file_id: string;
    message: string;
    processing_status: string;
  }>> {
    const formData = new FormData();
    formData.append('file', file);

    return this.makeRequest('/music/upload/reference', {
      method: 'POST',
      body: formData,
      headers: {}, // Don't set Content-Type for FormData
    });
  }

  // WebSocket URL
  getWebSocketURL(clientId: string): string {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = this.baseURL.replace(/^https?:/, '').replace('/api/v1', '');
    return `${wsProtocol}${wsHost}/ws/${clientId}`;
  }
}

// WebSocket Service for real-time updates
export class WebSocketService {
  private ws: WebSocket | null = null;
  private clientId: string;
  private callbacks: Map<string, (data: any) => void> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  constructor(clientId: string) {
    this.clientId = clientId;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const apiService = new APIService();
        const wsUrl = apiService.getWebSocketURL(this.clientId);
        
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onclose = () => {
          console.log('WebSocket disconnected');
          this.attemptReconnect();
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  private handleMessage(data: any): void {
    const { type } = data;
    
    // Call registered callbacks
    if (this.callbacks.has(type)) {
      this.callbacks.get(type)?.(data);
    }

    // Call general message callback
    if (this.callbacks.has('*')) {
      this.callbacks.get('*')?.(data);
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.pow(2, this.reconnectAttempts) * 1000; // Exponential backoff
      
      console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
      
      setTimeout(() => {
        this.connect().catch(console.error);
      }, delay);
    }
  }

  subscribe(eventType: string, callback: (data: any) => void): void {
    this.callbacks.set(eventType, callback);
    
    // Send subscription message to server
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        events: [eventType]
      }));
    }
  }

  unsubscribe(eventType: string): void {
    this.callbacks.delete(eventType);
    
    // Send unsubscription message to server
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'unsubscribe',
        events: [eventType]
      }));
    }
  }

  send(message: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  disconnect(): void {
    this.callbacks.clear();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

// Global API service instance
export const apiService = new APIService();
export default apiService;

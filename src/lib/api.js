/**
 * Beat Addicts API client
 * Provides a simple interface for interacting with the backend API
 */
class ApiClient {
  constructor(baseUrl = '') {
    this.baseUrl = baseUrl || '';
    this.defaultHeaders = {
      'Content-Type': 'application/json'
    };
  }
  
  /**
   * Set API key for authenticated requests
   */
  setApiKey(apiKey) {
    this.defaultHeaders['Authorization'] = `Bearer ${apiKey}`;
  }
  
  /**
   * Make a generic request to the API
   */
  async request(endpoint, options = {}) {
    const url = this.baseUrl + endpoint;
    const headers = { ...this.defaultHeaders, ...options.headers };
    
    const config = {
      ...options,
      headers
    };
    
    try {
      const response = await fetch(url, config);
      
      // Handle non-JSON responses
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const data = await response.json();
        
        if (!response.ok) {
          throw new Error(data.detail || 'API request failed');
        }
        
        return data;
      } else {
        if (!response.ok) {
          throw new Error('API request failed');
        }
        
        return await response.text();
      }
    } catch (error) {
      console.error('API request error:', error);
      throw error;
    }
  }
  
  /**
   * Get request
   */
  async get(endpoint, params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const url = queryString ? `${endpoint}?${queryString}` : endpoint;
    
    return this.request(url, { method: 'GET' });
  }
  
  /**
   * Post request
   */
  async post(endpoint, data = {}) {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(data)
    });
  }
  
  /**
   * Put request
   */
  async put(endpoint, data = {}) {
    return this.request(endpoint, {
      method: 'PUT',
      body: JSON.stringify(data)
    });
  }
  
  /**
   * Delete request
   */
  async delete(endpoint) {
    return this.request(endpoint, { method: 'DELETE' });
  }
  
  // Collaboration specific methods
  
  /**
   * List collaboration sessions
   */
  async listSessions(limit = 10, creatorId = null) {
    const params = { limit };
    if (creatorId) {
      params.creator_id = creatorId;
    }
    return this.get('/api/v1/collab-lab/sessions', params);
  }
  
  /**
   * Get session details
   */
  async getSession(sessionId) {
    return this.get(`/api/v1/collab-lab/sessions/${sessionId}`);
  }
  
  /**
   * Create a new collaboration session
   */
  async createSession(name, creatorId, isPublic = true) {
    return this.post('/api/v1/collab-lab/sessions', {
      name,
      creator_id: creatorId,
      is_public: isPublic
    });
  }
  
  /**
   * Generate music
   */
  async generateFullSong(params) {
    return this.post('/api/v1/generate/full-song', params);
  }
}

// Create and export a singleton instance
const apiClient = new ApiClient();
export default apiClient;

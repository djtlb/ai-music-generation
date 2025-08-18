#!/usr/bin/env node

/**
 * Beat Addicts - Integration Script
 * 
 * This script handles the complete integration between:
 * - Frontend (React/Vite) 
 * - Backend (FastAPI)
 * 
 * Features:
 * - Development proxy setup with CORS
 * - WebSocket forwarding
 * - Frontend-backend wiring
 * - Environment configuration
 * - Production build optimization
 */

const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const http = require('http');
const httpProxy = require('http-proxy');
const chalk = require('chalk');
const inquirer = require('inquirer');
const portfinder = require('portfinder');
const dotenv = require('dotenv');
const boxen = require('boxen');
const ora = require('ora');

// Configuration
const DEFAULT_CONFIG = {
  frontendPort: 5173,
  backendPort: 8000,
  backendHost: '127.0.0.1',
  proxyPort: 3000,
  environment: 'development',
  exportDir: 'exports',
  generatedAudioDir: 'generated_audio',
  staticDir: 'static'
};

// State tracking
let frontendProcess = null;
let backendProcess = null;
let proxyServer = null;

/**
 * Load configuration from environment variables and .env files
 */
function loadConfig() {
  // Load .env file if it exists
  if (fs.existsSync('.env')) {
    dotenv.config();
  }
  
  return {
    frontendPort: parseInt(process.env.FRONTEND_PORT || DEFAULT_CONFIG.frontendPort),
    backendPort: parseInt(process.env.BACKEND_PORT || DEFAULT_CONFIG.backendPort),
    backendHost: process.env.BACKEND_HOST || DEFAULT_CONFIG.backendHost,
    proxyPort: parseInt(process.env.PROXY_PORT || DEFAULT_CONFIG.proxyPort),
    environment: process.env.NODE_ENV || DEFAULT_CONFIG.environment,
    exportDir: process.env.EXPORT_DIR || DEFAULT_CONFIG.exportDir,
    generatedAudioDir: process.env.GENERATED_AUDIO_DIR || DEFAULT_CONFIG.generatedAudioDir,
    staticDir: process.env.STATIC_DIR || DEFAULT_CONFIG.staticDir
  };
}

/**
 * Ensure required directories exist
 */
function ensureDirectories(config) {
  const spinner = ora('Ensuring required directories exist...').start();
  const dirs = [
    config.exportDir,
    config.generatedAudioDir,
    config.staticDir
  ];
  
  dirs.forEach(dir => {
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
      spinner.info(`Created directory: ${dir}`);
    }
  });
  
  spinner.succeed('Directory structure verified');
}

/**
 * Create a development proxy that handles API and WebSocket forwarding
 */
function createDevProxy(config) {
  const spinner = ora('Setting up development proxy...').start();
  
  // Create a proxy server for HTTP and WebSocket
  const proxy = httpProxy.createProxyServer({
    ws: true,
    changeOrigin: true
  });
  
  // Handle proxy errors
  proxy.on('error', (err, req, res) => {
    console.error(chalk.red('Proxy error:'), err);
    if (res.writeHead) {
      res.writeHead(500, { 'Content-Type': 'text/plain' });
      res.end('Proxy error occurred');
    }
  });

  // Create the server
  const server = http.createServer((req, res) => {
    // Add CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    
    // Handle preflight OPTIONS request
    if (req.method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }
    
    // API requests to backend
    if (req.url.startsWith('/api/') || req.url.startsWith('/docs') || 
        req.url.startsWith('/redoc') || req.url.startsWith('/openapi.json') ||
        req.url.startsWith('/static/') || req.url.startsWith('/audio/')) {
      // Forward to backend
      proxy.web(req, res, {
        target: `http://${config.backendHost}:${config.backendPort}`
      });
    } else {
      // All other requests to frontend
      proxy.web(req, res, {
        target: `http://localhost:${config.frontendPort}`
      });
    }
  });
  
  // Handle WebSocket connections
  server.on('upgrade', (req, socket, head) => {
    // WebSocket requests to backend
    if (req.url.startsWith('/ws/') || req.url.startsWith('/api/v1/ws/')) {
      proxy.ws(req, socket, head, {
        target: `ws://${config.backendHost}:${config.backendPort}`
      });
    } else {
      // Other WebSocket connections to frontend
      proxy.ws(req, socket, head, {
        target: `ws://localhost:${config.frontendPort}`
      });
    }
  });
  
  // Find available port
  portfinder.basePort = config.proxyPort;
  portfinder.getPort((err, port) => {
    if (err) {
      spinner.fail('Failed to find available port');
      console.error(chalk.red('Port finder error:'), err);
      return;
    }
    
    // Start the server
    server.listen(port, () => {
      spinner.succeed(`Development proxy running at ${chalk.green(`http://localhost:${port}`)}`);
      console.log(boxen(
        `${chalk.bold('Beat Addicts Integration Proxy')}\n\n` +
        `Frontend: ${chalk.cyan(`http://localhost:${config.frontendPort}`)}\n` +
        `Backend: ${chalk.cyan(`http://${config.backendHost}:${config.backendPort}`)}\n` +
        `Proxy: ${chalk.green(`http://localhost:${port}`)}\n\n` +
        `${chalk.yellow('Use the proxy URL for development')}`,
        { padding: 1, borderStyle: 'round', borderColor: 'green' }
      ));
    });
    
    proxyServer = server;
  });
  
  return server;
}

/**
 * Start the frontend development server
 */
function startFrontend(config) {
  const spinner = ora('Starting frontend development server...').start();
  
  frontendProcess = spawn('npm', ['run', 'dev', '--', `--port=${config.frontendPort}`], {
    stdio: 'pipe',
    shell: true
  });
  
  let frontendStarted = false;
  
  frontendProcess.stdout.on('data', (data) => {
    const output = data.toString();
    if (output.includes('Local:') && !frontendStarted) {
      frontendStarted = true;
      spinner.succeed('Frontend development server started');
    }
    console.log(chalk.blue('[Frontend]'), output.trim());
  });
  
  frontendProcess.stderr.on('data', (data) => {
    console.error(chalk.red('[Frontend Error]'), data.toString().trim());
  });
  
  frontendProcess.on('close', (code) => {
    if (code !== 0 && code !== null) {
      spinner.fail(`Frontend process exited with code ${code}`);
    } else {
      console.log(chalk.blue('[Frontend]'), 'Process ended');
    }
  });
  
  return frontendProcess;
}

/**
 * Start the backend server
 */
function startBackend(config) {
  const spinner = ora('Starting backend server...').start();
  
  // Create or update .env file for backend
  const envContent = `
ENVIRONMENT=${config.environment}
EXPORT_DIR=${config.exportDir}
GENERATED_AUDIO_DIR=${config.generatedAudioDir}
STATIC_DIR=${config.staticDir}
HOST=${config.backendHost}
PORT=${config.backendPort}
FRONTEND_URL=http://localhost:${config.proxyPort}
ALLOWED_ORIGINS=http://localhost:${config.proxyPort},http://localhost:${config.frontendPort}
  `.trim();
  
  fs.writeFileSync(path.join('backend', '.env'), envContent);
  
  backendProcess = spawn('python', ['main.py'], {
    stdio: 'pipe',
    shell: true,
    cwd: path.join(process.cwd(), 'backend')
  });
  
  let backendStarted = false;
  
  backendProcess.stdout.on('data', (data) => {
    const output = data.toString();
    if (output.includes('Application startup complete') && !backendStarted) {
      backendStarted = true;
      spinner.succeed('Backend server started');
    }
    console.log(chalk.green('[Backend]'), output.trim());
  });
  
  backendProcess.stderr.on('data', (data) => {
    console.error(chalk.red('[Backend Error]'), data.toString().trim());
  });
  
  backendProcess.on('close', (code) => {
    if (code !== 0 && code !== null) {
      spinner.fail(`Backend process exited with code ${code}`);
    } else {
      console.log(chalk.green('[Backend]'), 'Process ended');
    }
  });
  
  return backendProcess;
}

/**
 * Update package.json scripts
 */
function updatePackageJson() {
  const spinner = ora('Updating package.json scripts...').start();
  
  try {
    const packageJsonPath = path.join(process.cwd(), 'package.json');
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    
    // Add or update scripts
    packageJson.scripts = {
      ...packageJson.scripts,
      'start': 'node connect.js',
      'start:frontend': 'vite',
      'start:backend': 'cd backend && python main.py',
      'build:all': 'npm run build && cd backend && pip install -r requirements.txt',
      'connect': 'node connect.js',
      'setup': 'npm install && cd backend && pip install -r requirements.txt'
    };
    
    fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 4));
    spinner.succeed('package.json updated with integration scripts');
  } catch (error) {
    spinner.fail('Failed to update package.json');
    console.error(chalk.red('Error:'), error);
  }
}

/**
 * Update Vite configuration to add API proxy
 */
function updateViteConfig() {
  const spinner = ora('Updating Vite configuration...').start();
  
  try {
    const viteConfigPath = path.join(process.cwd(), 'vite.config.ts');
    let viteConfig = fs.readFileSync(viteConfigPath, 'utf8');
    
    // Check if proxy is already configured
    if (!viteConfig.includes('server: {')) {
      // Add proxy configuration
      const insertPosition = viteConfig.lastIndexOf('}');
      const proxyConfig = `
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
      '/audio': {
        target: 'http://localhost:8000',
      },
      '/static': {
        target: 'http://localhost:8000',
      }
    },
  },`;
      
      viteConfig = viteConfig.slice(0, insertPosition) + 
                   proxyConfig + 
                   viteConfig.slice(insertPosition);
      
      fs.writeFileSync(viteConfigPath, viteConfig);
      spinner.succeed('Vite configuration updated with API proxy');
    } else {
      spinner.info('Vite proxy already configured, skipping');
    }
  } catch (error) {
    spinner.fail('Failed to update Vite configuration');
    console.error(chalk.red('Error:'), error);
  }
}

/**
 * Create a React WebSocket hook file
 */
function createWebSocketHook() {
  const spinner = ora('Creating WebSocket hook...').start();
  
  const hookContent = `
import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Custom hook for WebSocket connections
 * Provides a simple interface for connecting to WebSockets
 * and handling messages
 */
export function useWebSocket() {
  const [lastMessage, setLastMessage] = useState(null);
  const [readyState, setReadyState] = useState(WebSocket.CLOSED);
  const socketRef = useRef(null);

  const connect = useCallback((url) => {
    // Close existing connection if any
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.close();
    }
    
    // Create new WebSocket connection
    const socket = new WebSocket(url);
    socketRef.current = socket;
    
    // Set up event handlers
    socket.onopen = () => {
      console.log('WebSocket connection established');
      setReadyState(WebSocket.OPEN);
    };
    
    socket.onmessage = (event) => {
      setLastMessage(event);
    };
    
    socket.onclose = () => {
      console.log('WebSocket connection closed');
      setReadyState(WebSocket.CLOSED);
    };
    
    socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setReadyState(WebSocket.CLOSED);
    };
    
    return () => {
      socket.close();
    };
  }, []);
  
  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
  }, []);
  
  const sendMessage = useCallback((message) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(message);
      return true;
    }
    return false;
  }, []);
  
  // Clean up on component unmount
  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);
  
  return {
    connect,
    disconnect,
    sendMessage,
    lastMessage,
    readyState
  };
}

export default useWebSocket;
`;
  
  const hookDir = path.join(process.cwd(), 'src', 'lib');
  const hookPath = path.join(hookDir, 'websocket.js');
  
  // Create directory if it doesn't exist
  if (!fs.existsSync(hookDir)) {
    fs.mkdirSync(hookDir, { recursive: true });
  }
  
  // Write the hook file
  fs.writeFileSync(hookPath, hookContent);
  spinner.succeed(`WebSocket hook created at ${chalk.cyan('src/lib/websocket.js')}`);
}

/**
 * Create an auth hook for user context
 */
function createAuthHook() {
  const spinner = ora('Creating authentication hook...').start();
  
  const hookContent = `
import { createContext, useContext, useState, useEffect } from 'react';

/**
 * User authentication context
 */
const UserContext = createContext(null);

/**
 * User Provider component
 * Manages user authentication state
 */
export function UserProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Load user on mount
  useEffect(() => {
    async function loadUser() {
      try {
        // Check if user is stored in localStorage
        const storedUser = localStorage.getItem('user');
        if (storedUser) {
          setUser(JSON.parse(storedUser));
        } else {
          // For development, create a demo user
          const demoUser = {
            id: 'demo-' + Math.random().toString(36).substring(2, 8),
            name: 'Demo User',
            email: 'demo@example.com',
            role: 'user'
          };
          localStorage.setItem('user', JSON.stringify(demoUser));
          setUser(demoUser);
        }
      } catch (err) {
        console.error('Error loading user:', err);
        setError(err);
      } finally {
        setLoading(false);
      }
    }
    
    loadUser();
  }, []);
  
  /**
   * Sign in a user
   */
  const signIn = async (email, password) => {
    try {
      setLoading(true);
      // In a real app, this would call your API
      // const response = await fetch('/api/v1/auth/login', { ... });
      
      // For now, just create a mock user
      const newUser = {
        id: 'user-' + Math.random().toString(36).substring(2, 8),
        name: email.split('@')[0],
        email,
        role: 'user'
      };
      
      localStorage.setItem('user', JSON.stringify(newUser));
      setUser(newUser);
      return newUser;
    } catch (err) {
      setError(err);
      throw err;
    } finally {
      setLoading(false);
    }
  };
  
  /**
   * Sign out the current user
   */
  const signOut = () => {
    localStorage.removeItem('user');
    setUser(null);
  };
  
  /**
   * Update user profile
   */
  const updateProfile = (updates) => {
    if (!user) return;
    const updatedUser = { ...user, ...updates };
    localStorage.setItem('user', JSON.stringify(updatedUser));
    setUser(updatedUser);
  };
  
  const value = {
    user,
    loading,
    error,
    signIn,
    signOut,
    updateProfile
  };
  
  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
}

/**
 * Hook to use the user context
 */
export function useUser() {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
}
`;
  
  const hookDir = path.join(process.cwd(), 'src', 'lib');
  const hookPath = path.join(hookDir, 'auth.js');
  
  // Create directory if it doesn't exist
  if (!fs.existsSync(hookDir)) {
    fs.mkdirSync(hookDir, { recursive: true });
  }
  
  // Write the hook file
  fs.writeFileSync(hookPath, hookContent);
  spinner.succeed(`Authentication hook created at ${chalk.cyan('src/lib/auth.js')}`);
}

/**
 * Create API client for frontend
 */
function createApiClient() {
  const spinner = ora('Creating API client...').start();
  
  const clientContent = `
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
    this.defaultHeaders['Authorization'] = \`Bearer \${apiKey}\`;
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
    const url = queryString ? \`\${endpoint}?\${queryString}\` : endpoint;
    
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
    return this.get(\`/api/v1/collab-lab/sessions/\${sessionId}\`);
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
`;
  
  const clientDir = path.join(process.cwd(), 'src', 'lib');
  const clientPath = path.join(clientDir, 'api.js');
  
  // Create directory if it doesn't exist
  if (!fs.existsSync(clientDir)) {
    fs.mkdirSync(clientDir, { recursive: true });
  }
  
  // Write the client file
  fs.writeFileSync(clientPath, clientContent);
  spinner.succeed(`API client created at ${chalk.cyan('src/lib/api.js')}`);
}

/**
 * Create constants file for shared values
 */
function createConstantsFile() {
  const spinner = ora('Creating constants file...').start();
  
  const constantsContent = `
/**
 * Application-wide constants
 */

// Genre options for music generation
export const GENRE_OPTIONS = [
  { id: 'rock_punk', name: 'Rock/Punk' },
  { id: 'rnb_ballad', name: 'R&B/Ballad' },
  { id: 'country_pop', name: 'Country/Pop' },
  { id: 'electronic', name: 'Electronic' },
  { id: 'ambient', name: 'Ambient' },
  { id: 'jazz', name: 'Jazz' },
  { id: 'classical', name: 'Classical' },
  { id: 'hiphop', name: 'Hip-Hop' }
];

// WebSocket message types
export const MESSAGE_TYPES = {
  LYRICS_UPDATE: 'lyrics_update',
  GENRE_UPDATE: 'genre_update',
  STYLE_UPDATE: 'style_update',
  CHAT: 'chat',
  USER_JOINED: 'user_joined',
  USER_LEFT: 'user_left',
  GENERATION_STARTED: 'generation_started',
  GENERATION_PROGRESS: 'generation_progress',
  GENERATION_COMPLETE: 'generation_complete',
  GENERATION_ERROR: 'generation_error'
};

// API endpoints
export const API_ENDPOINTS = {
  SESSIONS: '/api/v1/collab-lab/sessions',
  GENERATE: '/api/v1/generate/full-song',
  HEALTH: '/health',
  METRICS: '/metrics'
};

// WebSocket endpoints
export const WS_ENDPOINTS = {
  COLLAB: '/api/v1/ws/collab',
  SYSTEM: '/ws'
};

// Audio file types
export const AUDIO_TYPES = {
  MP3: 'audio/mpeg',
  WAV: 'audio/wav',
  OGG: 'audio/ogg'
};

// Local storage keys
export const STORAGE_KEYS = {
  USER: 'user',
  AUTH_TOKEN: 'auth_token',
  PREFERENCES: 'preferences'
};
`;
  
  const constantsDir = path.join(process.cwd(), 'src', 'lib');
  const constantsPath = path.join(constantsDir, 'constants.js');
  
  // Create directory if it doesn't exist
  if (!fs.existsSync(constantsDir)) {
    fs.mkdirSync(constantsDir, { recursive: true });
  }
  
  // Write the constants file
  fs.writeFileSync(constantsPath, constantsContent);
  spinner.succeed(`Constants file created at ${chalk.cyan('src/lib/constants.js')}`);
}

/**
 * Create a backend .env file
 */
function createBackendEnv(config) {
  const spinner = ora('Creating backend .env file...').start();
  
  const envContent = `
# Beat Addicts Backend Environment Configuration
ENVIRONMENT=${config.environment}
EXPORT_DIR=${config.exportDir}
GENERATED_AUDIO_DIR=${config.generatedAudioDir}
STATIC_DIR=${config.staticDir}
HOST=${config.backendHost}
PORT=${config.backendPort}
FRONTEND_URL=http://localhost:${config.proxyPort}
ALLOWED_ORIGINS=http://localhost:${config.proxyPort},http://localhost:${config.frontendPort}

# CORS Settings
CORS_ALLOW_CREDENTIALS=true

# Database Settings
# Uncomment and configure for production
# DATABASE_URL=postgresql://user:password@localhost:5432/beataddicts

# Redis Settings
# Uncomment and configure for production
# REDIS_URL=redis://localhost:6379/0

# S3 Storage Settings (for audio files)
# Uncomment and configure for production
# S3_BUCKET=beataddicts-audio
# S3_ACCESS_KEY=
# S3_SECRET_KEY=
# S3_REGION=us-east-1

# Security Settings
# Generate a secure key for production
JWT_SECRET=development_secret_key
API_KEY_HEADER=X-API-Key
DEFAULT_API_KEY=dev_api_key_for_testing

# Logging
LOG_LEVEL=INFO
`;
  
  const envPath = path.join(process.cwd(), 'backend', '.env');
  
  // Write the .env file
  fs.writeFileSync(envPath, envContent);
  spinner.succeed(`Backend .env file created at ${chalk.cyan('backend/.env')}`);
}

/**
 * Update the FastAPI main.py CORS configuration
 */
function updateBackendCors() {
  const spinner = ora('Updating backend CORS configuration...').start();
  
  try {
    const mainPyPath = path.join(process.cwd(), 'backend', 'main.py');
    let mainPy = fs.readFileSync(mainPyPath, 'utf8');
    
    // Check if we need to update CORS
    if (mainPy.includes('allow_origins=["*"]')) {
      // Replace with environment-based configuration
      const corsReplacement = `
# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=os.environ.get("CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
    allow_methods=["*"],
    allow_headers=["*"],
)
`;
      
      // Find and replace the CORS middleware
      const corsPattern = /app\.add_middleware\(\s*CORSMiddleware[\s\S]*?\)/;
      mainPy = mainPy.replace(corsPattern, corsReplacement.trim());
      
      fs.writeFileSync(mainPyPath, mainPy);
      spinner.succeed('Backend CORS configuration updated');
    } else {
      spinner.info('Backend CORS already configured properly, skipping');
    }
  } catch (error) {
    spinner.fail('Failed to update backend CORS configuration');
    console.error(chalk.red('Error:'), error);
  }
}

/**
 * Create a combined start script
 */
function createStartScript() {
  const spinner = ora('Creating combined start script...').start();
  
  // Check if we need to install dependencies
  const needsDeps = !fs.existsSync(path.join(process.cwd(), 'node_modules', 'http-proxy'));
  
  if (needsDeps) {
    spinner.text = 'Installing required dependencies...';
    // Install required npm packages
    try {
      exec('npm install --save-dev http-proxy portfinder chalk inquirer dotenv boxen ora', (error) => {
        if (error) {
          spinner.fail('Failed to install dependencies');
          console.error(chalk.red('Error:'), error);
        } else {
          spinner.succeed('Dependencies installed');
          updatePackageJson();
        }
      });
    } catch (error) {
      spinner.fail('Failed to install dependencies');
      console.error(chalk.red('Error:'), error);
    }
  } else {
    updatePackageJson();
    spinner.succeed('Start script created');
  }
}

/**
 * Main function
 */
async function main() {
  console.log(boxen(
    chalk.bold('Beat Addicts - Integration Script') + '\n\n' +
    'This script will set up the complete integration between\n' +
    'the frontend and backend of the Beat Addicts application.',
    { padding: 1, borderStyle: 'double', borderColor: 'green' }
  ));
  
  // Load configuration
  const config = loadConfig();
  
  // Get user input for mode
  const { mode } = await inquirer.prompt([
    {
      type: 'list',
      name: 'mode',
      message: 'What would you like to do?',
      choices: [
        { name: 'Start development environment', value: 'start' },
        { name: 'Set up integration files', value: 'setup' },
        { name: 'Update configuration', value: 'config' },
        { name: 'Exit', value: 'exit' }
      ]
    }
  ]);
  
  if (mode === 'exit') {
    console.log(chalk.blue('Exiting...'));
    return;
  }
  
  if (mode === 'config') {
    // Get configuration options
    const answers = await inquirer.prompt([
      {
        type: 'number',
        name: 'frontendPort',
        message: 'Frontend port:',
        default: config.frontendPort
      },
      {
        type: 'number',
        name: 'backendPort',
        message: 'Backend port:',
        default: config.backendPort
      },
      {
        type: 'number',
        name: 'proxyPort',
        message: 'Development proxy port:',
        default: config.proxyPort
      },
      {
        type: 'input',
        name: 'backendHost',
        message: 'Backend host:',
        default: config.backendHost
      },
      {
        type: 'list',
        name: 'environment',
        message: 'Environment:',
        choices: ['development', 'production', 'testing'],
        default: config.environment
      }
    ]);
    
    // Save configuration to .env
    const envContent = Object.entries(answers)
      .map(([key, value]) => `${key.toUpperCase()}=${value}`)
      .join('\n');
      
    fs.writeFileSync('.env', envContent);
    console.log(chalk.green('Configuration saved to .env file'));
    
    // Update config
    Object.assign(config, answers);
  }
  
  if (mode === 'setup' || mode === 'start') {
    // Ensure directories exist
    ensureDirectories(config);
    
    // Create integration files
    createWebSocketHook();
    createAuthHook();
    createApiClient();
    createConstantsFile();
    createBackendEnv(config);
    updateViteConfig();
    updateBackendCors();
    createStartScript();
    
    console.log(chalk.green('\n✅ Integration setup complete!'));
  }
  
  if (mode === 'start') {
    console.log(chalk.blue('\nStarting development environment...'));
    
    // Start services
    startBackend(config);
    startFrontend(config);
    createDevProxy(config);
    
    // Handle process termination
    process.on('SIGINT', () => {
      console.log(chalk.yellow('\nShutting down...'));
      
      if (frontendProcess) {
        frontendProcess.kill();
      }
      
      if (backendProcess) {
        backendProcess.kill();
      }
      
      if (proxyServer) {
        proxyServer.close();
      }
      
      process.exit(0);
    });
  }
}

// Run the main function
main().catch(error => {
  console.error(chalk.red('Error:'), error);
  process.exit(1);
});

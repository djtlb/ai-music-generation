# Beat Addicts - Integration Guide

This guide explains how the frontend and backend components of the Beat Addicts AI Music Generation system are integrated.

## Architecture Overview

The Beat Addicts application consists of:

1. **Frontend**: A React application built with Vite
2. **Backend**: A FastAPI Python application
3. **Integration Layer**: Connecting scripts and configuration

## Integration Components

### 1. Connection Script (`connect.js`)

The `connect.js` script provides a complete integration solution:

- **Development Proxy**: Routes API requests and WebSocket connections
- **CORS Configuration**: Handles cross-origin resource sharing
- **Environment Management**: Sets up development and production environments
- **Startup Orchestration**: Starts both frontend and backend services

### 2. WebSocket Integration

Real-time communication is essential for the collaborative features:

- **WebSocket Hook**: `src/lib/websocket.js` provides a React hook for WebSocket connections
- **Message Handling**: Standardized message types for collaboration
- **Proxy Forwarding**: WebSocket connections are properly forwarded between frontend and backend

### 3. API Integration

The frontend communicates with the backend through a structured API client:

- **API Client**: `src/lib/api.js` provides a consistent interface to the backend APIs
- **Authentication**: Handles user authentication and API keys
- **Error Handling**: Consistent error handling and reporting

### 4. Authentication Flow

User authentication is handled through:

- **Auth Hook**: `src/lib/auth.js` provides user context and authentication methods
- **Token Management**: Secure storage and transmission of authentication tokens
- **User Context**: React context for accessing user information throughout the application

## Getting Started

### Prerequisites

- Node.js and npm
- Python 3.8+ with pip

### Setup

1. Run the setup script to install all dependencies:

```bash
./setup.sh
```

2. Start the development environment:

```bash
node connect.js
```

Or after the package.json is updated:

```bash
npm run start
```

### Configuration

The connection script will guide you through configuration options, or you can manually edit:

- `.env` file for environment variables
- `backend/.env` for backend-specific configuration

## Development Workflow

1. Start the integrated development environment
2. Access the application at the proxy URL (default: http://localhost:3000)
3. Changes to frontend code will hot-reload
4. Changes to backend code will trigger automatic restart

## Production Deployment

For production deployment:

1. Build the frontend:

```bash
npm run build
```

2. Configure the backend for production in `backend/.env`
3. Deploy the backend using a production ASGI server like Uvicorn with Gunicorn
4. Serve the frontend static files from a CDN or web server

## Troubleshooting

### Common Issues

1. **CORS Errors**: Check the CORS configuration in both frontend and backend
2. **WebSocket Connection Failures**: Ensure proper proxy setup and firewall rules
3. **API 404 Errors**: Check API endpoint paths and routing configuration

### Logs

- Frontend logs are available in the browser console
- Backend logs are available in the terminal output
- Proxy logs show request routing

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Vite Documentation](https://vitejs.dev/)
- [WebSocket API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)

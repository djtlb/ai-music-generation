import { createRoot } from 'react-dom/client'
import { ErrorBoundary } from "react-error-boundary";
// import "@github/spark/spark"

import App from './App.tsx'
import { AuthProvider } from './contexts/AuthContext'
import { ErrorFallback } from './ErrorFallback.tsx'
import { MusicPipelineProvider } from './contexts/MusicPipelineContext.tsx'

import "./main.css"
import "./styles/theme.css"
import "./index.css"

createRoot(document.getElementById('root')!).render(
  <ErrorBoundary FallbackComponent={ErrorFallback}>
    <AuthProvider>
      <MusicPipelineProvider>
        <App />
      </MusicPipelineProvider>
    </AuthProvider>
   </ErrorBoundary>
)

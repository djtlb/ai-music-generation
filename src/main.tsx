import { createRoot } from 'react-dom/client'
import { ErrorBoundary } from "react-error-boundary";
// import "@github/spark/spark"

import App from './App.tsx'
import { ErrorFallback } from './ErrorFallback.tsx'
import { MusicPipelineProvider } from './contexts/MusicPipelineContext.tsx'

import "./main.css"
import "./styles/theme.css"
import "./index.css"

createRoot(document.getElementById('root')!).render(
  <ErrorBoundary FallbackComponent={ErrorFallback}>
    <MusicPipelineProvider>
      <App />
    </MusicPipelineProvider>
   </ErrorBoundary>
)

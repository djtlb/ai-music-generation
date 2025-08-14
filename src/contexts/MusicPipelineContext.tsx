import React, { createContext, useContext, ReactNode } from 'react';
import { useMusicPipeline } from '@/hooks/useMusicPipeline';

// Create the context
const MusicPipelineContext = createContext<ReturnType<typeof useMusicPipeline> | null>(null);

// Provider component
interface MusicPipelineProviderProps {
  children: ReactNode;
}

export function MusicPipelineProvider({ children }: MusicPipelineProviderProps) {
  const pipelineHook = useMusicPipeline();
  
  return (
    <MusicPipelineContext.Provider value={pipelineHook}>
      {children}
    </MusicPipelineContext.Provider>
  );
}

// Hook to use the context
export function useMusicPipelineContext() {
  const context = useContext(MusicPipelineContext);
  if (!context) {
    throw new Error('useMusicPipelineContext must be used within a MusicPipelineProvider');
  }
  return context;
}

// Export both the context and the hook for flexibility
export { MusicPipelineContext };

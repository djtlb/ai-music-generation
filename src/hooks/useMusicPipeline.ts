import { useState, useCallback, useEffect } from 'react';
import { useKV } from '@/hooks/useKV';
import { apiService, WebSocketService, type StyleConfig as APIStyleConfig } from '@/services/api';

export interface StyleConfig {
  genre: string;
  subgenre?: string;
  energy: number; // 0-1
  mood: string;
  tempo: number;
  key: string;
  timeSignature: string;
}

export interface LyricsData {
  id: string;
  title: string;
  theme: string;
  verses: string[];
  chorus: string;
  bridge?: string;
  style: string;
  mood: string;
  createdAt: string;
}

export interface ArrangementData {
  id: string;
  name: string;
  genre: string;
  bpm: number;
  timeSignature: string;
  key: string;
  structure: {
    section: string;
    startBar: number;
    endBar: number;
    duration: number;
  }[];
  totalBars: number;
  totalDuration: number;
  createdAt: string;
}

export interface CompositionData {
  id: string;
  name: string;
  arrangementId?: string;
  lyricsId?: string;
  tracks: {
    name: string;
    instrument: string;
    midiData: any;
    notes: Array<{
      pitch: number;
      startTime: number;
      duration: number;
      velocity: number;
    }>;
  }[];
  tempo: number;
  key: string;
  chordProgression: string[];
  createdAt: string;
}

export interface SoundDesignData {
  id: string;
  name: string;
  compositionId?: string;
  patches: {
    trackName: string;
    synthType: string;
    parameters: Record<string, number>;
    effects: Array<{
      type: string;
      parameters: Record<string, number>;
    }>;
  }[];
  globalEffects: Array<{
    type: string;
    parameters: Record<string, number>;
  }>;
  createdAt: string;
}

export interface MixMasterData {
  id: string;
  name: string;
  soundDesignId?: string;
  compositionId?: string;
  trackSettings: {
    trackName: string;
    volume: number;
    pan: number;
    eq: {
      lowGain: number;
      midGain: number;
      highGain: number;
    };
    compression: {
      threshold: number;
      ratio: number;
      attack: number;
      release: number;
    };
    effects: Array<{
      type: string;
      parameters: Record<string, number>;
    }>;
  }[];
  masterBus: {
    eq: { lowGain: number; midGain: number; highGain: number };
    compression: { threshold: number; ratio: number; attack: number; release: number };
    limiter: { threshold: number; ceiling: number };
  };
  createdAt: string;
}

// Additional types for enhanced pipeline
export type PipelineStage = 'lyrics' | 'arrangement' | 'composition' | 'soundDesign' | 'mixMaster';

export interface Project {
  id: string;
  name: string;
  styleConfig: StyleConfig;
  lyricsId?: string;
  arrangementId?: string;
  compositionId?: string;
  soundDesignId?: string;
  mixMasterId?: string;
  backendProjectId?: string; // Link to backend project
  createdAt?: Date;
  lastModified?: Date;
  lyrics?: LyricsData;
  arrangement?: ArrangementData;
  composition?: CompositionData;
  soundDesign?: SoundDesignData;
  mixMaster?: MixMasterData;
}

export interface PipelineState {
  currentProject: {
    id: string;
    name: string;
    styleConfig: StyleConfig;
    lyricsId?: string;
    arrangementId?: string;
    compositionId?: string;
    soundDesignId?: string;
    mixMasterId?: string;
    backendProjectId?: string; // Link to backend project
  } | null;
  isProcessing: {
    lyrics: boolean;
    arrangement: boolean;
    composition: boolean;
    soundDesign: boolean;
    mixMaster: boolean;
  };
  errors: Record<string, string>;
  isConnected: boolean;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
}

export function useMusicPipeline() {
  // Local storage for all data types
  const [lyrics, setLyrics] = useKV<LyricsData[]>('generated-lyrics', []);
  const [arrangements, setArrangements] = useKV<ArrangementData[]>('song-structures', []);
  const [compositions, setCompositions] = useKV<CompositionData[]>('melody-harmony-compositions', []);
  const [soundDesigns, setSoundDesigns] = useKV<SoundDesignData[]>('sound-designs', []);
  const [mixMasters, setMixMasters] = useKV<MixMasterData[]>('mix-masters', []);
  const [projects, setProjects] = useKV<PipelineState['currentProject'][]>('projects', []);
  
  // WebSocket instance
  const [wsService, setWsService] = useState<WebSocketService | null>(null);
  
  // Additional state for new features
  const [currentProject, setCurrentProject] = useState<Project | null>(null);
  const [isGenerationLoading, setIsGenerationLoading] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(0);
  
  // Pipeline state
  const [pipelineState, setPipelineState] = useState<PipelineState>({
    currentProject: null,
    isProcessing: {
      lyrics: false,
      arrangement: false,
      composition: false,
      soundDesign: false,
      mixMaster: false,
    },
    errors: {},
    isConnected: false,
    connectionStatus: 'disconnected',
  });

  // Initialize WebSocket connection
  useEffect(() => {
    const userId = localStorage.getItem('user_id') || `user_${Date.now()}`;
    localStorage.setItem('user_id', userId);
    
    const ws = new WebSocketService(userId);
    setWsService(ws);
    
    // Connect to WebSocket
    setPipelineState(prev => ({ ...prev, connectionStatus: 'connecting' }));
    
    ws.connect()
      .then(() => {
        setPipelineState(prev => ({ 
          ...prev, 
          isConnected: true, 
          connectionStatus: 'connected' 
        }));
        
        // Subscribe to project updates
        ws.subscribe('project_update', handleProjectUpdate);
        ws.subscribe('task_complete', handleTaskComplete);
        ws.subscribe('generation_progress', handleGenerationProgress);
      })
      .catch((error) => {
        console.error('WebSocket connection failed:', error);
        setPipelineState(prev => ({ 
          ...prev, 
          isConnected: false, 
          connectionStatus: 'error' 
        }));
      });
    
    return () => {
      ws.disconnect();
    };
  }, []);

  // WebSocket event handlers
  const handleProjectUpdate = useCallback((data: any) => {
    console.log('Project update received:', data);
    // Update current project with new data
    if (pipelineState.currentProject?.backendProjectId === data.project_id) {
      // Sync backend project data with local state
      // Implementation would map backend data to local format
    }
  }, [pipelineState.currentProject]);

  const handleTaskComplete = useCallback((data: any) => {
    console.log('Task completed:', data);
    setProcessing(data.stage as keyof PipelineState['isProcessing'], false);
  }, []);

  const handleGenerationProgress = useCallback((data: any) => {
    console.log('Generation progress:', data);
    // Update progress indicators in UI
  }, []);

  // Create new project
  const createProject = useCallback((name: string, styleConfig: StyleConfig) => {
    const newProject = {
      id: `project-${Date.now()}`,
      name,
      styleConfig,
    };
    
    setProjects(prev => [newProject, ...prev]);
    setPipelineState(prev => ({
      ...prev,
      currentProject: newProject,
      errors: {},
    }));
    
    return newProject;
  }, [setProjects]);

  // Load existing project
  const loadProject = useCallback((projectId: string) => {
    const project = projects.find(p => p?.id === projectId);
    if (project) {
      setPipelineState(prev => ({
        ...prev,
        currentProject: project,
        errors: {},
      }));
    }
    return project;
  }, [projects]);

  // Update current project
  const updateCurrentProject = useCallback((updates: Partial<PipelineState['currentProject']>) => {
    if (!pipelineState.currentProject) return;
    
    const updatedProject = { ...pipelineState.currentProject, ...updates };
    
    setProjects(prev => 
      prev.map(p => p?.id === updatedProject.id ? updatedProject : p)
    );
    
    setPipelineState(prev => ({
      ...prev,
      currentProject: updatedProject,
    }));
  }, [pipelineState.currentProject, setProjects]);

  // Add data to specific pipeline stage
  const addLyrics = useCallback((data: Omit<LyricsData, 'id' | 'createdAt'>) => {
    const newLyrics = {
      ...data,
      id: `lyrics-${Date.now()}`,
      createdAt: new Date().toISOString(),
    };
    
    setLyrics(prev => [newLyrics, ...prev]);
    updateCurrentProject({ lyricsId: newLyrics.id });
    
    return newLyrics;
  }, [setLyrics, updateCurrentProject]);

  const addArrangement = useCallback((data: Omit<ArrangementData, 'id' | 'createdAt'>) => {
    const newArrangement = {
      ...data,
      id: `arrangement-${Date.now()}`,
      createdAt: new Date().toISOString(),
    };
    
    setArrangements(prev => [newArrangement, ...prev]);
    updateCurrentProject({ arrangementId: newArrangement.id });
    
    return newArrangement;
  }, [setArrangements, updateCurrentProject]);

  const addComposition = useCallback((data: Omit<CompositionData, 'id' | 'createdAt'>) => {
    const newComposition = {
      ...data,
      id: `composition-${Date.now()}`,
      createdAt: new Date().toISOString(),
    };
    
    setCompositions(prev => [newComposition, ...prev]);
    updateCurrentProject({ compositionId: newComposition.id });
    
    return newComposition;
  }, [setCompositions, updateCurrentProject]);

  const addSoundDesign = useCallback((data: Omit<SoundDesignData, 'id' | 'createdAt'>) => {
    const newSoundDesign = {
      ...data,
      id: `sound-${Date.now()}`,
      createdAt: new Date().toISOString(),
    };
    
    setSoundDesigns(prev => [newSoundDesign, ...prev]);
    updateCurrentProject({ soundDesignId: newSoundDesign.id });
    
    return newSoundDesign;
  }, [setSoundDesigns, updateCurrentProject]);

  const addMixMaster = useCallback((data: Omit<MixMasterData, 'id' | 'createdAt'>) => {
    const newMixMaster = {
      ...data,
      id: `mix-${Date.now()}`,
      createdAt: new Date().toISOString(),
    };
    
    setMixMasters(prev => [newMixMaster, ...prev]);
    updateCurrentProject({ mixMasterId: newMixMaster.id });
    
    return newMixMaster;
  }, [setMixMasters, updateCurrentProject]);

  // Get data by ID
  const getLyricsById = useCallback((id: string) => 
    lyrics.find(l => l.id === id), [lyrics]);
  
  const getArrangementById = useCallback((id: string) => 
    arrangements.find(a => a.id === id), [arrangements]);
  
  const getCompositionById = useCallback((id: string) => 
    compositions.find(c => c.id === id), [compositions]);
  
  const getSoundDesignById = useCallback((id: string) => 
    soundDesigns.find(s => s.id === id), [soundDesigns]);
  
  const getMixMasterById = useCallback((id: string) => 
    mixMasters.find(m => m.id === id), [mixMasters]);

  // Get current project data
  const getCurrentProjectData = useCallback(() => {
    if (!pipelineState.currentProject) return null;
    
    const { currentProject } = pipelineState;
    
    return {
      project: currentProject,
      lyrics: currentProject.lyricsId ? getLyricsById(currentProject.lyricsId) : null,
      arrangement: currentProject.arrangementId ? getArrangementById(currentProject.arrangementId) : null,
      composition: currentProject.compositionId ? getCompositionById(currentProject.compositionId) : null,
      soundDesign: currentProject.soundDesignId ? getSoundDesignById(currentProject.soundDesignId) : null,
      mixMaster: currentProject.mixMasterId ? getMixMasterById(currentProject.mixMasterId) : null,
    };
  }, [pipelineState.currentProject, getLyricsById, getArrangementById, getCompositionById, getSoundDesignById, getMixMasterById]);

  // Process pipeline stage
  const setProcessing = useCallback((stage: keyof PipelineState['isProcessing'], isProcessing: boolean) => {
    setPipelineState(prev => ({
      ...prev,
      isProcessing: {
        ...prev.isProcessing,
        [stage]: isProcessing,
      },
    }));
  }, []);

  // Set error for stage
  const setError = useCallback((stage: string, error: string) => {
    setPipelineState(prev => ({
      ...prev,
      errors: {
        ...prev.errors,
        [stage]: error,
      },
    }));
  }, []);

  // Clear error for stage
  const clearError = useCallback((stage: string) => {
    setPipelineState(prev => ({
      ...prev,
      errors: Object.fromEntries(
        Object.entries(prev.errors).filter(([key]) => key !== stage)
      ),
    }));
  }, []);

  // Auto-advance pipeline when previous stage completes
  const canAdvanceToStage = useCallback((stage: string) => {
    const projectData = getCurrentProjectData();
    if (!projectData) return false;

    switch (stage) {
      case 'arrangement':
        return true; // Can always start with arrangement
      case 'composition':
        return !!projectData.arrangement; // Need arrangement data
      case 'soundDesign':
        return !!projectData.composition; // Need composition data
      case 'mixMaster':
        return !!projectData.soundDesign; // Need sound design data
      default:
        return true;
    }
  }, [getCurrentProjectData]);

  // Full pipeline automation with backend integration
  const runFullPipeline = useCallback(async (
    projectName: string,
    styleConfig: StyleConfig,
    options: {
      lyricsTheme?: string;
      autoAdvance?: boolean;
    } = {}
  ) => {
    try {
      // Create local project first
      const localProject = createProject(projectName, styleConfig);
      
      // Convert styleConfig to API format
      const apiStyleConfig: APIStyleConfig = {
        genre: styleConfig.genre,
        subgenre: styleConfig.subgenre,
        energy: styleConfig.energy,
        mood: styleConfig.mood,
        tempo: styleConfig.tempo,
        key: styleConfig.key,
        timeSignature: styleConfig.timeSignature,
      };

      // Prepare API request
      const fullSongRequest = {
        project_name: projectName,
        style_config: apiStyleConfig,
        lyrics_request: options.lyricsTheme ? {
          theme: options.lyricsTheme,
          style: styleConfig.genre,
          mood: styleConfig.mood,
          language: 'english',
          explicit: false,
        } : undefined,
        advanced_options: {},
        collaboration_enabled: false,
      };

      // Call backend API
      const response = await apiService.generateFullSong(fullSongRequest);
      
      if (response.success && response.data) {
        // Update local project with backend project ID
        updateCurrentProject({ 
          backendProjectId: response.data.project_id 
        });
        
        // Set all stages as processing
        setPipelineState(prev => ({
          ...prev,
          isProcessing: {
            lyrics: true,
            arrangement: true,
            composition: true,
            soundDesign: true,
            mixMaster: true,
          }
        }));

        console.log('🎵 Million-dollar song generation started!', response.data);
        
        // Start polling for project status
        pollProjectStatus(response.data.project_id);
        
        return localProject;
      } else {
        throw new Error(response.error || 'Failed to start song generation');
      }
      
    } catch (error) {
      console.error('Pipeline error:', error);
      setError('pipeline', error instanceof Error ? error.message : 'Unknown error');
      throw error;
    }
  }, [createProject, updateCurrentProject, setError]);

  // Poll project status for updates
  const pollProjectStatus = useCallback(async (backendProjectId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await apiService.getProjectStatus(backendProjectId);
        
        if (response.success && response.data) {
          const { status, stages } = response.data;
          
          // Update processing states based on backend status
          if (stages) {
            setPipelineState(prev => ({
              ...prev,
              isProcessing: {
                lyrics: !stages.lyrics,
                arrangement: !stages.arrangement,
                composition: !stages.composition,
                soundDesign: !stages.sound_design,
                mixMaster: !stages.mix_master,
              }
            }));
            
            // Update local data with backend results
            if (stages.lyrics && !getCurrentProjectData()?.lyrics) {
              addLyrics({
                title: stages.lyrics.title || 'AI Generated Song',
                theme: stages.lyrics.theme || 'AI Music',
                verses: stages.lyrics.verses || [],
                chorus: stages.lyrics.chorus || '',
                bridge: stages.lyrics.bridge,
                style: stages.lyrics.style || 'pop',
                mood: stages.lyrics.mood || 'uplifting',
              });
            }
            
            if (stages.arrangement && !getCurrentProjectData()?.arrangement) {
              addArrangement({
                name: stages.arrangement.name || 'AI Arrangement',
                genre: stages.arrangement.genre || 'pop',
                bpm: stages.arrangement.bpm || 120,
                timeSignature: stages.arrangement.time_signature || '4/4',
                key: stages.arrangement.key || 'C',
                structure: stages.arrangement.structure || [],
                totalBars: stages.arrangement.total_bars || 64,
                totalDuration: stages.arrangement.duration || 180,
              });
            }
            
            // Similar updates for other stages...
          }
          
          // Stop polling when complete or failed
          if (status === 'completed' || status === 'failed') {
            clearInterval(pollInterval);
            
            if (status === 'completed') {
              setPipelineState(prev => ({
                ...prev,
                isProcessing: {
                  lyrics: false,
                  arrangement: false,
                  composition: false,
                  soundDesign: false,
                  mixMaster: false,
                }
              }));
              
              console.log('🎉 Million-dollar song completed!');
            } else if (status === 'failed') {
              setError('pipeline', response.data.error || 'Song generation failed');
            }
          }
        }
      } catch (error) {
        console.error('Error polling project status:', error);
      }
    }, 2000); // Poll every 2 seconds
    
    // Clear interval after 10 minutes to prevent infinite polling
    setTimeout(() => clearInterval(pollInterval), 10 * 60 * 1000);
  }, [getCurrentProjectData, addLyrics, addArrangement, setError]);

  // Update functions for pipeline stages
  const updateLyrics = useCallback((updates: Partial<LyricsData>) => {
    if (!currentProject) return;
    
    const projectData = getCurrentProjectData();
    if (!projectData?.lyrics) return;
    
    const updatedLyrics = { ...projectData.lyrics, ...updates };
    // Update the lyrics in the separate lyrics array
    setLyrics(prev => prev.map(l => l.id === projectData.lyrics?.id ? updatedLyrics : l));
  }, [currentProject, getCurrentProjectData]);

  const updateArrangement = useCallback((updates: Partial<ArrangementData>) => {
    if (!currentProject) return;
    
    const projectData = getCurrentProjectData();
    if (!projectData?.arrangement) return;
    
    const updatedArrangement = { ...projectData.arrangement, ...updates };
    // Update the arrangement in the separate arrangements array
    setArrangements(prev => prev.map(a => a.id === projectData.arrangement?.id ? updatedArrangement : a));
  }, [currentProject, getCurrentProjectData]);

  const updateComposition = useCallback((updates: Partial<CompositionData>) => {
    if (!currentProject) return;
    
    const projectData = getCurrentProjectData();
    if (!projectData?.composition) return;
    
    const updatedComposition = { ...projectData.composition, ...updates };
    // Update the composition in the separate compositions array
    setCompositions(prev => prev.map(c => c.id === projectData.composition?.id ? updatedComposition : c));
  }, [currentProject, getCurrentProjectData]);

  const updateSoundDesign = useCallback((updates: Partial<SoundDesignData>) => {
    if (!currentProject) return;
    
    const projectData = getCurrentProjectData();
    if (!projectData?.soundDesign) return;
    
    const updatedSoundDesign = { ...projectData.soundDesign, ...updates };
    // Update the sound design in the separate soundDesigns array
    setSoundDesigns(prev => prev.map(s => s.id === projectData.soundDesign?.id ? updatedSoundDesign : s));
  }, [currentProject, getCurrentProjectData]);

  const updateMixMaster = useCallback((updates: Partial<MixMasterData>) => {
    if (!currentProject) return;
    
    const projectData = getCurrentProjectData();
    if (!projectData?.mixMaster) return;
    
    const updatedMixMaster = { ...projectData.mixMaster, ...updates };
    // Update the mix master in the separate mixMasters array
    setMixMasters(prev => prev.map(m => m.id === projectData.mixMaster?.id ? updatedMixMaster : m));
  }, [currentProject, getCurrentProjectData]);

  // Enhanced pipeline functions
  const triggerGeneration = useCallback(async (stage: PipelineStage, params: any = {}) => {
    setIsGenerationLoading(true);
    setGenerationProgress(0);
    
    try {
      let response;
      
      switch (stage) {
        case 'lyrics':
          response = await apiService.generateLyrics(params);
          break;
        case 'arrangement':
          response = await apiService.generateArrangement(params);
          break;
        case 'composition':
          // Use basic generate endpoint since specific ones don't exist yet
          response = await apiService.generateFullSong(params);
          break;
        case 'soundDesign':
          response = await apiService.generateFullSong(params);
          break;
        case 'mixMaster':
          response = await apiService.generateFullSong(params);
          break;
        default:
          throw new Error(`Unknown stage: ${stage}`);
      }
      
      if (response.success && response.data) {
        setGenerationProgress(100);
        return response.data;
      } else {
        throw new Error(response.error || `${stage} generation failed`);
      }
    } catch (error) {
      setError(stage, error instanceof Error ? error.message : 'Unknown error');
      throw error;
    } finally {
      setIsGenerationLoading(false);
    }
  }, [setError]);

  const exportProject = useCallback(async (format: 'json' | 'midi' | 'wav' | 'mp3' = 'json') => {
    if (!currentProject) {
      throw new Error('No project selected');
    }
    
    try {
      // For now, export local project data as JSON
      if (format === 'json') {
        const projectData = getCurrentProjectData();
        const exportData = {
          project: currentProject,
          data: projectData,
          exportedAt: new Date().toISOString(),
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${currentProject.name}-export.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        return exportData;
      } else {
        // For audio exports, would integrate with backend API
        throw new Error(`Export format ${format} not yet implemented`);
      }
    } catch (error) {
      console.error('Export error:', error);
      throw error;
    }
  }, [currentProject, getCurrentProjectData]);

  // Health check for backend connection
  const checkBackendHealth = useCallback(async () => {
    try {
      const response = await apiService.healthCheck();
      return response.success;
    } catch (error) {
      console.error('Backend health check failed:', error);
      return false;
    }
  }, []);

  return {
    // State
    projects,
    currentProject,
    pipelineState,
    isGenerationLoading,
    generationProgress,

    // Core project management
    createProject,
    deleteProject: (projectId: string) => {
      setProjects(prev => prev.filter(p => p?.id !== projectId));
      if (currentProject?.id === projectId) {
        setCurrentProject(null);
      }
    },
    setCurrentProject,
    updateCurrentProject,
    getCurrentProjectData,
    loadProject,

    // Legacy data access (for backward compatibility)
    lyrics,
    arrangements,
    compositions,
    soundDesigns,
    mixMasters,
    currentProjectData: getCurrentProjectData(),

    // Pipeline stage management
    addLyrics,
    updateLyrics,
    addArrangement,
    updateArrangement,
    addComposition,
    updateComposition,
    addSoundDesign,
    updateSoundDesign,
    addMixMaster,
    updateMixMaster,

    // Get data by ID (legacy support)
    getLyricsById,
    getArrangementById,
    getCompositionById,
    getSoundDesignById,
    getMixMasterById,

    // Generation and processing
    runFullPipeline,
    triggerGeneration,
    exportProject,
    
    // Backend integration
    checkBackendHealth,
    pollProjectStatus,
    
    // WebSocket connection status  
    isConnected: pipelineState.isConnected || false,
    
    // AI model interfaces
    generateLyrics: async (params: any) => {
      try {
        const response = await apiService.generateLyrics(params);
        if (response.success && response.data) {
          return response.data;
        }
        throw new Error(response.error || 'Lyrics generation failed');
      } catch (error) {
        setError('lyrics', error instanceof Error ? error.message : 'Unknown error');
        throw error;
      }
    },
    
    generateArrangement: async (params: any) => {
      try {
        const response = await apiService.generateArrangement(params);
        if (response.success && response.data) {
          return response.data;
        }
        throw new Error(response.error || 'Arrangement generation failed');
      } catch (error) {
        setError('arrangement', error instanceof Error ? error.message : 'Unknown error');
        throw error;
      }
    },
    
    generateComposition: async (params: any) => {
      try {
        // Use full song generation for now since specific endpoint doesn't exist
        const response = await apiService.generateFullSong(params);
        if (response.success && response.data) {
          return response.data;
        }
        throw new Error(response.error || 'Composition generation failed');
      } catch (error) {
        setError('composition', error instanceof Error ? error.message : 'Unknown error');
        throw error;
      }
    },
    
    generateSoundDesign: async (params: any) => {
      try {
        // Use full song generation for now since specific endpoint doesn't exist
        const response = await apiService.generateFullSong(params);
        if (response.success && response.data) {
          return response.data;
        }
        throw new Error(response.error || 'Sound design generation failed');
      } catch (error) {
        setError('soundDesign', error instanceof Error ? error.message : 'Unknown error');
        throw error;
      }
    },
    
    generateMixMaster: async (params: any) => {
      try {
        // Use full song generation for now since specific endpoint doesn't exist
        const response = await apiService.generateFullSong(params);
        if (response.success && response.data) {
          return response.data;
        }
        throw new Error(response.error || 'Mix/master generation failed');
      } catch (error) {
        setError('mixMaster', error instanceof Error ? error.message : 'Unknown error');
        throw error;
      }
    },

    // Pipeline control (legacy and new)
    setProcessing,
    setError,
    clearError,
    clearAllErrors: () => {
      setPipelineState(prev => ({
        ...prev,
        errors: {},
      }));
    },
    canAdvanceToStage,

    // Error handling
    errors: pipelineState.errors,

    // Utility functions
    getStageProgress: (stage: PipelineStage) => {
      if (!currentProject) return 0;
      
      const data = getCurrentProjectData();
      if (!data) return 0;

      switch (stage) {
        case 'lyrics':
          return data.lyrics ? 100 : 0;
        case 'arrangement':
          return data.arrangement ? 100 : 0;
        case 'composition':
          return data.composition ? 100 : 0;
        case 'soundDesign':
          return data.soundDesign ? 100 : 0;
        case 'mixMaster':
          return data.mixMaster ? 100 : 0;
        default:
          return 0;
      }
    },

    // Million-dollar platform features
    subscriptionTier: 'free' as const, // Would come from auth context
    creditsRemaining: 100, // Would come from backend
    isPremiumFeature: (feature: string) => {
      // Premium features for million-dollar platform
      const premiumFeatures = [
        'advanced_mixing',
        'ai_mastering',
        'stem_separation',
        'collaboration',
        'nft_minting',
        'royalty_tracking',
        'unlimited_exports',
      ];
      return premiumFeatures.includes(feature);
    },
  };
};

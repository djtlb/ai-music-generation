import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { useMusicPipeline } from '@/hooks/useMusicPipeline';
import { 
  Music2, 
  FileText, 
  Layers, 
  Volume2, 
  Sliders,
  CheckCircle2, 
  Clock, 
  AlertCircle,
  Play,
  Square,
  RefreshCw,
  Wand2
} from 'lucide-react';

interface PipelineStage {
  key: string;
  label: string;
  icon: React.ReactNode;
  description: string;
}

const PIPELINE_STAGES: PipelineStage[] = [
  {
    key: 'lyrics',
    label: 'Lyrics Generation',
    icon: <FileText className="h-4 w-4" />,
    description: 'Generate thematic lyrics with verse, chorus, and bridge structures'
  },
  {
    key: 'arrangement',
    label: 'Song Arrangement',
    icon: <Layers className="h-4 w-4" />,
    description: 'Create song structure with tempo, key, and section timing'
  },
  {
    key: 'composition',
    label: 'Melody & Harmony',
    icon: <Music2 className="h-4 w-4" />,
    description: 'Generate instrumental tracks and chord progressions'
  },
  {
    key: 'soundDesign',
    label: 'Sound Design',
    icon: <Volume2 className="h-4 w-4" />,
    description: 'Apply synthesizer patches and audio effects'
  },
  {
    key: 'mixMaster',
    label: 'Mix & Master',
    icon: <Sliders className="h-4 w-4" />,
    description: 'Balance levels, EQ, and final master processing'
  }
];

export function PipelineStatusMonitor() {
  const { 
    pipelineState, 
    currentProjectData,
    setProcessing,
    clearError 
  } = useMusicPipeline();

  const getStageStatus = (stage: string) => {
    const projectData = currentProjectData;
    if (!projectData) return 'disabled';
    
    const hasError = stage in pipelineState.errors;
    if (hasError) return 'error';
    
    const isProcessing = pipelineState.isProcessing[stage as keyof typeof pipelineState.isProcessing];
    if (isProcessing) return 'processing';
    
    switch (stage) {
      case 'lyrics':
        return projectData.lyrics ? 'completed' : 'ready';
      case 'arrangement':
        return projectData.arrangement ? 'completed' : 'ready';
      case 'composition':
        return projectData.composition ? 'completed' : (projectData.arrangement ? 'ready' : 'waiting');
      case 'soundDesign':
        return projectData.soundDesign ? 'completed' : (projectData.composition ? 'ready' : 'waiting');
      case 'mixMaster':
        return projectData.mixMaster ? 'completed' : (projectData.soundDesign ? 'ready' : 'waiting');
      default:
        return 'waiting';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'processing':
        return <Clock className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'ready':
        return <div className="h-4 w-4 rounded-full bg-blue-500 border-2 border-blue-600" />;
      case 'waiting':
        return <div className="h-4 w-4 rounded-full border-2 border-gray-300" />;
      case 'disabled':
        return <div className="h-4 w-4 rounded-full border-2 border-gray-200 opacity-50" />;
      default:
        return <div className="h-4 w-4 rounded-full border-2 border-gray-300" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge variant="default" className="bg-green-500">Complete</Badge>;
      case 'processing':
        return <Badge variant="default" className="bg-blue-500">Processing...</Badge>;
      case 'error':
        return <Badge variant="destructive">Error</Badge>;
      case 'ready':
        return <Badge variant="outline" className="border-blue-500 text-blue-600">Ready</Badge>;
      case 'waiting':
        return <Badge variant="secondary">Waiting</Badge>;
      case 'disabled':
        return <Badge variant="secondary" className="opacity-50">No Project</Badge>;
      default:
        return <Badge variant="secondary">Unknown</Badge>;
    }
  };

  const calculateOverallProgress = () => {
    if (!currentProjectData) return 0;
    
    const statuses = PIPELINE_STAGES.map(stage => getStageStatus(stage.key));
    const completedCount = statuses.filter(status => status === 'completed').length;
    
    return (completedCount / PIPELINE_STAGES.length) * 100;
  };

  const getNextReadyStage = () => {
    return PIPELINE_STAGES.find(stage => getStageStatus(stage.key) === 'ready');
  };

  const handleRetryStage = (stageKey: string) => {
    clearError(stageKey);
    // In a real implementation, this would trigger the stage processing
    console.log(`Retrying stage: ${stageKey}`);
  };

  const handleRunStage = (stageKey: string) => {
    setProcessing(stageKey as any, true);
    // In a real implementation, this would trigger the actual AI processing
    console.log(`Running stage: ${stageKey}`);
    
    // Simulate processing time
    setTimeout(() => {
      setProcessing(stageKey as any, false);
    }, 3000);
  };

  if (!currentProjectData) {
    return (
      <Card className="opacity-60">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Music2 className="h-5 w-5" />
            Pipeline Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <Music2 className="mx-auto h-12 w-12 mb-4 opacity-50" />
            <p>No active project</p>
            <p className="text-sm">Create or load a project to see pipeline status</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const overallProgress = calculateOverallProgress();
  const nextStage = getNextReadyStage();
  const hasErrors = Object.keys(pipelineState.errors).length > 0;
  const isAnyProcessing = Object.values(pipelineState.isProcessing).some(Boolean);

  return (
    <div className="space-y-4">
      {/* Overall Progress */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Music2 className="h-5 w-5" />
              Pipeline Progress
            </CardTitle>
            <div className="text-right">
              <div className="text-2xl font-bold">{Math.round(overallProgress)}%</div>
              <div className="text-sm text-muted-foreground">Complete</div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Progress value={overallProgress} className="w-full h-2" />
          
          {/* Quick Actions */}
          <div className="flex items-center gap-2 mt-4">
            {nextStage && !isAnyProcessing && (
              <Button 
                size="sm" 
                onClick={() => handleRunStage(nextStage.key)}
                className="flex items-center gap-2"
              >
                <Wand2 className="h-4 w-4" />
                Run {nextStage.label}
              </Button>
            )}
            
            {hasErrors && (
              <Button 
                size="sm" 
                variant="outline"
                onClick={() => Object.keys(pipelineState.errors).forEach(clearError)}
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Clear Errors
              </Button>
            )}
            
            {isAnyProcessing && (
              <Badge variant="default" className="bg-blue-500">
                <Clock className="h-4 w-4 mr-1 animate-spin" />
                Processing...
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Detailed Stage Status */}
      <Card>
        <CardHeader>
          <CardTitle>Stage Details</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {PIPELINE_STAGES.map((stage, index) => {
              const status = getStageStatus(stage.key);
              const error = pipelineState.errors[stage.key];
              const isProcessing = pipelineState.isProcessing[stage.key as keyof typeof pipelineState.isProcessing];
              
              return (
                <div key={stage.key} className="relative">
                  <div className={`
                    border rounded-lg p-4 transition-all
                    ${status === 'ready' ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/20' : ''}
                    ${status === 'processing' ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/20 animate-pulse' : ''}
                    ${status === 'completed' ? 'border-green-500 bg-green-50 dark:bg-green-950/20' : ''}
                    ${status === 'error' ? 'border-red-500 bg-red-50 dark:bg-red-950/20' : ''}
                    ${status === 'waiting' || status === 'disabled' ? 'border-gray-200 opacity-60' : ''}
                  `}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="flex items-center gap-2">
                          {stage.icon}
                          {getStatusIcon(status)}
                        </div>
                        <div>
                          <h3 className="font-medium">{stage.label}</h3>
                          <p className="text-sm text-muted-foreground">{stage.description}</p>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-2">
                        {getStatusBadge(status)}
                        
                        {status === 'ready' && !isProcessing && (
                          <Button size="sm" onClick={() => handleRunStage(stage.key)}>
                            <Play className="h-4 w-4" />
                          </Button>
                        )}
                        
                        {status === 'error' && (
                          <Button 
                            size="sm" 
                            variant="outline"
                            onClick={() => handleRetryStage(stage.key)}
                          >
                            <RefreshCw className="h-4 w-4" />
                          </Button>
                        )}
                        
                        {isProcessing && (
                          <Button size="sm" variant="outline" disabled>
                            <Square className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </div>
                    
                    {error && (
                      <div className="mt-3 p-2 bg-red-100 dark:bg-red-900/20 rounded text-sm text-red-600 dark:text-red-400">
                        <strong>Error:</strong> {error}
                      </div>
                    )}
                    
                    {isProcessing && (
                      <div className="mt-3">
                        <Progress value={undefined} className="h-1" />
                        <p className="text-xs text-muted-foreground mt-1">
                          AI is processing this stage...
                        </p>
                      </div>
                    )}
                  </div>
                  
                  {/* Connection line to next stage */}
                  {index < PIPELINE_STAGES.length - 1 && (
                    <div className="flex justify-center my-2">
                      <div className={`
                        w-0.5 h-4 rounded
                        ${status === 'completed' ? 'bg-green-500' : 'bg-gray-300'}
                      `} />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

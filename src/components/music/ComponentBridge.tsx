import React from 'react';
import { useMusicPipelineContext } from '@/contexts/MusicPipelineContext';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ArrowRight, Music, Wand2, CheckCircle2 } from 'lucide-react';

interface ComponentBridgeProps {
  component: React.ComponentType<any>;
  stageKey: 'lyrics' | 'arrangement' | 'composition' | 'soundDesign' | 'mixMaster';
  title: string;
  description: string;
  icon: React.ReactNode;
  canGenerate?: boolean;
  generateHandler?: () => Promise<void>;
}

export function ComponentBridge({
  component: Component,
  stageKey,
  title,
  description,
  icon,
  canGenerate = false,
  generateHandler
}: ComponentBridgeProps) {
  const { currentProjectData, pipelineState } = useMusicPipelineContext();
  
  const stageData = currentProjectData?.[stageKey];
  const isProcessing = pipelineState.isProcessing[stageKey];
  const hasError = stageKey in pipelineState.errors;
  
  const handleGenerate = async () => {
    if (generateHandler) {
      await generateHandler();
    }
  };

  return (
    <div className="space-y-4">
      {/* Integration Status */}
      {currentProjectData && (
        <Card className="border-primary/20">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {icon}
                <div>
                  <CardTitle className="text-lg">{title} Integration</CardTitle>
                  <CardDescription>
                    {currentProjectData.project.name} • {stageKey} stage
                  </CardDescription>
                </div>
              </div>
              
              <div className="flex items-center gap-2">
                {stageData ? (
                  <Badge variant="default" className="bg-green-500">
                    <CheckCircle2 className="w-3 h-3 mr-1" />
                    Connected
                  </Badge>
                ) : (
                  <Badge variant="outline">
                    Not Generated
                  </Badge>
                )}
                
                {canGenerate && !stageData && !isProcessing && (
                  <Button size="sm" onClick={handleGenerate}>
                    <Wand2 className="w-4 h-4 mr-2" />
                    Generate for Project
                  </Button>
                )}
              </div>
            </div>
          </CardHeader>
          
          {stageData && (
            <CardContent className="pt-0">
              <div className="flex items-center gap-4 text-sm text-muted-foreground">
                <div>
                  <span className="font-medium">ID:</span> {stageData.id}
                </div>
                <ArrowRight className="w-4 h-4" />
                <div>
                  <span className="font-medium">Created:</span> {new Date(stageData.createdAt).toLocaleDateString()}
                </div>
                <ArrowRight className="w-4 h-4" />
                <div>
                  <span className="font-medium">Status:</span> Synced with Pipeline
                </div>
              </div>
            </CardContent>
          )}
        </Card>
      )}
      
      {/* Original Component */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {icon}
            {title}
          </CardTitle>
          <CardDescription>
            {description}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Component />
        </CardContent>
      </Card>
    </div>
  );
}

// Higher-order component to wrap existing components
export function withPipelineIntegration<P extends object>(
  Component: React.ComponentType<P>,
  config: Omit<ComponentBridgeProps, 'component'>
) {
  return function IntegratedComponent(props: P) {
    return (
      <ComponentBridge
        component={() => <Component {...props} />}
        {...config}
      />
    );
  };
}

// Individual bridges for each component type
export const LyricsGeneratorBridge = withPipelineIntegration(
  ({ children }: { children: React.ReactNode }) => <>{children}</>,
  {
    stageKey: 'lyrics',
    title: 'Lyrics Generator',
    description: 'Generate thematic lyrics with verse, chorus, and bridge structures',
    icon: <Music className="w-5 h-5" />,
    canGenerate: true,
  }
);

export const ArrangementBridge = withPipelineIntegration(
  ({ children }: { children: React.ReactNode }) => <>{children}</>,
  {
    stageKey: 'arrangement',
    title: 'Song Arrangement',
    description: 'Create song structure with tempo, key, and section timing',
    icon: <Music className="w-5 h-5" />,
    canGenerate: true,
  }
);

export const CompositionBridge = withPipelineIntegration(
  ({ children }: { children: React.ReactNode }) => <>{children}</>,
  {
    stageKey: 'composition',
    title: 'Melody & Harmony',
    description: 'Generate instrumental tracks and chord progressions',
    icon: <Music className="w-5 h-5" />,
    canGenerate: true,
  }
);

export const SoundDesignBridge = withPipelineIntegration(
  ({ children }: { children: React.ReactNode }) => <>{children}</>,
  {
    stageKey: 'soundDesign',
    title: 'Sound Design',
    description: 'Apply synthesizer patches and audio effects',
    icon: <Music className="w-5 h-5" />,
    canGenerate: true,
  }
);

export const MixMasterBridge = withPipelineIntegration(
  ({ children }: { children: React.ReactNode }) => <>{children}</>,
  {
    stageKey: 'mixMaster',
    title: 'Mix & Master',
    description: 'Balance levels, EQ, and final master processing',
    icon: <Music className="w-5 h-5" />,
    canGenerate: true,
  }
);

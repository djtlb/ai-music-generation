import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { 
  Activity, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Timer,
  Cpu,
  HardDrive,
  Waveform,
  BarChart3
} from "@phosphor-icons/react";

interface ProgressStep {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  progress: number;
  duration?: number;
  estimatedDuration?: number;
  throughput?: string;
  metrics?: {
    processedItems?: number;
    totalItems?: number;
    rate?: number;
  };
}

interface SystemMetrics {
  cpu: number;
  memory: number;
  disk: number;
  network: number;
}

interface ProgressVisualizationProps {
  steps: ProgressStep[];
  currentStep: number;
  totalProgress: number;
  systemMetrics: SystemMetrics;
  isRunning: boolean;
  elapsed: number;
  estimated: number;
}

export function ProgressVisualization({ 
  steps, 
  currentStep, 
  totalProgress, 
  systemMetrics, 
  isRunning, 
  elapsed, 
  estimated 
}: ProgressVisualizationProps) {
  const [animatedProgress, setAnimatedProgress] = useState(0);

  // Smooth progress animation
  useEffect(() => {
    const interval = setInterval(() => {
      setAnimatedProgress(prev => {
        const diff = totalProgress - prev;
        return prev + diff * 0.1;
      });
    }, 50);

    return () => clearInterval(interval);
  }, [totalProgress]);

  const formatTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getStepIcon = (status: ProgressStep['status'], isActive: boolean) => {
    const baseClasses = "w-5 h-5";
    
    switch (status) {
      case 'running': 
        return <Activity className={`${baseClasses} text-blue-500 ${isActive ? 'animate-spin' : ''}`} />;
      case 'completed': 
        return <CheckCircle className={`${baseClasses} text-green-500`} />;
      case 'error': 
        return <XCircle className={`${baseClasses} text-red-500`} />;
      default: 
        return <div className={`${baseClasses} bg-gray-300 rounded-full`} />;
    }
  };

  const getMetricColor = (value: number, thresholds = { warning: 70, critical: 90 }) => {
    if (value >= thresholds.critical) return "text-red-500";
    if (value >= thresholds.warning) return "text-yellow-500";
    return "text-green-500";
  };

  return (
    <div className="space-y-6">
      {/* Overall Progress Header */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center justify-between">
            <span>Pipeline Progress</span>
            <Badge variant={isRunning ? "default" : "secondary"}>
              {isRunning ? "Running" : "Idle"}
            </Badge>
          </CardTitle>
          <CardDescription>
            Real-time visualization of AI music generation pipeline
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Main Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Overall Progress</span>
              <span>{Math.round(animatedProgress)}%</span>
            </div>
            <div className="relative">
              <Progress value={animatedProgress} className="h-3" />
              {isRunning && (
                <div 
                  className="absolute top-0 left-0 h-3 bg-gradient-to-r from-transparent via-white to-transparent opacity-30 animate-pulse"
                  style={{ width: '20%', transform: `translateX(${(animatedProgress / 100) * 400}%)` }}
                />
              )}
            </div>
          </div>

          {/* Time Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="flex items-center justify-center gap-2 text-muted-foreground mb-1">
                <Timer className="w-4 h-4" />
                <span className="text-sm">Elapsed</span>
              </div>
              <div className="text-lg font-mono">{formatTime(elapsed)}</div>
            </div>
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="flex items-center justify-center gap-2 text-muted-foreground mb-1">
                <Clock className="w-4 h-4" />
                <span className="text-sm">Remaining</span>
              </div>
              <div className="text-lg font-mono">{formatTime(estimated)}</div>
            </div>
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="flex items-center justify-center gap-2 text-muted-foreground mb-1">
                <Activity className="w-4 h-4" />
                <span className="text-sm">Step</span>
              </div>
              <div className="text-lg font-mono">{currentStep + 1}/{steps.length}</div>
            </div>
            <div className="text-center p-3 bg-muted rounded-lg">
              <div className="flex items-center justify-center gap-2 text-muted-foreground mb-1">
                <BarChart3 className="w-4 h-4" />
                <span className="text-sm">ETA</span>
              </div>
              <div className="text-lg font-mono">
                {elapsed > 0 ? formatTime(elapsed + estimated) : '--:--'}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Resources */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <Cpu className="w-5 h-5" />
            System Resources
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>CPU</span>
                <span className={getMetricColor(systemMetrics.cpu)}>{systemMetrics.cpu.toFixed(0)}%</span>
              </div>
              <Progress value={systemMetrics.cpu} className="h-2" />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Memory</span>
                <span className={getMetricColor(systemMetrics.memory)}>{systemMetrics.memory.toFixed(0)}%</span>
              </div>
              <Progress value={systemMetrics.memory} className="h-2" />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Disk I/O</span>
                <span className={getMetricColor(systemMetrics.disk, { warning: 80, critical: 95 })}>
                  {systemMetrics.disk.toFixed(0)}%
                </span>
              </div>
              <Progress value={systemMetrics.disk} className="h-2" />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Network</span>
                <span className={getMetricColor(systemMetrics.network, { warning: 80, critical: 95 })}>
                  {systemMetrics.network.toFixed(0)}%
                </span>
              </div>
              <Progress value={systemMetrics.network} className="h-2" />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detailed Step Progress */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <Waveform className="w-5 h-5" />
            Step Details
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {steps.map((step, index) => {
            const isActive = index === currentStep && isRunning;
            const isCompleted = step.status === 'completed';
            const isCurrent = index === currentStep;

            return (
              <div 
                key={step.id}
                className={`p-4 rounded-lg border transition-all duration-300 ${
                  isActive ? 'border-blue-200 bg-blue-50 ring-2 ring-blue-100 transform scale-[1.02]' :
                  isCompleted ? 'border-green-200 bg-green-50' :
                  step.status === 'error' ? 'border-red-200 bg-red-50' :
                  'border-gray-200 bg-gray-50'
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    {getStepIcon(step.status, isActive)}
                    <div>
                      <div className={`font-medium ${isActive ? 'text-blue-700' : ''}`}>
                        {index + 1}. {step.name}
                      </div>
                      {step.throughput && (
                        <div className="text-xs text-muted-foreground">{step.throughput}</div>
                      )}
                    </div>
                  </div>
                  <div className="text-right">
                    {step.duration ? (
                      <div className="text-sm font-mono">{step.duration.toFixed(1)}s</div>
                    ) : step.estimatedDuration && isCurrent ? (
                      <div className="text-sm text-muted-foreground">~{step.estimatedDuration.toFixed(1)}s</div>
                    ) : null}
                  </div>
                </div>

                {/* Step Progress Bar */}
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span>Progress</span>
                    <span>{step.progress.toFixed(0)}%</span>
                  </div>
                  <Progress value={step.progress} className="h-2" />
                  
                  {/* Metrics */}
                  {step.metrics && (
                    <div className="flex justify-between text-xs text-muted-foreground">
                      {step.metrics.processedItems !== undefined && (
                        <span>{step.metrics.processedItems}/{step.metrics.totalItems} items</span>
                      )}
                      {step.metrics.rate && (
                        <span>{step.metrics.rate.toFixed(1)} items/sec</span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>
    </div>
  );
}
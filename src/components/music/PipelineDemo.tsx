import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  Play, 
  Download, 
  Settings, 
  FileText,
  Music,
  BarChart3,
  Clock,
  Activity,
  Stop,
  CheckCircle,
  XCircle,
  Timer,
  Cpu
} from "@phosphor-icons/react";
import { toast } from "sonner";

interface PipelineStep {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'completed' | 'error';
  duration?: number;
  output?: string;
  subProgress?: number;
  detailedOutput?: string[];
  startTime?: number;
  estimatedDuration?: number;
}

interface PipelineConfig {
  style: 'rock_punk' | 'rnb_ballad' | 'country_pop';
  duration_bars: number;
  bpm: number;
  key: string;
}

interface RealtimeMetrics {
  totalElapsed: number;
  estimatedRemaining: number;
  currentStepProgress: number;
  memoryUsage: number;
  cpuUsage: number;
}

export function PipelineDemo() {
  const [config, setConfig] = useState<PipelineConfig>({
    style: 'rock_punk',
    duration_bars: 32,
    bpm: 140,
    key: 'E'
  });
  
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [totalProgress, setTotalProgress] = useState(0);
  const [outputDir, setOutputDir] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<RealtimeMetrics>({
    totalElapsed: 0,
    estimatedRemaining: 0,
    currentStepProgress: 0,
    memoryUsage: 0,
    cpuUsage: 0
  });
  
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const pipelineStartTime = useRef<number>(0);
  const abortControllerRef = useRef<AbortController | null>(null);
  
  const [steps, setSteps] = useState<PipelineStep[]>([
    {
      id: 'ingest',
      name: 'Data Ingestion',
      description: 'Processing style parameters and creating composition metadata',
      status: 'pending',
      estimatedDuration: 2.5
    },
    {
      id: 'tokenize',
      name: 'MIDI Tokenization',
      description: 'Converting musical concepts to discrete tokens',
      status: 'pending',
      estimatedDuration: 3.2
    },
    {
      id: 'arrange',
      name: 'Arrangement Generation',
      description: 'Creating song structure using Transformer decoder',
      status: 'pending',
      estimatedDuration: 5.8
    },
    {
      id: 'melody',
      name: 'Melody & Harmony',
      description: 'Generating multi-track MIDI with style conditioning',
      status: 'pending',
      estimatedDuration: 8.1
    },
    {
      id: 'render',
      name: 'Stem Rendering',
      description: 'Converting MIDI to audio using sample libraries',
      status: 'pending',
      estimatedDuration: 12.4
    },
    {
      id: 'mix',
      name: 'Mix & Master',
      description: 'Auto-mixing with style-specific LUFS targets',
      status: 'pending',
      estimatedDuration: 6.7
    },
    {
      id: 'export',
      name: 'Export & Report',
      description: 'Generating final WAV and analysis report',
      status: 'pending',
      estimatedDuration: 2.1
    }
  ]);

  const styleConfigs = {
    rock_punk: {
      defaultBpm: 140,
      defaultKey: 'E',
      instruments: ['drums', 'bass_pick', 'guitar_distorted', 'guitar_clean'],
      lufsTarget: -9.5,
      description: 'High-energy, aggressive, distorted guitars'
    },
    rnb_ballad: {
      defaultBpm: 70,
      defaultKey: 'C',
      instruments: ['drums', 'bass_finger', 'piano', 'strings', 'vocals'],
      lufsTarget: -12.0,
      description: 'Smooth, warm, gradual builds'
    },
    country_pop: {
      defaultBpm: 110,
      defaultKey: 'G',
      instruments: ['drums', 'bass_pick', 'acoustic_guitar', 'electric_guitar', 'fiddle'],
      lufsTarget: -10.5,
      description: 'Balanced, acoustic-electric blend'
    }
  };

  // Real-time metrics update
  useEffect(() => {
    if (isRunning && !isPaused) {
      intervalRef.current = setInterval(() => {
        const elapsed = (Date.now() - pipelineStartTime.current) / 1000;
        const totalEstimated = steps.reduce((sum, step) => sum + (step.estimatedDuration || 0), 0);
        const completedTime = steps.slice(0, currentStep).reduce((sum, step) => sum + (step.duration || step.estimatedDuration || 0), 0);
        const remaining = Math.max(0, totalEstimated - elapsed);
        
        setMetrics({
          totalElapsed: elapsed,
          estimatedRemaining: remaining,
          currentStepProgress: Math.min(100, ((elapsed - completedTime) / (steps[currentStep]?.estimatedDuration || 1)) * 100),
          memoryUsage: 45 + Math.random() * 20, // Simulated
          cpuUsage: 60 + Math.random() * 30 // Simulated
        });
      }, 100);
    } else {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    }
    
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isRunning, isPaused, currentStep, steps]);

  const handleStyleChange = (style: string) => {
    const styleConfig = styleConfigs[style as keyof typeof styleConfigs];
    setConfig(prev => ({
      ...prev,
      style: style as PipelineConfig['style'],
      bpm: styleConfig.defaultBpm,
      key: styleConfig.defaultKey
    }));
  };

  const stopPipeline = () => {
    setIsRunning(false);
    setIsPaused(false);
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    toast.info('Pipeline stopped by user');
  };

  const pausePipeline = () => {
    setIsPaused(!isPaused);
    toast.info(isPaused ? 'Pipeline resumed' : 'Pipeline paused');
  };

  const simulatePipeline = async () => {
    setIsRunning(true);
    setIsPaused(false);
    setCurrentStep(0);
    setTotalProgress(0);
    setOutputDir(null);
    pipelineStartTime.current = Date.now();
    abortControllerRef.current = new AbortController();
    
    // Reset all steps
    setSteps(prev => prev.map(step => ({ 
      ...step, 
      status: 'pending', 
      duration: undefined, 
      output: undefined,
      subProgress: 0,
      detailedOutput: [],
      startTime: undefined
    })));
    
    toast.info(`Starting ${config.style} composition pipeline...`);
    
    try {
      for (let i = 0; i < steps.length; i++) {
        if (abortControllerRef.current?.signal.aborted) {
          throw new Error('Pipeline aborted');
        }
        
        setCurrentStep(i);
        const stepStartTime = Date.now();
        
        // Update step to running
        setSteps(prev => prev.map((step, idx) => 
          idx === i ? { ...step, status: 'running', startTime: stepStartTime } : step
        ));
        
        // Simulate detailed sub-progress
        const stepDuration = steps[i].estimatedDuration! * 1000;
        const subSteps = 10;
        
        for (let j = 0; j < subSteps; j++) {
          if (abortControllerRef.current?.signal.aborted) {
            throw new Error('Pipeline aborted');
          }
          
          // Wait for pause
          while (isPaused && !abortControllerRef.current?.signal.aborted) {
            await new Promise(resolve => setTimeout(resolve, 100));
          }
          
          await new Promise(resolve => setTimeout(resolve, stepDuration / subSteps));
          
          const subProgress = ((j + 1) / subSteps) * 100;
          setSteps(prev => prev.map((step, idx) => 
            idx === i ? { ...step, subProgress } : step
          ));
        }
        
        // Update step to completed with detailed output
        const actualDuration = (Date.now() - stepStartTime) / 1000;
        let stepOutput = '';
        let detailedOutput: string[] = [];
        
        switch (steps[i].id) {
          case 'ingest':
            stepOutput = 'composition_metadata.json created';
            detailedOutput = [
              '✓ Style configuration loaded',
              '✓ BPM and key signature validated',
              '✓ Instrument mapping initialized',
              '✓ Target LUFS configured'
            ];
            break;
          case 'tokenize':
            const tokenCount = Math.floor(Math.random() * 200 + 50);
            stepOutput = `${tokenCount} tokens generated`;
            detailedOutput = [
              '✓ MIDI events parsed',
              `✓ ${tokenCount} tokens created`,
              '✓ Vocabulary mapping complete',
              '✓ Token sequence validated'
            ];
            break;
          case 'arrange':
            const sectionCount = Math.floor(Math.random() * 3 + 5);
            stepOutput = `${sectionCount} sections created`;
            detailedOutput = [
              '✓ Transformer model loaded',
              '✓ Style conditioning applied',
              `✓ ${sectionCount} sections generated`,
              '✓ Coverage penalty applied'
            ];
            break;
          case 'melody':
            stepOutput = 'Multi-track MIDI generated';
            detailedOutput = [
              '✓ Chord progressions generated',
              '✓ Melody lines created',
              '✓ Bass patterns added',
              '✓ Drum grooves synthesized'
            ];
            break;
          case 'render':
            const stemCount = styleConfigs[config.style].instruments.length;
            stepOutput = `${stemCount} stems rendered`;
            detailedOutput = [
              '✓ Sample libraries loaded',
              '✓ MIDI to audio conversion',
              `✓ ${stemCount} stems rendered`,
              '✓ Latency compensation applied'
            ];
            break;
          case 'mix':
            stepOutput = `Target LUFS: ${styleConfigs[config.style].lufsTarget}`;
            detailedOutput = [
              '✓ EQ curves applied',
              '✓ Compression settings optimized',
              '✓ Stereo imaging balanced',
              `✓ LUFS target ${styleConfigs[config.style].lufsTarget} achieved`
            ];
            break;
          case 'export':
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
            setOutputDir(`exports/${timestamp}`);
            stepOutput = 'final.wav + report.json created';
            detailedOutput = [
              '✓ Audio rendered to WAV',
              '✓ Analysis report generated',
              '✓ Metadata embedded',
              '✓ Export complete'
            ];
            break;
        }
        
        setSteps(prev => prev.map((step, idx) => 
          idx === i ? { 
            ...step, 
            status: 'completed', 
            duration: actualDuration,
            output: stepOutput,
            detailedOutput,
            subProgress: 100
          } : step
        ));
        
        setTotalProgress(((i + 1) / steps.length) * 100);
      }
      
      toast.success('Pipeline completed successfully!');
      
    } catch (error) {
      if (error instanceof Error && error.message === 'Pipeline aborted') {
        toast.info('Pipeline stopped');
        setSteps(prev => prev.map((step, idx) => 
          idx === currentStep ? { ...step, status: 'error' } : step
        ));
      } else {
        toast.error('Pipeline failed!');
        setSteps(prev => prev.map((step, idx) => 
          idx === currentStep ? { ...step, status: 'error' } : step
        ));
      }
    } finally {
      setIsRunning(false);
      setIsPaused(false);
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    }
  };

  const getStepIcon = (status: PipelineStep['status']) => {
    switch (status) {
      case 'running': return <Activity className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'error': return <XCircle className="w-4 h-4 text-red-500" />;
      default: return <span className="w-4 h-4 bg-gray-300 rounded-full flex" />;
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const estimatedDuration = config.duration_bars * 4 * 60 / config.bpm; // seconds

  return (
    <div className="space-y-6">
      <Tabs defaultValue="config" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="config">Configuration</TabsTrigger>
          <TabsTrigger value="pipeline">Pipeline Status</TabsTrigger>
          <TabsTrigger value="output">Output & Analysis</TabsTrigger>
        </TabsList>
        
        <TabsContent value="config" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5" />
                Pipeline Configuration
              </CardTitle>
              <CardDescription>
                Configure the AI music generation parameters
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="style">Music Style</Label>
                  <Select value={config.style} onValueChange={handleStyleChange}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="rock_punk">
                        <div className="flex flex-col">
                          <span>Rock Punk</span>
                          <span className="text-xs text-muted-foreground">
                            {styleConfigs.rock_punk.description}
                          </span>
                        </div>
                      </SelectItem>
                      <SelectItem value="rnb_ballad">
                        <div className="flex flex-col">
                          <span>R&B Ballad</span>
                          <span className="text-xs text-muted-foreground">
                            {styleConfigs.rnb_ballad.description}
                          </span>
                        </div>
                      </SelectItem>
                      <SelectItem value="country_pop">
                        <div className="flex flex-col">
                          <span>Country Pop</span>
                          <span className="text-xs text-muted-foreground">
                            {styleConfigs.country_pop.description}
                          </span>
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="duration">Duration (bars)</Label>
                  <Input
                    id="duration"
                    type="number"
                    value={config.duration_bars}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      duration_bars: parseInt(e.target.value) || 32
                    }))}
                    min="8"
                    max="128"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="bpm">BPM</Label>
                  <Input
                    id="bpm"
                    type="number"
                    value={config.bpm}
                    onChange={(e) => setConfig(prev => ({
                      ...prev,
                      bpm: parseInt(e.target.value) || 120
                    }))}
                    min="60"
                    max="200"
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="key">Musical Key</Label>
                  <Select value={config.key} onValueChange={(key) => setConfig(prev => ({ ...prev, key }))}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {['C', 'D', 'E', 'F', 'G', 'A', 'B'].map(key => (
                        <SelectItem key={key} value={key}>{key}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              <div className="p-4 bg-muted rounded-lg">
                <h4 className="font-medium mb-2">Style Configuration</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Instruments:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {styleConfigs[config.style].instruments.map(inst => (
                        <Badge key={inst} variant="secondary" className="text-xs">
                          {inst.replace('_', ' ')}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Target LUFS:</span>
                    <span className="ml-2 font-mono">{styleConfigs[config.style].lufsTarget}</span>
                  </div>
                </div>
                <div className="mt-2 text-sm">
                  <span className="text-muted-foreground">Estimated Duration:</span>
                  <span className="ml-2">{estimatedDuration.toFixed(1)} seconds</span>
                </div>
              </div>
              
              <Button 
                onClick={simulatePipeline} 
                disabled={isRunning}
                className="w-full"
                size="lg"
              >
                {isRunning ? (
                  <Activity className="w-4 h-4 mr-2 animate-spin" />
                ) : (
                  <Play className="w-4 h-4 mr-2" />
                )}
                {isRunning ? 'Running Pipeline...' : 'Start AI Music Generation'}
              </Button>
              
              {isRunning && (
                <div className="flex gap-2">
                  <Button 
                    onClick={pausePipeline}
                    variant="outline"
                    className="flex-1"
                  >
                    {isPaused ? 'Resume' : 'Pause'}
                  </Button>
                  <Button 
                    onClick={stopPipeline}
                    variant="destructive"
                    className="flex-1"
                  >
                    <Stop className="w-4 h-4 mr-2" />
                    Stop
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="pipeline" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />
                Pipeline Execution
              </CardTitle>
              <CardDescription>
                Real-time status of the 7-step AI music composition process
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Real-time metrics */}
              {isRunning && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="text-center">
                    <div className="flex items-center justify-center gap-1 text-blue-600 mb-1">
                      <Timer className="w-4 h-4" />
                      <span className="text-sm font-medium">Elapsed</span>
                    </div>
                    <div className="text-lg font-mono">{formatTime(metrics.totalElapsed)}</div>
                  </div>
                  <div className="text-center">
                    <div className="flex items-center justify-center gap-1 text-blue-600 mb-1">
                      <Clock className="w-4 h-4" />
                      <span className="text-sm font-medium">Remaining</span>
                    </div>
                    <div className="text-lg font-mono">{formatTime(metrics.estimatedRemaining)}</div>
                  </div>
                  <div className="text-center">
                    <div className="flex items-center justify-center gap-1 text-blue-600 mb-1">
                      <Cpu className="w-4 h-4" />
                      <span className="text-sm font-medium">CPU</span>
                    </div>
                    <div className="text-lg font-mono">{metrics.cpuUsage.toFixed(0)}%</div>
                  </div>
                  <div className="text-center">
                    <div className="flex items-center justify-center gap-1 text-blue-600 mb-1">
                      <BarChart3 className="w-4 h-4" />
                      <span className="text-sm font-medium">Memory</span>
                    </div>
                    <div className="text-lg font-mono">{metrics.memoryUsage.toFixed(0)}%</div>
                  </div>
                </div>
              )}
              
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Overall Progress</span>
                  <span>{Math.round(totalProgress)}%</span>
                </div>
                <Progress value={totalProgress} className="w-full" />
                {isRunning && (
                  <div className="text-xs text-muted-foreground text-center">
                    {isPaused ? 'Pipeline is paused' : `Running step ${currentStep + 1} of ${steps.length}`}
                  </div>
                )}
              </div>
              
              <div className="space-y-3">
                {steps.map((step, index) => (
                  <div 
                    key={step.id}
                    className={`p-4 rounded-lg border transition-all duration-200 ${
                      step.status === 'running' ? 'border-blue-200 bg-blue-50 ring-2 ring-blue-100' :
                      step.status === 'completed' ? 'border-green-200 bg-green-50' :
                      step.status === 'error' ? 'border-red-200 bg-red-50' :
                      'border-gray-200 bg-gray-50'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3">
                        {getStepIcon(step.status)}
                        <div>
                          <div className="font-medium">{index + 1}. {step.name}</div>
                          <div className="text-sm text-muted-foreground">{step.description}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        {step.duration && (
                          <div className="text-xs text-muted-foreground">
                            {step.duration.toFixed(1)}s
                          </div>
                        )}
                        {step.status === 'running' && step.estimatedDuration && (
                          <div className="text-xs text-blue-600">
                            ~{step.estimatedDuration}s
                          </div>
                        )}
                      </div>
                    </div>
                    
                    {/* Sub-progress for running step */}
                    {step.status === 'running' && (
                      <div className="mb-2">
                        <Progress value={step.subProgress || 0} className="h-2" />
                        <div className="text-xs text-muted-foreground mt-1">
                          {(step.subProgress || 0).toFixed(0)}% complete
                        </div>
                      </div>
                    )}
                    
                    {/* Step output */}
                    {step.output && (
                      <div className="text-xs text-blue-600 mb-2 font-medium">{step.output}</div>
                    )}
                    
                    {/* Detailed output */}
                    {step.detailedOutput && step.detailedOutput.length > 0 && (
                      <div className="text-xs text-muted-foreground space-y-1">
                        {step.detailedOutput.map((output, idx) => (
                          <div key={idx} className="flex items-center gap-2">
                            <span className="text-green-500">•</span>
                            {output}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="output" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Generated Output
              </CardTitle>
              <CardDescription>
                Files and analysis from the completed pipeline
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {outputDir ? (
                <div className="space-y-4">
                  <Alert>
                    <Music className="w-4 h-4" />
                    <AlertDescription>
                      Pipeline completed successfully! Output directory: <code className="bg-muted px-1 rounded">{outputDir}</code>
                    </AlertDescription>
                  </Alert>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-base flex items-center gap-2">
                          <Download className="w-4 h-4" />
                          Generated Files
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span>final.wav</span>
                          <Badge variant="outline">24.7 MB</Badge>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span>report.json</span>
                          <Badge variant="outline">4.2 KB</Badge>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span>arrangement.json</span>
                          <Badge variant="outline">1.8 KB</Badge>
                        </div>
                        <div className="flex items-center justify-between text-sm">
                          <span>stems/ (5 files)</span>
                          <Badge variant="outline">47.3 MB</Badge>
                        </div>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardHeader className="pb-3">
                        <CardTitle className="text-base flex items-center gap-2">
                          <BarChart3 className="w-4 h-4" />
                          Audio Analysis
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span>Duration:</span>
                          <span>{estimatedDuration.toFixed(1)}s</span>
                        </div>
                        <div className="flex justify-between">
                          <span>LUFS:</span>
                          <span>{styleConfigs[config.style].lufsTarget}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Sample Rate:</span>
                          <span>48 kHz</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Bit Depth:</span>
                          <span>16-bit</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Spectral Centroid:</span>
                          <span>{Math.floor(Math.random() * 1000 + 1500)} Hz</span>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                  
                  <div className="space-y-2">
                    <h4 className="font-medium">Pipeline Timing</h4>
                    <div className="grid grid-cols-4 gap-2 text-xs">
                      {steps.filter(s => s.duration).map(step => (
                        <div key={step.id} className="text-center p-2 bg-muted rounded">
                          <div className="font-medium">{step.name}</div>
                          <div className="text-muted-foreground">{step.duration?.toFixed(1)}s</div>
                        </div>
                      ))}
                    </div>
                    <div className="text-center p-2 bg-primary/10 rounded">
                      <div className="font-medium">Total Time</div>
                      <div className="text-muted-foreground">
                        {steps.reduce((sum, s) => sum + (s.duration || 0), 0).toFixed(1)}s
                      </div>
                    </div>
                  </div>
                  
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Music className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Run the pipeline to see generated output files and analysis</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { useKV } from "@/hooks/useKV";
import { 
  ArrowRight, 
  Database, 
  Wand2, 
  Layout, 
  MusicNote, 
  Waveform, 
  Sliders,
  Play,
  CheckCircle,
  AlertCircle
} from "@phosphor-icons/react";

interface PipelineStage {
  id: string;
  name: string;
  icon: React.ComponentType<any>;
  description: string;
  inputs: string[];
  outputs: string[];
  dataKey: string;
  status: 'empty' | 'ready' | 'processing' | 'complete';
}

interface DataFlowState {
  lyrics: any[];
  arrangements: any[];
  compositions: any[];
  soundDesigns: any[];
  mixMasters: any[];
}

export function DataFlowPipeline() {
  const [lyrics] = useKV<any[]>("generated-lyrics", []);
  const [arrangements] = useKV<any[]>("song-structures", []);
  const [compositions] = useKV<any[]>("melody-harmony-compositions", []);
  const [soundDesigns] = useKV<any[]>("sound-designs", []);
  const [mixMasters] = useKV<any[]>("mix-masters", []);
  const [isAnimating, setIsAnimating] = useState(false);

  const pipelineStages: PipelineStage[] = [
    {
      id: "lyrics",
      name: "Lyrics",
      icon: Wand2,
      description: "AI-generated lyrics with themes, moods, and structure",
      inputs: ["Theme", "Style", "Mood"],
      outputs: ["Structured Lyrics", "Verse/Chorus Content"],
      dataKey: "lyrics",
      status: lyrics.length > 0 ? 'complete' : 'empty'
    },
    {
      id: "arrangement",
      name: "Arrangement",
      icon: Layout,
      description: "Song structure mapping with timing and sections",
      inputs: ["Genre", "BPM", "Target Length"],
      outputs: ["Section Map", "Timing Data", "Structure JSON"],
      dataKey: "arrangements",
      status: arrangements.length > 0 ? 'complete' : 'empty'
    },
    {
      id: "melody",
      name: "Melody & Harmony",
      icon: MusicNote,
      description: "MIDI composition with multi-track arrangements",
      inputs: ["Key Signature", "Style", "Arrangement Data"],
      outputs: ["MIDI Tracks", "Chord Progressions", "Melodic Content"],
      dataKey: "compositions",
      status: compositions.length > 0 ? 'complete' : 'empty'
    },
    {
      id: "sound",
      name: "Sound Design",
      icon: Waveform,
      description: "Synthesizer patches and audio texture generation",
      inputs: ["Composition Data", "Style Token", "Instrument List"],
      outputs: ["Synth Patches", "Effect Settings", "Audio Textures"],
      dataKey: "soundDesigns",
      status: soundDesigns.length > 0 ? 'complete' : 'empty'
    },
    {
      id: "mixing",
      name: "Mix & Master",
      icon: Sliders,
      description: "Professional mixing and mastering settings",
      inputs: ["Sound Design", "Composition", "Target Style"],
      outputs: ["Mix Settings", "Master Chain", "Final Audio"],
      dataKey: "mixMasters",
      status: mixMasters.length > 0 ? 'complete' : 'empty'
    }
  ];

  const dataFlowState: DataFlowState = {
    lyrics,
    arrangements,
    compositions,
    soundDesigns,
    mixMasters
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'complete':
        return 'text-green-600 bg-green-100';
      case 'ready':
        return 'text-blue-600 bg-blue-100';
      case 'processing':
        return 'text-yellow-600 bg-yellow-100';
      default:
        return 'text-gray-500 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'complete':
        return CheckCircle;
      case 'ready':
      case 'processing':
        return Play;
      default:
        return AlertCircle;
    }
  };

  const animateDataFlow = () => {
    setIsAnimating(true);
    setTimeout(() => setIsAnimating(false), 3000);
  };

  const getDataCount = (dataKey: string) => {
    const data = dataFlowState[dataKey as keyof DataFlowState];
    return Array.isArray(data) ? data.length : 0;
  };

  const getLatestData = (dataKey: string) => {
    const data = dataFlowState[dataKey as keyof DataFlowState];
    if (Array.isArray(data) && data.length > 0) {
      return data[0]; // Most recent item (assuming they're stored with newest first)
    }
    return null;
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h3 className="text-xl font-semibold mb-2">AI Music Production Pipeline</h3>
        <p className="text-muted-foreground">
          Data flows through each stage with loose coupling and style consistency
        </p>
      </div>

      <div className="flex justify-center mb-6">
        <Button onClick={animateDataFlow} variant="outline" className="flex items-center gap-2">
          <Database className="w-4 h-4" />
          Visualize Data Flow
        </Button>
      </div>

      {/* Pipeline Visualization */}
      <div className="relative">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          {pipelineStages.map((stage, index) => {
            const IconComponent = stage.icon;
            const StatusIcon = getStatusIcon(stage.status);
            const dataCount = getDataCount(stage.dataKey);
            const latestData = getLatestData(stage.dataKey);
            
            return (
              <div key={stage.id} className="flex flex-col lg:flex-row lg:items-center">
                <Card className={`w-full lg:w-64 transition-all duration-300 ${
                  isAnimating ? 'shadow-lg scale-105' : ''
                }`}>
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <div className={`p-2 rounded-lg ${getStatusColor(stage.status)}`}>
                          <IconComponent className="w-4 h-4" />
                        </div>
                        <div>
                          <h4 className="font-medium text-sm">{stage.name}</h4>
                          <div className="flex items-center gap-1 mt-1">
                            <StatusIcon className={`w-3 h-3 ${getStatusColor(stage.status).split(' ')[0]}`} />
                            <span className="text-xs text-muted-foreground">
                              {dataCount} items
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>

                    <p className="text-xs text-muted-foreground mb-3">
                      {stage.description}
                    </p>

                    {/* Latest Data Preview */}
                    {latestData && (
                      <div className="mb-3 p-2 bg-muted/30 rounded text-xs">
                        <div className="font-medium">Latest:</div>
                        <div className="truncate">
                          {latestData.name || latestData.title || `${stage.name} Data`}
                        </div>
                        {latestData.style && (
                          <Badge variant="outline" className="text-xs mt-1">
                            {latestData.style}
                          </Badge>
                        )}
                      </div>
                    )}

                    {/* Input/Output Flow */}
                    <div className="space-y-2">
                      <div>
                        <div className="text-xs font-medium text-muted-foreground mb-1">Inputs:</div>
                        <div className="flex flex-wrap gap-1">
                          {stage.inputs.map((input, inputIndex) => (
                            <Badge key={inputIndex} variant="secondary" className="text-xs">
                              {input}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs font-medium text-muted-foreground mb-1">Outputs:</div>
                        <div className="flex flex-wrap gap-1">
                          {stage.outputs.map((output, outputIndex) => (
                            <Badge key={outputIndex} variant="outline" className="text-xs">
                              {output}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Arrow connector */}
                {index < pipelineStages.length - 1 && (
                  <div className="flex justify-center lg:mx-4 my-2 lg:my-0">
                    <div className={`transition-all duration-300 ${
                      isAnimating ? 'text-accent animate-pulse' : 'text-muted-foreground'
                    }`}>
                      <ArrowRight className="w-6 h-6 lg:rotate-0 rotate-90" />
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <Separator />

      {/* Architecture Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardContent className="p-4">
            <h4 className="font-medium mb-3 flex items-center gap-2">
              <Database className="w-4 h-4" />
              Loose Coupling Benefits
            </h4>
            <div className="space-y-2 text-sm text-muted-foreground">
              <div className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                <span>Each module is independent and swappable</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                <span>Can replace melody generator without affecting mixing</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                <span>JSON-based data exchange format</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" />
                <span>Easy to test and debug individual stages</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <h4 className="font-medium mb-3 flex items-center gap-2">
              <Waveform className="w-4 h-4" />
              Style Control System
            </h4>
            <div className="space-y-2 text-sm text-muted-foreground">
              <div className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                <span>Style tokens pass through entire pipeline</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                <span>All modules "agree" on genre characteristics</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                <span>Consistent musical vocabulary across stages</span>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                <span>Global style configuration for cohesive output</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Statistics */}
      <Card>
        <CardContent className="p-4">
          <h4 className="font-medium mb-3">Pipeline Statistics</h4>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
            {pipelineStages.map((stage) => {
              const count = getDataCount(stage.dataKey);
              return (
                <div key={stage.id} className="space-y-1">
                  <div className="text-2xl font-bold text-accent">{count}</div>
                  <div className="text-xs text-muted-foreground">{stage.name}</div>
                  <Badge 
                    variant={count > 0 ? "default" : "secondary"} 
                    className="text-xs"
                  >
                    {count > 0 ? "Active" : "Empty"}
                  </Badge>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <div className="p-4 bg-muted/50 rounded-lg">
        <h4 className="font-medium mb-2">Architecture Overview:</h4>
        <div className="text-sm text-muted-foreground space-y-1">
          <p>• <strong>Arrangement →</strong> Generates structural timing and section maps</p>
          <p>• <strong>Melody/Harmony →</strong> Creates MIDI compositions following arrangement data</p>
          <p>• <strong>Sound Design →</strong> Synthesizes patches matching composition instruments</p>
          <p>• <strong>Mixing/Mastering →</strong> Applies professional audio processing settings</p>
          <p>• <strong>Final Track →</strong> Complete production-ready music composition</p>
        </div>
      </div>
    </div>
  );
}
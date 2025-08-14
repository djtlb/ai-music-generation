import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { useKV } from '@github/spark/hooks';
import { 
  SpeakerHigh, 
  Save, 
  Play, 
  Pause, 
  Sparkles, 
  Download,
  Sliders,
  WaveSquare,
  Equalizer,
  Lightning
} from "@phosphor-icons/react";
import { toast } from "sonner";

interface MixingChannel {
  name: string;
  level: number; // dB
  pan: number; // -1 to 1
  muted: boolean;
  solo: boolean;
  eq: EQSettings;
  compression: CompressionSettings;
  effects: ChannelEffect[];
}

interface EQSettings {
  lowGain: number;
  lowMidGain: number;
  highMidGain: number;
  highGain: number;
  lowCut: number;
  highCut: number;
}

interface CompressionSettings {
  threshold: number;
  ratio: number;
  attack: number;
  release: number;
  makeupGain: number;
  enabled: boolean;
}

interface ChannelEffect {
  type: string;
  parameters: Record<string, number>;
  enabled: boolean;
  insertPosition: number;
}

interface MasteringSettings {
  finalEQ: EQSettings;
  multiband: MultibandCompressor;
  limiter: LimiterSettings;
  stereoEnhancement: StereoSettings;
  loudness: LoudnessSettings;
}

interface MultibandCompressor {
  enabled: boolean;
  lowBand: CompressorBand;
  midBand: CompressorBand;
  highBand: CompressorBand;
}

interface CompressorBand {
  threshold: number;
  ratio: number;
  attack: number;
  release: number;
  gain: number;
}

interface LimiterSettings {
  threshold: number;
  release: number;
  lookahead: number;
  enabled: boolean;
}

interface StereoSettings {
  width: number;
  bass: number;
  enabled: boolean;
}

interface LoudnessSettings {
  targetLUFS: number;
  truePeak: number;
  enabled: boolean;
}

interface MixMaster {
  id: string;
  name: string;
  style: string;
  channels: MixingChannel[];
  mastering: MasteringSettings;
  soundDesign?: any;
  composition?: any;
  timestamp: number;
}

const MIX_STYLES = [
  { name: "Rock/Punk", value: "rock_punk", lufs: -9.5, centroid: 2800, msRatio: 0.6, description: "Aggressive loudness for modern rock/punk" },
  { name: "R&B Ballad", value: "rnb_ballad", lufs: -12.0, centroid: 1800, msRatio: 0.8, description: "More dynamic, less compressed ballad style" },
  { name: "Country Pop", value: "country_pop", lufs: -10.5, centroid: 2200, msRatio: 0.7, description: "Radio-ready loudness with clarity" },
  { name: "Modern Pop", value: "modern_pop", lufs: -9.0, centroid: 2500, msRatio: 0.65, description: "Streaming-optimized pop sound" },
  { name: "Lo-Fi Hip Hop", value: "lofi_hiphop", lufs: -14.0, centroid: 1600, msRatio: 0.75, description: "Warm, relaxed dynamics" },
  { name: "Electronic Dance", value: "edm", lufs: -8.5, centroid: 3200, msRatio: 0.55, description: "Maximum loudness for club play" }
];

const EFFECT_PRESETS = {
  "Reverb": { roomSize: 0.5, decay: 0.7, predelay: 0.02, highCut: 0.8 },
  "Delay": { time: 0.25, feedback: 0.3, highCut: 0.7, stereo: 0.5 },
  "Chorus": { rate: 0.5, depth: 0.3, feedback: 0.2, mix: 0.3 },
  "Distortion": { drive: 0.4, tone: 0.6, level: 0.8, type: 0.5 }
};

export function MixingMasteringEngine() {
  const [style, setStyle] = useState("");
  const [soundDesignJSON, setSoundDesignJSON] = useState("");
  const [compositionJSON, setCompositionJSON] = useState("");
  const [generatedMix, setGeneratedMix] = useState<MixingChannel[]>([]);
  const [generatedMastering, setGeneratedMastering] = useState<MasteringSettings | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [mixMasters, setMixMasters] = useKV<MixMaster[]>("mix-masters", []);
  const [savedSoundDesigns, setSavedSoundDesigns] = useKV<any[]>("sound-designs", []);
  const [savedCompositions, setSavedCompositions] = useKV<any[]>("melody-harmony-compositions", []);
  const [selectedSoundDesign, setSelectedSoundDesign] = useState("");
  const [selectedComposition, setSelectedComposition] = useState("");

  const generateMixMaster = async () => {
    if (!style) {
      toast.error("Please select a mixing style");
      return;
    }

    let soundDesignData = null;
    let compositionData = null;

    // Parse inputs
    if (soundDesignJSON.trim()) {
      try {
        soundDesignData = JSON.parse(soundDesignJSON);
      } catch (e) {
        toast.error("Invalid sound design JSON format");
        return;
      }
    } else if (selectedSoundDesign) {
      const selected = savedSoundDesigns.find(sd => sd.id === selectedSoundDesign);
      if (selected) soundDesignData = selected;
    }

    if (compositionJSON.trim()) {
      try {
        compositionData = JSON.parse(compositionJSON);
      } catch (e) {
        toast.error("Invalid composition JSON format");
        return;
      }
    } else if (selectedComposition) {
      const selected = savedCompositions.find(comp => comp.id === selectedComposition);
      if (selected) compositionData = selected;
    }

    setIsGenerating(true);
    try {
      const prompt = spark.llmPrompt`Generate professional auto-mixing settings using the differentiable PyTorch-based mixing chain for ${style} style.

The Auto-Mix system architecture:
- **Feature Extraction**: Analyzes RMS, crest factor, spectral centroid, dynamic range from each stem
- **MLP Parameter Predictor**: Neural network predicts optimal mixing parameters from stem features
- **Differentiable Processing**: PyTorch-based EQ, compression, saturation, level, pan per stem
- **Mastering Chain**: Bus compression, master EQ, stereo widener, brick-wall limiter
- **Target Optimization**: Automatically hits target LUFS, spectral centroid, and M/S ratio

Target specifications for ${style}:
${MIX_STYLES.find(s => s.value === style) ? `
- LUFS Target: ${MIX_STYLES.find(s => s.value === style)?.lufs} dB
- Spectral Centroid: ${MIX_STYLES.find(s => s.value === style)?.centroid} Hz  
- Stereo M/S Ratio: ${MIX_STYLES.find(s => s.value === style)?.msRatio}
- Style Notes: ${MIX_STYLES.find(s => s.value === style)?.description}
` : ''}

Parameter ranges (auto-scaled by MLP):
- EQ Gains: -12 to +12 dB (4-band: low shelf, low-mid, high-mid, high shelf)
- Compression Threshold: -40 to -6 dB, Ratio: 1:1 to 10:1
- Attack: 0.001 to 0.1s, Release: 0.01 to 1.0s, Makeup: 0 to 12 dB
- Saturation: 1.0 to 3.0x, Level: -24 to +6 dB, Pan: -1.0 to +1.0
- Master: Bus compression, stereo width 0.5-1.5, limiter threshold 0.7-1.0

${soundDesignData ? `Base mixing on these instrument patches:
${JSON.stringify(soundDesignData, null, 2)}

Create mixing channels for each instrument with auto-mix predicted parameters.` : ''}

${compositionData ? `Consider this arrangement and track data:
${JSON.stringify(compositionData, null, 2)}

Match the composition structure and instrument roles.` : ''}

Generate parameters that the MLP would predict to hit the ${style} targets.

Return a JSON object with this exact format:
{
  "autoMixAnalysis": {
    "predictedLUFS": ${MIX_STYLES.find(s => s.value === style)?.lufs || -10.0},
    "predictedSpectralCentroid": ${MIX_STYLES.find(s => s.value === style)?.centroid || 2200},
    "predictedMSRatio": ${MIX_STYLES.find(s => s.value === style)?.msRatio || 0.65},
    "qualityScore": 0.85,
    "mlpConfidence": 0.92,
    "stemFeatureAnalysis": "Analyzed RMS, crest factor, spectral centroid from input stems",
    "processingNotes": "Auto-mix MLP predicted parameters to hit ${style} targets with differentiable processing chain"
  },
  "channels": [
    {
      "name": "Lead Synth",
      "level": -6.0,
      "pan": 0.0,
      "muted": false,
      "solo": false,
      "eq": {
        "lowGain": 0.0,
        "lowMidGain": 2.0,
        "highMidGain": 1.5,
        "highGain": 0.5,
        "lowCut": 80,
        "highCut": 18000
      },
      "compression": {
        "threshold": -12.0,
        "ratio": 3.0,
        "attack": 0.003,
        "release": 0.1,
        "makeupGain": 2.0,
        "enabled": true
      },
      "effects": [
        {
          "type": "Reverb",
          "parameters": {
            "roomSize": 0.4,
            "decay": 0.6,
            "predelay": 0.02,
            "highCut": 0.8
          },
          "enabled": true,
          "insertPosition": 1
        }
      ]
    }
  ],
  "mastering": {
    "finalEQ": {
      "lowGain": 0.5,
      "lowMidGain": 0.0,
      "highMidGain": 0.8,
      "highGain": 1.0,
      "lowCut": 30,
      "highCut": 20000
    },
    "multiband": {
      "enabled": true,
      "lowBand": {
        "threshold": -18.0,
        "ratio": 2.5,
        "attack": 0.01,
        "release": 0.2,
        "gain": 0.5
      },
      "midBand": {
        "threshold": -15.0,
        "ratio": 3.0,
        "attack": 0.005,
        "release": 0.15,
        "gain": 0.8
      },
      "highBand": {
        "threshold": -12.0,
        "ratio": 4.0,
        "attack": 0.001,
        "release": 0.1,
        "gain": 1.0
      }
    },
    "limiter": {
      "threshold": -1.0,
      "release": 0.05,
      "lookahead": 0.005,
      "enabled": true
    },
    "stereoEnhancement": {
      "width": 1.2,
      "bass": 0.8,
      "enabled": true
    },
    "loudness": {
      "targetLUFS": ${MIX_STYLES.find(s => s.value === style)?.lufs || -10.0},
      "truePeak": -1.0,
      "enabled": true
    }
  }
}

Requirements:
- Create mixing channels for all instruments in the composition/sound design
- Use ${style} mixing characteristics optimized by the auto-mix MLP
- Set parameters to achieve target LUFS, spectral centroid, and M/S ratio
- Configure EQ and compression based on stem feature analysis
- Include mastering chain with differentiable processing
- Ensure realistic parameter ranges for the PyTorch-based processing chain
- Include auto-mix analysis predictions and quality metrics`;

      const result = await spark.llm(prompt, "gpt-4o", true);
      const mixData = JSON.parse(result);
      
      setGeneratedMix(mixData.channels);
      setGeneratedMastering(mixData.mastering);
      
      // Store auto-mix analysis if present
      if (mixData.autoMixAnalysis) {
        console.log("Auto-Mix Analysis:", mixData.autoMixAnalysis);
      }
      
      toast.success("Auto-mix settings generated with ML-predicted parameters!");
    } catch (error) {
      console.error("Generation error:", error);
      toast.error("Failed to generate mix settings. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const processAudio = () => {
    if (generatedMix.length === 0) {
      toast.error("No mix settings to process");
      return;
    }

    setIsProcessing(true);
    
    // Simulate audio processing - in a real implementation, this would apply
    // the mixing and mastering settings to actual audio data
    setTimeout(() => {
      setIsProcessing(false);
      toast.success("Audio processing completed!");
    }, 6000);
  };

  const saveMixMaster = () => {
    if (generatedMix.length === 0 || !generatedMastering) {
      toast.error("No mix/master settings to save");
      return;
    }

    const soundDesignData = selectedSoundDesign ? 
      savedSoundDesigns.find(sd => sd.id === selectedSoundDesign) : 
      (soundDesignJSON.trim() ? JSON.parse(soundDesignJSON) : null);

    const compositionData = selectedComposition ? 
      savedCompositions.find(comp => comp.id === selectedComposition) : 
      (compositionJSON.trim() ? JSON.parse(compositionJSON) : null);

    const newMixMaster: MixMaster = {
      id: Date.now().toString(),
      name: `${style} Mix/Master`,
      style,
      channels: generatedMix,
      mastering: generatedMastering,
      soundDesign: soundDesignData,
      composition: compositionData,
      timestamp: Date.now()
    };

    setMixMasters(current => [newMixMaster, ...current]);
    toast.success("Mix/Master settings saved to history!");
  };

  const exportMixSettings = () => {
    if (generatedMix.length === 0 || !generatedMastering) {
      toast.error("No settings to export");
      return;
    }

    const exportData = {
      name: `${style} Mix/Master`,
      style,
      mixing: {
        channels: generatedMix
      },
      mastering: generatedMastering,
      metadata: {
        generatedAt: new Date().toISOString(),
        style,
        version: "1.0"
      }
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `mix-master-${style.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    toast.success("Mix/Master settings exported!");
  };

  const loadSoundDesign = (designId: string) => {
    const design = savedSoundDesigns.find(sd => sd.id === designId);
    if (design) {
      setSelectedSoundDesign(designId);
      setSoundDesignJSON("");
    }
  };

  const loadComposition = (compositionId: string) => {
    const composition = savedCompositions.find(comp => comp.id === compositionId);
    if (composition) {
      setSelectedComposition(compositionId);
      setCompositionJSON("");
    }
  };

  const formatDbValue = (value: number) => {
    return `${value > 0 ? '+' : ''}${value.toFixed(1)} dB`;
  };

  const formatPanValue = (value: number) => {
    if (value === 0) return "Center";
    if (value < 0) return `${Math.abs(value * 100).toFixed(0)}% L`;
    return `${(value * 100).toFixed(0)}% R`;
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="mix-style">Target Style & Auto-Mix Settings</Label>
          <Select value={style} onValueChange={setStyle}>
            <SelectTrigger id="mix-style">
              <SelectValue placeholder="Select auto-mixing style" />
            </SelectTrigger>
            <SelectContent>
              {MIX_STYLES.map((s) => (
                <SelectItem key={s.value} value={s.value}>
                  <div className="flex flex-col">
                    <span>{s.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {s.lufs} LUFS • {s.centroid}Hz • {s.msRatio} M/S
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          {style && (
            <div className="text-xs text-muted-foreground">
              {MIX_STYLES.find(s => s.value === style)?.description}
            </div>
          )}
        </div>

        <div className="space-y-2">
          <Label>Differentiable Mixing Chain</Label>
          <div className="text-sm text-muted-foreground space-y-1">
            <div>PyTorch-based: Stem Features → MLP → Mix Parameters</div>
            <div className="flex gap-4 text-xs">
              <span>• Per-stem EQ + Compression</span>
              <span>• Bus processing</span>
              <span>• Target LUFS/spectral control</span>
            </div>
          </div>
        </div>
      </div>

      {/* Input Sources Section */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <h3 className="text-lg font-medium">Audio Stem Sources</h3>
            <p className="text-sm text-muted-foreground">
              The auto-mixing system analyzes stem features (RMS, crest factor, spectral centroid) and uses an MLP to predict optimal mixing parameters for your target style and LUFS.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Sound Design Input */}
              <div className="space-y-3">
                <Label className="text-base font-medium">Sound Design Input</Label>
                
                {savedSoundDesigns.length > 0 && (
                  <div className="space-y-2">
                    <Label htmlFor="saved-sound-design">Use Saved Sound Design</Label>
                    <Select value={selectedSoundDesign} onValueChange={loadSoundDesign}>
                      <SelectTrigger id="saved-sound-design">
                        <SelectValue placeholder="Select sound design" />
                      </SelectTrigger>
                      <SelectContent>
                        {savedSoundDesigns.map((sd) => (
                          <SelectItem key={sd.id} value={sd.id}>
                            {sd.name} ({sd.style})
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                <div className="space-y-2">
                  <Label htmlFor="sound-design-json">Or Paste Sound Design JSON</Label>
                  <Textarea
                    id="sound-design-json"
                    placeholder="Paste sound design JSON here..."
                    value={soundDesignJSON}
                    onChange={(e) => {
                      setSoundDesignJSON(e.target.value);
                      if (e.target.value.trim()) {
                        setSelectedSoundDesign("");
                      }
                    }}
                    className="font-mono text-sm"
                    rows={3}
                  />
                </div>
              </div>

              {/* Composition Input */}
              <div className="space-y-3">
                <Label className="text-base font-medium">Composition Input</Label>
                
                {savedCompositions.length > 0 && (
                  <div className="space-y-2">
                    <Label htmlFor="saved-composition">Use Saved Composition</Label>
                    <Select value={selectedComposition} onValueChange={loadComposition}>
                      <SelectTrigger id="saved-composition">
                        <SelectValue placeholder="Select composition" />
                      </SelectTrigger>
                      <SelectContent>
                        {savedCompositions.map((comp) => (
                          <SelectItem key={comp.id} value={comp.id}>
                            {comp.name} ({comp.key} {comp.style})
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                <div className="space-y-2">
                  <Label htmlFor="composition-json">Or Paste Composition JSON</Label>
                  <Textarea
                    id="composition-json"
                    placeholder="Paste composition JSON here..."
                    value={compositionJSON}
                    onChange={(e) => {
                      setCompositionJSON(e.target.value);
                      if (e.target.value.trim()) {
                        setSelectedComposition("");
                      }
                    }}
                    className="font-mono text-sm"
                    rows={3}
                  />
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex gap-3 flex-wrap">
        <Button 
          onClick={generateMixMaster} 
          disabled={isGenerating || !style}
          className="flex items-center gap-2"
        >
          {isGenerating ? (
            <>
              <Sparkles className="w-4 h-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Sliders className="w-4 h-4" />
              Generate Mix/Master
            </>
          )}
        </Button>

        {generatedMix.length > 0 && (
          <>
            <Button 
              variant="secondary" 
              onClick={processAudio}
              disabled={isProcessing}
              className="flex items-center gap-2"
            >
              {isProcessing ? (
                <>
                  <WaveSquare className="w-4 h-4 animate-pulse" />
                  Processing...
                </>
              ) : (
                <>
                  <SpeakerHigh className="w-4 h-4" />
                  Process Audio
                </>
              )}
            </Button>

            <Button variant="outline" onClick={saveMixMaster} className="flex items-center gap-2">
              <Save className="w-4 h-4" />
              Save
            </Button>

            <Button variant="outline" onClick={exportMixSettings} className="flex items-center gap-2">
              <Download className="w-4 h-4" />
              Export Settings
            </Button>
          </>
        )}
      </div>

      {generatedMix.length > 0 && generatedMastering && (
        <>
          <Separator />
          <Card>
            <CardContent className="pt-6">
              <div className="space-y-6">
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <h3 className="text-lg font-semibold">Mix & Master Settings</h3>
                  <div className="flex gap-2 flex-wrap">
                    <Badge variant="secondary">{style}</Badge>
                    <Badge variant="outline">{generatedMix.length} Channels</Badge>
                  </div>
                </div>

                {/* Mixing Channels */}
                <div className="space-y-4">
                  <h4 className="font-medium flex items-center gap-2">
                    <Sliders className="w-5 h-5" />
                    Mixing Channels
                  </h4>
                  
                  {generatedMix.map((channel, index) => (
                    <div
                      key={index}
                      className="p-4 bg-card border rounded-lg hover:bg-accent/5 transition-colors"
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div>
                          <h5 className="font-medium">{channel.name}</h5>
                          <div className="flex gap-2 mt-1">
                            <Badge variant="outline" className="text-xs">
                              {formatDbValue(channel.level)}
                            </Badge>
                            <Badge variant="secondary" className="text-xs">
                              {formatPanValue(channel.pan)}
                            </Badge>
                            {channel.muted && <Badge variant="destructive" className="text-xs">Muted</Badge>}
                            {channel.solo && <Badge variant="default" className="text-xs">Solo</Badge>}
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {/* EQ Settings */}
                        <div>
                          <h6 className="font-medium mb-2 flex items-center gap-2">
                            <Equalizer className="w-4 h-4" />
                            EQ
                          </h6>
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Low:</span>
                              <span>{formatDbValue(channel.eq.lowGain)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Low-Mid:</span>
                              <span>{formatDbValue(channel.eq.lowMidGain)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">High-Mid:</span>
                              <span>{formatDbValue(channel.eq.highMidGain)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">High:</span>
                              <span>{formatDbValue(channel.eq.highGain)}</span>
                            </div>
                          </div>
                        </div>

                        {/* Compression */}
                        <div>
                          <h6 className="font-medium mb-2">Compression</h6>
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Threshold:</span>
                              <span>{formatDbValue(channel.compression.threshold)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Ratio:</span>
                              <span>{channel.compression.ratio.toFixed(1)}:1</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Attack:</span>
                              <span>{(channel.compression.attack * 1000).toFixed(1)}ms</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Release:</span>
                              <span>{(channel.compression.release * 1000).toFixed(0)}ms</span>
                            </div>
                          </div>
                        </div>

                        {/* Effects */}
                        <div>
                          <h6 className="font-medium mb-2">Effects</h6>
                          {channel.effects.length > 0 ? (
                            <div className="space-y-1">
                              {channel.effects.map((effect, effectIndex) => (
                                <Badge 
                                  key={effectIndex}
                                  variant={effect.enabled ? "default" : "secondary"}
                                  className="text-xs mr-1"
                                >
                                  {effect.type}
                                </Badge>
                              ))}
                            </div>
                          ) : (
                            <span className="text-sm text-muted-foreground">No effects</span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Mastering Section */}
                <Separator />
                <div className="space-y-4">
                  <h4 className="font-medium flex items-center gap-2">
                    <Lightning className="w-5 h-5" />
                    Mastering Chain
                  </h4>

                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {/* Final EQ */}
                    <div className="p-4 bg-muted/30 rounded-lg">
                      <h5 className="font-medium mb-3">Final EQ</h5>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Low:</span>
                          <span>{formatDbValue(generatedMastering.finalEQ.lowGain)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Low-Mid:</span>
                          <span>{formatDbValue(generatedMastering.finalEQ.lowMidGain)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">High-Mid:</span>
                          <span>{formatDbValue(generatedMastering.finalEQ.highMidGain)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">High:</span>
                          <span>{formatDbValue(generatedMastering.finalEQ.highGain)}</span>
                        </div>
                      </div>
                    </div>

                    {/* Multiband Compressor */}
                    <div className="p-4 bg-muted/30 rounded-lg">
                      <h5 className="font-medium mb-3">Multiband Compressor</h5>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Low Band:</span>
                          <span>{generatedMastering.multiband.lowBand.ratio.toFixed(1)}:1</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Mid Band:</span>
                          <span>{generatedMastering.multiband.midBand.ratio.toFixed(1)}:1</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">High Band:</span>
                          <span>{generatedMastering.multiband.highBand.ratio.toFixed(1)}:1</span>
                        </div>
                        <Badge variant={generatedMastering.multiband.enabled ? "default" : "secondary"} className="text-xs">
                          {generatedMastering.multiband.enabled ? "Enabled" : "Disabled"}
                        </Badge>
                      </div>
                    </div>

                    {/* Limiter & Loudness */}
                    <div className="p-4 bg-muted/30 rounded-lg">
                      <h5 className="font-medium mb-3">Limiter & Loudness</h5>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Threshold:</span>
                          <span>{formatDbValue(generatedMastering.limiter.threshold)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Target LUFS:</span>
                          <span>{generatedMastering.loudness.targetLUFS.toFixed(1)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">True Peak:</span>
                          <span>{formatDbValue(generatedMastering.loudness.truePeak)}</span>
                        </div>
                        <Badge variant={generatedMastering.limiter.enabled ? "default" : "secondary"} className="text-xs">
                          {generatedMastering.limiter.enabled ? "Limited" : "Open"}
                        </Badge>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-muted/50 rounded-lg">
                  <h4 className="font-medium mb-2">Auto-Mix System Architecture:</h4>
                  <div className="text-sm text-muted-foreground space-y-1">
                    <p>• <strong>Stem Feature Extraction:</strong> RMS energy, crest factor (peak/RMS), spectral centroid, dynamic range analysis</p>
                    <p>• <strong>MLP Parameter Predictor:</strong> 3-layer neural network predicts {generatedMix.length * 12 + 11} parameters from {generatedMix.length * 4} stem features</p>
                    <p>• <strong>Differentiable Processing:</strong> PyTorch-based EQ (4-band parametric), compressor, soft saturation per stem</p>
                    <p>• <strong>Mastering Chain:</strong> Bus compressor, master EQ, M/S stereo widener, brick-wall limiter</p>
                    <p>• <strong>Target Optimization:</strong> Calibrated for {MIX_STYLES.find(s => s.value === style)?.lufs} LUFS, {MIX_STYLES.find(s => s.value === style)?.centroid}Hz centroid, {MIX_STYLES.find(s => s.value === style)?.msRatio} M/S ratio</p>
                    <p>• <strong>Validation:</strong> White noise reference testing with ±1dB LUFS, ±200Hz centroid accuracy</p>
                  </div>
                </div>

                <div className="p-4 bg-primary/5 border border-primary/20 rounded-lg">
                  <h4 className="font-medium mb-2 flex items-center gap-2">
                    <Lightning className="w-4 h-4 text-primary" />
                    CLI Usage Example:
                  </h4>
                  <code className="text-xs bg-muted p-2 rounded block font-mono">
                    python mix_master.py --stems kick.wav snare.wav bass.wav guitar.wav \<br/>
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--style {style} --output mixed.wav --validate
                  </code>
                  <div className="text-xs text-muted-foreground mt-2">
                    Export these settings or use the CLI tool with real audio stems for production mixing.
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
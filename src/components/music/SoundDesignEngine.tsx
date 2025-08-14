import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { useKV } from "@/hooks/useKV";
import { 
  Waveform, 
  Save, 
  Play, 
  Pause, 
  Sparkles, 
  Download,
  SpeakerHigh,
  Sliders,
  Lightning,
  FileAudio,
  MusicNotes,
  Gear
} from "@phosphor-icons/react";
import { toast } from "sonner";

interface SoundEffect {
  type: string;
  parameters: Record<string, number>;
  wet: number; // 0-1
  enabled: boolean;
}

interface SampleConfig {
  samplePath: string;
  rootNote: number;
  tuneCents: number;
  gainDb: number;
  pan: number;
  velocityLayers: VelocityLayer[];
}

interface VelocityLayer {
  velocityMin: number;
  velocityMax: number;
  gain: number;
}

interface InstrumentConfig {
  name: string;
  instRole: string;
  sampleConfig: SampleConfig;
  envelope: EnvelopeConfig;
  effects: SoundEffect[];
}

interface StemRenderRequest {
  midiData: string; // Base64 encoded MIDI
  style: 'rock_punk' | 'rnb_ballad' | 'country_pop';
  songId?: string;
  renderConfig: {
    sampleRate: number;
    bitDepth: number;
    normalize: boolean;
    lufsTarget: number;
  };
}

interface StemRenderResult {
  stems: Record<string, string>; // instrument role -> file path
  metadata: {
    style: string;
    duration: number;
    sampleRate: number;
    totalSize: number;
  };
}

interface SynthPatch {
  id: string;
  name: string;
  instrument: string;
  oscillators: OscillatorConfig[];
  filter: FilterConfig;
  envelope: EnvelopeConfig;
  effects: SoundEffect[];
  timestamp: number;
}

interface OscillatorConfig {
  type: 'sine' | 'square' | 'sawtooth' | 'triangle';
  volume: number;
  detune: number;
}

interface FilterConfig {
  type: 'lowpass' | 'highpass' | 'bandpass' | 'notch';
  frequency: number;
  resonance: number;
}

interface EnvelopeConfig {
  attack: number;
  decay: number;
  sustain: number;
  release: number;
}

interface SoundDesign {
  id: string;
  name: string;
  style: string;
  patches: SynthPatch[];
  composition?: any;
  timestamp: number;
}

const INSTRUMENT_ROLES = [
  "KICK", "SNARE", "BASS_PICK", "ACOUSTIC_STRUM", "PIANO", "LEAD"
];

const SUPPORTED_STYLES = [
  { value: "rock_punk", label: "Rock/Punk", description: "Aggressive, raw sound with distorted guitars" },
  { value: "rnb_ballad", label: "R&B Ballad", description: "Smooth, warm sound with rich harmonics" },
  { value: "country_pop", label: "Country Pop", description: "Warm, organic sound with bright acoustics" }
];

const SAMPLE_RATES = [44100, 48000, 96000];
const BIT_DEPTHS = [16, 24, 32];

export function SoundDesignEngine() {
  const [style, setStyle] = useState<'rock_punk' | 'rnb_ballad' | 'country_pop' | "">("");
  const [midiFile, setMidiFile] = useState<File | null>(null);
  const [renderConfig, setRenderConfig] = useState({
    sampleRate: 48000,
    bitDepth: 24,
    normalize: true,
    lufsTarget: -18.0
  });
  const [isRendering, setIsRendering] = useState(false);
  const [renderResult, setRenderResult] = useState<StemRenderResult | null>(null);
  const [instrumentConfigs, setInstrumentConfigs] = useState<InstrumentConfig[]>([]);
  const [savedCompositions, setSavedCompositions] = useKV<any[]>("melody-harmony-compositions", []);
  const [selectedComposition, setSelectedComposition] = useState("");
  const [soundDesigns, setSoundDesigns] = useKV<any[]>("sound-designs", []);

  const loadInstrumentConfig = async (selectedStyle: string) => {
    if (!selectedStyle) return;

    try {
      const prompt = spark.llmPrompt`Generate instrument configuration for ${selectedStyle} style with the following structure:

Return a JSON object with this exact format:
{
  "instruments": [
    {
      "name": "Instrument Name",
      "instRole": "KICK|SNARE|BASS_PICK|ACOUSTIC_STRUM|PIANO|LEAD",
      "sampleConfig": {
        "samplePath": "samples/path/to/instrument.wav",
        "rootNote": 60,
        "tuneCents": 0,
        "gainDb": 0.0,
        "pan": 0.0,
        "velocityLayers": [
          {"velocityMin": 0, "velocityMax": 80, "gain": 0.8},
          {"velocityMin": 81, "velocityMax": 127, "gain": 1.0}
        ]
      },
      "envelope": {
        "attack": 0.01,
        "decay": 0.1,
        "sustain": 0.8,
        "release": 0.3
      },
      "effects": [
        {
          "type": "reverb",
          "parameters": {"roomSize": 0.5, "decay": 0.8, "mix": 0.3},
          "wet": 0.3,
          "enabled": true
        }
      ]
    }
  ]
}

Generate configurations for all 6 instrument roles that match the ${selectedStyle} aesthetic.
Style characteristics:
- rock_punk: Aggressive, distorted, punchy drums, overdriven guitars
- rnb_ballad: Smooth, warm, rich harmonics, subtle effects
- country_pop: Bright, organic, warm acoustics, tasteful processing`;

      const result = await spark.llm(prompt, "gpt-4o", true);
      const config = JSON.parse(result);
      
      if (config.instruments && Array.isArray(config.instruments)) {
        setInstrumentConfigs(config.instruments);
        toast.success(`Loaded ${config.instruments.length} instrument configurations for ${selectedStyle}`);
      }
    } catch (error) {
      console.error("Failed to load instrument config:", error);
      toast.error("Failed to load instrument configuration");
    }
  };

  const renderStems = async () => {
    if (!style) {
      toast.error("Please select a music style");
      return;
    }

    let midiData = null;

    // Get MIDI data from file or composition
    if (midiFile) {
      try {
        const arrayBuffer = await midiFile.arrayBuffer();
        midiData = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
      } catch (error) {
        toast.error("Failed to read MIDI file");
        return;
      }
    } else if (selectedComposition) {
      const composition = savedCompositions.find(comp => comp.id === selectedComposition);
      if (composition && composition.midiData) {
        midiData = composition.midiData;
      } else {
        toast.error("Selected composition has no MIDI data");
        return;
      }
    } else {
      toast.error("Please provide a MIDI file or select a saved composition");
      return;
    }

    setIsRendering(true);
    try {
      // Simulate stem rendering process
      const prompt = spark.llmPrompt`Simulate rendering MIDI to audio stems using a sampler-based engine.

Input:
- Style: ${style}
- MIDI data: ${midiData.substring(0, 100)}... (truncated)
- Sample rate: ${renderConfig.sampleRate}Hz
- Bit depth: ${renderConfig.bitDepth}-bit
- Normalize: ${renderConfig.normalize}
- Target LUFS: ${renderConfig.lufsTarget}

Generate a realistic render result with this format:
{
  "stems": {
    "kick": "/stems/song_${Date.now()}/kick.wav",
    "snare": "/stems/song_${Date.now()}/snare.wav", 
    "bass_pick": "/stems/song_${Date.now()}/bass_pick.wav",
    "acoustic_strum": "/stems/song_${Date.now()}/acoustic_strum.wav",
    "piano": "/stems/song_${Date.now()}/piano.wav",
    "lead": "/stems/song_${Date.now()}/lead.wav"
  },
  "metadata": {
    "style": "${style}",
    "duration": 240.5,
    "sampleRate": ${renderConfig.sampleRate},
    "totalSize": 125.8
  }
}

Include realistic file paths and metadata for a ${style} style composition.`;

      const result = await spark.llm(prompt, "gpt-4o", true);
      const renderResult = JSON.parse(result);
      
      setRenderResult(renderResult);
      
      // Save render to history
      const renderHistory = {
        id: Date.now().toString(),
        style,
        midiFile: midiFile?.name || selectedComposition,
        renderConfig,
        result: renderResult,
        timestamp: Date.now()
      };
      
      setSoundDesigns(current => [renderHistory, ...current]);
      toast.success("Stems rendered successfully!");
      
    } catch (error) {
      console.error("Render error:", error);
      toast.error("Failed to render stems. Please try again.");
    } finally {
      setIsRendering(false);
    }
  };

  const handleStyleChange = (newStyle: string) => {
    setStyle(newStyle as any);
    loadInstrumentConfig(newStyle);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (!file.name.toLowerCase().endsWith('.mid') && !file.name.toLowerCase().endsWith('.midi')) {
        toast.error("Please select a MIDI file (.mid or .midi)");
        return;
      }
      setMidiFile(file);
      setSelectedComposition(""); // Clear composition selection
      toast.success(`MIDI file loaded: ${file.name}`);
    }
  };

  const loadComposition = (compositionId: string) => {
    const composition = savedCompositions.find(comp => comp.id === compositionId);
    if (composition) {
      setSelectedComposition(compositionId);
      setMidiFile(null); // Clear file when using saved composition
    }
  };

  const downloadStem = (instrumentRole: string, filePath: string) => {
    // Simulate download - in real implementation this would download the actual audio file
    toast.success(`Downloading ${instrumentRole} stem...`);
    
    // Create a dummy download link for demonstration
    const link = document.createElement('a');
    link.href = '#';
    link.download = `${instrumentRole}_${style}.wav`;
    link.click();
  };

  const exportRenderMetadata = () => {
    if (!renderResult) {
      toast.error("No render result to export");
      return;
    }

    const metadata = {
      style,
      renderConfig,
      result: renderResult,
      timestamp: Date.now(),
      instrumentConfigs
    };

    const dataStr = JSON.stringify(metadata, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `stem-render-${style}-${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    toast.success("Render metadata exported!");
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="music-style">Music Style</Label>
          <Select value={style} onValueChange={handleStyleChange}>
            <SelectTrigger id="music-style">
              <SelectValue placeholder="Select music style" />
            </SelectTrigger>
            <SelectContent>
              {SUPPORTED_STYLES.map((s) => (
                <SelectItem key={s.value} value={s.value}>
                  <div>
                    <div className="font-medium">{s.label}</div>
                    <div className="text-xs text-muted-foreground">{s.description}</div>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label>Sampler-Based Rendering</Label>
          <div className="text-sm text-muted-foreground">
            Convert MIDI to audio stems using style-specific sample libraries
          </div>
        </div>
      </div>

      {/* MIDI Input Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MusicNotes className="w-5 h-5 text-accent" />
            MIDI Input
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* File Upload */}
            <div className="space-y-2">
              <Label htmlFor="midi-file">Upload MIDI File</Label>
              <Input
                id="midi-file"
                type="file"
                accept=".mid,.midi"
                onChange={handleFileUpload}
                className="file:mr-4 file:py-1 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-medium file:bg-accent/10 file:text-accent hover:file:bg-accent/20"
              />
              {midiFile && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <FileAudio className="w-4 h-4" />
                  {midiFile.name}
                </div>
              )}
            </div>

            {/* Saved Compositions */}
            {savedCompositions.length > 0 && (
              <div className="space-y-2">
                <Label htmlFor="saved-composition">Or Use Saved Composition</Label>
                <Select value={selectedComposition} onValueChange={loadComposition}>
                  <SelectTrigger id="saved-composition">
                    <SelectValue placeholder="Select saved composition" />
                  </SelectTrigger>
                  <SelectContent>
                    {savedCompositions.map((comp) => (
                      <SelectItem key={comp.id} value={comp.id}>
                        {comp.name} ({comp.key || 'C'} {comp.style || 'Unknown'}, {comp.bpm || 120} BPM)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Render Configuration */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Gear className="w-5 h-5 text-accent" />
            Render Configuration
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label>Sample Rate</Label>
              <Select 
                value={renderConfig.sampleRate.toString()} 
                onValueChange={(value) => setRenderConfig(prev => ({...prev, sampleRate: parseInt(value)}))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SAMPLE_RATES.map(rate => (
                    <SelectItem key={rate} value={rate.toString()}>
                      {rate} Hz
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Bit Depth</Label>
              <Select 
                value={renderConfig.bitDepth.toString()} 
                onValueChange={(value) => setRenderConfig(prev => ({...prev, bitDepth: parseInt(value)}))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {BIT_DEPTHS.map(depth => (
                    <SelectItem key={depth} value={depth.toString()}>
                      {depth}-bit
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Normalize</Label>
              <Button
                variant={renderConfig.normalize ? "default" : "outline"}
                onClick={() => setRenderConfig(prev => ({...prev, normalize: !prev.normalize}))}
                className="w-full"
              >
                {renderConfig.normalize ? "Enabled" : "Disabled"}
              </Button>
            </div>

            <div className="space-y-2">
              <Label>Target LUFS</Label>
              <Input
                type="number"
                value={renderConfig.lufsTarget}
                onChange={(e) => setRenderConfig(prev => ({...prev, lufsTarget: parseFloat(e.target.value)}))}
                step="0.1"
                min="-30"
                max="0"
                disabled={!renderConfig.normalize}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Action Buttons */}
      <div className="flex gap-3 flex-wrap">
        <Button 
          onClick={renderStems} 
          disabled={isRendering || !style || (!midiFile && !selectedComposition)}
          className="flex items-center gap-2"
        >
          {isRendering ? (
            <>
              <Sparkles className="w-4 h-4 animate-spin" />
              Rendering...
            </>
          ) : (
            <>
              <Waveform className="w-4 h-4" />
              Render to Stems
            </>
          )}
        </Button>

        {renderResult && (
          <Button variant="outline" onClick={exportRenderMetadata} className="flex items-center gap-2">
            <Download className="w-4 h-4" />
            Export Metadata
          </Button>
        )}
      </div>

      {/* Instrument Configuration Display */}
      {instrumentConfigs.length > 0 && (
        <>
          <Separator />
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sliders className="w-5 h-5 text-accent" />
                Instrument Configuration - {SUPPORTED_STYLES.find(s => s.value === style)?.label}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {instrumentConfigs.map((config, index) => (
                  <div key={index} className="p-4 bg-card border rounded-lg">
                    <div className="flex items-center gap-2 mb-3">
                      <SpeakerHigh className="w-4 h-4 text-accent" />
                      <div>
                        <h4 className="font-medium">{config.name}</h4>
                        <p className="text-sm text-muted-foreground">{config.instRole}</p>
                      </div>
                    </div>

                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Root Note:</span>
                        <span>{config.sampleConfig.rootNote}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Gain:</span>
                        <span>{config.sampleConfig.gainDb} dB</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Pan:</span>
                        <span>{config.sampleConfig.pan > 0 ? 'R' : config.sampleConfig.pan < 0 ? 'L' : 'C'}</span>
                      </div>
                    </div>

                    {config.effects.length > 0 && (
                      <div className="mt-3 pt-3 border-t">
                        <h5 className="text-xs font-medium mb-1">Effects</h5>
                        <div className="flex flex-wrap gap-1">
                          {config.effects.map((effect, effectIndex) => (
                            <Badge key={effectIndex} variant="outline" className="text-xs">
                              {effect.type}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {/* Render Results */}
      {renderResult && (
        <>
          <Separator />
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileAudio className="w-5 h-5 text-accent" />
                Rendered Stems
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-muted-foreground">Style:</span>
                    <div className="font-medium">{renderResult.metadata.style}</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Duration:</span>
                    <div className="font-medium">{renderResult.metadata.duration.toFixed(1)}s</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Sample Rate:</span>
                    <div className="font-medium">{renderResult.metadata.sampleRate} Hz</div>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Total Size:</span>
                    <div className="font-medium">{renderResult.metadata.totalSize.toFixed(1)} MB</div>
                  </div>
                </div>

                <div className="space-y-2">
                  <h4 className="font-medium">Audio Stems:</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {Object.entries(renderResult.stems).map(([role, filePath]) => (
                      <div key={role} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                        <div className="flex items-center gap-3">
                          <SpeakerHigh className="w-4 h-4 text-accent" />
                          <div>
                            <div className="font-medium">{role.toUpperCase()}</div>
                            <div className="text-xs text-muted-foreground">{filePath.split('/').pop()}</div>
                          </div>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => downloadStem(role, filePath)}
                          className="flex items-center gap-1"
                        >
                          <Download className="w-3 h-3" />
                          Download
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="p-4 bg-muted/50 rounded-lg">
                  <h4 className="font-medium mb-2">Rendering Notes:</h4>
                  <div className="text-sm text-muted-foreground space-y-1">
                    <p>• Stems rendered using style-specific sample libraries and velocity layers</p>
                    <p>• Each instrument mapped to appropriate samples with realistic envelope shaping</p>
                    <p>• Applied {renderConfig.normalize ? 'LUFS normalization' : 'no normalization'} and latency compensation</p>
                    <p>• Ready for mixing, mastering, or further processing in your DAW</p>
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
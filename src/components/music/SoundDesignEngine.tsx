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
  Lightning
} from "@phosphor-icons/react";
import { toast } from "sonner";

interface SoundEffect {
  type: string;
  parameters: Record<string, number>;
  wet: number; // 0-1
  enabled: boolean;
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

const INSTRUMENT_TYPES = [
  "Lead Synth", "Bass Synth", "Pad", "Pluck", "Brass", "Strings", 
  "Electric Piano", "Organ", "Guitar", "Drums", "Percussion", "FX"
];

const EFFECT_TYPES = [
  "Reverb", "Delay", "Chorus", "Distortion", "Compression", 
  "EQ", "Phaser", "Flanger", "Filter", "Bitcrush", "Saturation"
];

const SOUND_STYLES = [
  "Vintage Analog", "Modern Digital", "Lo-Fi", "Ambient", "Aggressive", 
  "Warm Tube", "Clean Digital", "Experimental", "Cinematic", "Retro Wave"
];

export function SoundDesignEngine() {
  const [style, setStyle] = useState("");
  const [compositionJSON, setCompositionJSON] = useState("");
  const [generatedPatches, setGeneratedPatches] = useState<SynthPatch[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [soundDesigns, setSoundDesigns] = useKV<SoundDesign[]>("sound-designs", []);
  const [savedCompositions, setSavedCompositions] = useKV<any[]>("melody-harmony-compositions", []);
  const [selectedComposition, setSelectedComposition] = useState("");

  const generateSoundDesign = async () => {
    if (!style) {
      toast.error("Please select a sound style");
      return;
    }

    let compositionData = null;

    // Parse composition if provided
    if (compositionJSON.trim()) {
      try {
        compositionData = JSON.parse(compositionJSON);
      } catch (e) {
        toast.error("Invalid composition JSON format");
        return;
      }
    } else if (selectedComposition) {
      const selected = savedCompositions.find(comp => comp.id === selectedComposition);
      if (selected) {
        compositionData = selected;
      }
    }

    setIsGenerating(true);
    try {
      const prompt = spark.llmPrompt`Generate synthesizer patches and sound design for a ${style} style composition.

${compositionData ? `Base this on the following composition:
${JSON.stringify(compositionData, null, 2)}

Create patches that match the instruments specified in the composition tracks.` : 'Create a complete set of patches for a typical band arrangement.'}

Return a JSON object with this exact format:
{
  "patches": [
    {
      "id": "patch_1",
      "name": "Lead Synth",
      "instrument": "Lead Synth",
      "oscillators": [
        {
          "type": "sawtooth",
          "volume": 0.8,
          "detune": 0
        },
        {
          "type": "square", 
          "volume": 0.3,
          "detune": -7
        }
      ],
      "filter": {
        "type": "lowpass",
        "frequency": 2400,
        "resonance": 0.3
      },
      "envelope": {
        "attack": 0.1,
        "decay": 0.2,
        "sustain": 0.7,
        "release": 0.5
      },
      "effects": [
        {
          "type": "Reverb",
          "parameters": {
            "roomSize": 0.4,
            "decay": 0.6,
            "highCut": 0.8
          },
          "wet": 0.3,
          "enabled": true
        }
      ]
    }
  ]
}

Requirements:
- Generate patches for all instruments needed (Lead, Bass, Pads, Drums, etc.)
- Use parameters appropriate for ${style} sound aesthetic
- Include realistic synthesizer parameters (frequencies 20-20000Hz, envelopes 0.01-5.0s)
- Add effects that enhance the ${style} character
- Ensure patches work well together in a mix
- Use appropriate oscillator combinations for each instrument type
- Consider the musical key and style from the composition data`;

      const result = await spark.llm(prompt, "gpt-4o", true);
      const soundDesign = JSON.parse(result);
      
      // Validate the response structure
      if (!soundDesign.patches || !Array.isArray(soundDesign.patches)) {
        throw new Error("Invalid response structure: missing patches array");
      }
      
      setGeneratedPatches(soundDesign.patches);
      toast.success("Sound design patches generated!");
    } catch (error) {
      console.error("Generation error:", error);
      toast.error("Failed to generate sound design. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const playSoundDesign = () => {
    if (generatedPatches.length === 0) {
      toast.error("No sound design to preview");
      return;
    }

    setIsPlaying(true);
    
    // Simulate sound playback - in a real implementation, this would use Web Audio API
    // to synthesize the actual sounds based on the patch parameters
    setTimeout(() => {
      setIsPlaying(false);
      toast.success("Sound design preview completed!");
    }, 8000);
  };

  const saveSoundDesign = () => {
    if (generatedPatches.length === 0) {
      toast.error("No sound design to save");
      return;
    }

    let compositionData = null;
    
    if (selectedComposition) {
      compositionData = savedCompositions.find(comp => comp.id === selectedComposition);
    } else if (compositionJSON.trim()) {
      try {
        compositionData = JSON.parse(compositionJSON);
      } catch (e) {
        toast.error("Invalid composition JSON format");
        return;
      }
    }

    const newSoundDesign: SoundDesign = {
      id: Date.now().toString(),
      name: `${style} Sound Design`,
      style,
      patches: generatedPatches,
      composition: compositionData,
      timestamp: Date.now()
    };

    setSoundDesigns(current => [newSoundDesign, ...current]);
    toast.success("Sound design saved to history!");
  };

  const exportPatches = () => {
    if (generatedPatches.length === 0) {
      toast.error("No patches to export");
      return;
    }

    const patchData = {
      name: `${style} Sound Design`,
      style,
      patches: generatedPatches.map(patch => ({
        ...patch,
        metadata: {
          generatedAt: new Date().toISOString(),
          style,
          version: "1.0"
        }
      }))
    };

    const dataStr = JSON.stringify(patchData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `sound-design-${style.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    toast.success("Sound design patches exported!");
  };

  const loadComposition = (compositionId: string) => {
    const composition = savedCompositions.find(comp => comp.id === compositionId);
    if (composition) {
      setSelectedComposition(compositionId);
      setCompositionJSON(""); // Clear manual JSON when using saved composition
    }
  };

  const formatEffect = (effect: SoundEffect) => {
    const paramStr = Object.entries(effect.parameters)
      .map(([key, value]) => `${key}: ${typeof value === 'number' ? value.toFixed(2) : value}`)
      .join(', ');
    return `${effect.type} (${paramStr}, ${Math.round(effect.wet * 100)}% wet)`;
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="sound-style">Sound Style</Label>
          <Select value={style} onValueChange={setStyle}>
            <SelectTrigger id="sound-style">
              <SelectValue placeholder="Select sound style" />
            </SelectTrigger>
            <SelectContent>
              {SOUND_STYLES.map((s) => (
                <SelectItem key={s} value={s.toLowerCase()}>
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label>Data Flow Input</Label>
          <div className="text-sm text-muted-foreground">
            Use composition data to generate matching sound design
          </div>
        </div>
      </div>

      {/* Composition Input Section */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <h3 className="text-lg font-medium">Composition Input (Optional)</h3>
            <p className="text-sm text-muted-foreground">
              Load a saved composition or provide custom MIDI data to generate sound patches that match the instruments and style.
            </p>

            {savedCompositions.length > 0 && (
              <div className="space-y-2">
                <Label htmlFor="saved-composition">Use Saved Composition</Label>
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

            <div className="space-y-2">
              <Label htmlFor="composition-json">Or Paste Composition JSON</Label>
              <Textarea
                id="composition-json"
                placeholder="Paste exported composition JSON here..."
                value={compositionJSON}
                onChange={(e) => {
                  setCompositionJSON(e.target.value);
                  if (e.target.value.trim()) {
                    setSelectedComposition(""); // Clear saved selection when using manual JSON
                  }
                }}
                className="font-mono text-sm"
                rows={4}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex gap-3 flex-wrap">
        <Button 
          onClick={generateSoundDesign} 
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
              <Waveform className="w-4 h-4" />
              Generate Sound Design
            </>
          )}
        </Button>

        {generatedPatches.length > 0 && (
          <>
            <Button 
              variant="secondary" 
              onClick={playSoundDesign}
              disabled={isPlaying}
              className="flex items-center gap-2"
            >
              {isPlaying ? (
                <>
                  <Pause className="w-4 h-4" />
                  Playing...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Preview Sounds
                </>
              )}
            </Button>

            <Button variant="outline" onClick={saveSoundDesign} className="flex items-center gap-2">
              <Save className="w-4 h-4" />
              Save
            </Button>

            <Button variant="outline" onClick={exportPatches} className="flex items-center gap-2">
              <Download className="w-4 h-4" />
              Export Patches
            </Button>
          </>
        )}
      </div>

      {generatedPatches.length > 0 && (
        <>
          <Separator />
          <Card>
            <CardContent className="pt-6">
              <div className="space-y-6">
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <h3 className="text-lg font-semibold">Generated Sound Design</h3>
                  <div className="flex gap-2 flex-wrap">
                    <Badge variant="secondary">{style}</Badge>
                    <Badge variant="outline">{generatedPatches.length} Patches</Badge>
                  </div>
                </div>

                {/* Patch Overview */}
                <div className="space-y-4">
                  {generatedPatches.map((patch, index) => (
                    <div
                      key={index}
                      className="p-4 bg-card border rounded-lg hover:bg-accent/5 transition-colors"
                    >
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex items-center gap-3">
                          <div className="p-2 bg-accent/10 rounded-lg">
                            <SpeakerHigh className="w-5 h-5 text-accent" />
                          </div>
                          <div>
                            <h4 className="font-medium">{patch.name}</h4>
                            <p className="text-sm text-muted-foreground">
                              {patch.instrument}
                            </p>
                          </div>
                        </div>
                        <Badge variant="outline" className="text-xs">
                          {patch.oscillators.length} OSC
                        </Badge>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
                        {/* Oscillators */}
                        <div>
                          <h5 className="font-medium mb-2 flex items-center gap-2">
                            <Lightning className="w-4 h-4" />
                            Oscillators
                          </h5>
                          <div className="space-y-1 text-sm">
                            {patch.oscillators.map((osc, oscIndex) => (
                              <div key={oscIndex} className="flex justify-between">
                                <span className="text-muted-foreground">{osc.type}</span>
                                <span>Vol: {Math.round(osc.volume * 100)}% Det: {osc.detune}</span>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Filter */}
                        <div>
                          <h5 className="font-medium mb-2 flex items-center gap-2">
                            <Sliders className="w-4 h-4" />
                            Filter
                          </h5>
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Type:</span>
                              <span>{patch.filter.type}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Freq:</span>
                              <span>{patch.filter.frequency}Hz</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Res:</span>
                              <span>{patch.filter.resonance.toFixed(2)}</span>
                            </div>
                          </div>
                        </div>

                        {/* Envelope */}
                        <div>
                          <h5 className="font-medium mb-2">ADSR</h5>
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">A:</span>
                              <span>{patch.envelope.attack}s</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">D:</span>
                              <span>{patch.envelope.decay}s</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">S:</span>
                              <span>{patch.envelope.sustain.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">R:</span>
                              <span>{patch.envelope.release}s</span>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Effects */}
                      {patch.effects.length > 0 && (
                        <div className="pt-3 border-t">
                          <h5 className="font-medium mb-2">Effects Chain</h5>
                          <div className="flex flex-wrap gap-1">
                            {patch.effects.map((effect, effectIndex) => (
                              <Badge 
                                key={effectIndex} 
                                variant={effect.enabled ? "default" : "secondary"}
                                className="text-xs"
                                title={formatEffect(effect)}
                              >
                                {effect.type} {Math.round(effect.wet * 100)}%
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                <div className="p-4 bg-muted/50 rounded-lg">
                  <h4 className="font-medium mb-2">Sound Design Notes:</h4>
                  <div className="text-sm text-muted-foreground space-y-1">
                    <p>• Each patch contains oscillator settings, filter parameters, and envelope shaping</p>
                    <p>• Effects are configured to enhance the {style} aesthetic</p>
                    <p>• Patches are designed to work together in a cohesive mix</p>
                    <p>• Export to use with synthesizers, samplers, or digital audio workstations</p>
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
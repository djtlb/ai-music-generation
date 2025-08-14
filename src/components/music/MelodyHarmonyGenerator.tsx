import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useKV } from "@github/spark/hooks";
import { 
  Music, 
  Save, 
  Play, 
  Pause, 
  Sparkles, 
  Download,
  Upload,
  MusicNote,
  Piano
} from "@phosphor-icons/react";
import { toast } from "sonner";

interface MIDITrack {
  name: string;
  channel: number;
  notes: MIDINote[];
  instrument: string;
}

interface MIDINote {
  pitch: number;
  start: number;
  duration: number;
  velocity: number;
  chord?: string;
}

interface MelodyHarmony {
  id: string;
  name: string;
  key: string;
  style: string;
  bpm: number;
  arrangement?: any;
  tracks: MIDITrack[];
  timestamp: number;
}

interface SongSection {
  name: string;
  duration: number;
  bars: number;
  description: string;
  bpm?: number;
  startTime?: number;
}

const KEYS = [
  "C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"
];

const MUSIC_STYLES = [
  "Pop", "Rock", "Jazz", "Blues", "Country", "Folk", "Electronic", 
  "R&B", "Indie", "Classical", "Hip-Hop", "Reggae", "Funk", "Gospel"
];

const CHORD_QUALITIES = ["major", "minor", "dominant7", "major7", "minor7", "suspended", "diminished"];

export function MelodyHarmonyGenerator() {
  const [key, setKey] = useState("");
  const [style, setStyle] = useState("");
  const [bpm, setBpm] = useState("");
  const [arrangementJSON, setArrangementJSON] = useState("");
  const [generatedTracks, setGeneratedTracks] = useState<MIDITrack[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [compositions, setCompositions] = useKV<MelodyHarmony[]>("melody-harmony-compositions", []);
  const [savedArrangements, setSavedArrangements] = useKV<any[]>("song-structures", []);
  const [selectedArrangement, setSelectedArrangement] = useState("");

  const generateComposition = async () => {
    if (!key || !style) {
      toast.error("Please select key and style");
      return;
    }

    let arrangementData = null;
    let finalBpm = parseInt(bpm) || 120;

    // Parse arrangement if provided
    if (arrangementJSON.trim()) {
      try {
        arrangementData = JSON.parse(arrangementJSON);
        if (arrangementData.bpm) {
          finalBpm = arrangementData.bpm;
        }
      } catch (e) {
        toast.error("Invalid arrangement JSON format");
        return;
      }
    } else if (selectedArrangement) {
      const selected = savedArrangements.find(arr => arr.id === selectedArrangement);
      if (selected) {
        arrangementData = selected;
        finalBpm = selected.bpm;
      }
    }

    setIsGenerating(true);
    try {
      const prompt = spark.llmPrompt`Generate a complete MIDI composition with melody, chords, and bassline in ${key} key, ${style} style at ${finalBpm} BPM.

${arrangementData ? `Use this song arrangement as the structure:
${JSON.stringify(arrangementData, null, 2)}

Generate music that follows this arrangement with appropriate melodies and harmonies for each section.` : 'Create a standard song structure with intro, verse, chorus, verse, chorus, bridge, chorus, outro.'}

Return a JSON object with this exact format:
{
  "tracks": [
    {
      "name": "Melody",
      "channel": 1,
      "instrument": "Piano",
      "notes": [
        {
          "pitch": 60,
          "start": 0.0,
          "duration": 1.0,
          "velocity": 80,
          "chord": "C"
        }
      ]
    },
    {
      "name": "Chords", 
      "channel": 2,
      "instrument": "Guitar",
      "notes": [
        {
          "pitch": 60,
          "start": 0.0,
          "duration": 4.0,
          "velocity": 70,
          "chord": "C"
        }
      ]
    },
    {
      "name": "Bass",
      "channel": 3, 
      "instrument": "Bass",
      "notes": [
        {
          "pitch": 36,
          "start": 0.0,
          "duration": 1.0,
          "velocity": 85,
          "chord": "C"
        }
      ]
    }
  ]
}

Requirements:
- Generate realistic MIDI notes with proper pitch values (C4 = 60)
- Create melody lines appropriate for ${style} music
- Generate chord progressions that fit ${key} major/minor scale
- Include basslines that support the harmony
- Use appropriate velocities (40-127)
- Notes should span ${arrangementData ? 'the duration of the provided arrangement' : '3-4 minutes'}
- Consider ${style} characteristics in rhythm and harmony
- Use instruments typical for ${style} music
- Generate chord symbols for harmonic reference
- Ensure notes follow music theory rules for ${key} key`;

      const result = await spark.llm(prompt, "gpt-4o", true);
      const composition = JSON.parse(result);
      
      setGeneratedTracks(composition.tracks);
      toast.success("Melody & harmony composition generated!");
    } catch (error) {
      console.error("Generation error:", error);
      toast.error("Failed to generate composition. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const playComposition = () => {
    if (generatedTracks.length === 0) {
      toast.error("No composition to play");
      return;
    }

    setIsPlaying(true);
    
    // Simulate playback - in a real implementation, this would use Web Audio API
    // to actually play the MIDI data
    setTimeout(() => {
      setIsPlaying(false);
      toast.success("Composition preview completed!");
    }, 5000);
  };

  const saveComposition = () => {
    if (generatedTracks.length === 0) {
      toast.error("No composition to save");
      return;
    }

    const arrangementData = selectedArrangement ? 
      savedArrangements.find(arr => arr.id === selectedArrangement) : 
      (arrangementJSON.trim() ? JSON.parse(arrangementJSON) : null);

    const newComposition: MelodyHarmony = {
      id: Date.now().toString(),
      name: `${key} ${style} Composition`,
      key,
      style,
      bpm: parseInt(bpm) || 120,
      arrangement: arrangementData,
      tracks: generatedTracks,
      timestamp: Date.now()
    };

    setCompositions(current => [newComposition, ...current]);
    toast.success("Composition saved to history!");
  };

  const exportMIDI = () => {
    if (generatedTracks.length === 0) {
      toast.error("No composition to export");
      return;
    }

    const midiData = {
      format: 1,
      division: 480,
      tracks: generatedTracks.map(track => ({
        name: track.name,
        channel: track.channel,
        instrument: track.instrument,
        events: track.notes.map(note => ({
          type: 'note',
          pitch: note.pitch,
          start: Math.round(note.start * 480), // Convert to ticks
          duration: Math.round(note.duration * 480),
          velocity: note.velocity,
          chord: note.chord
        }))
      }))
    };

    const dataStr = JSON.stringify(midiData, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `composition-${key}-${style}-${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    toast.success("MIDI composition exported!");
  };

  const loadArrangement = (arrangementId: string) => {
    const arrangement = savedArrangements.find(arr => arr.id === arrangementId);
    if (arrangement) {
      setSelectedArrangement(arrangementId);
      setBpm(arrangement.bpm.toString());
      setArrangementJSON(""); // Clear manual JSON when using saved arrangement
    }
  };

  const getTrackSummary = (track: MIDITrack) => {
    const noteCount = track.notes.length;
    const duration = Math.max(...track.notes.map(n => n.start + n.duration));
    const avgPitch = Math.round(track.notes.reduce((sum, n) => sum + n.pitch, 0) / noteCount);
    
    return {
      noteCount,
      duration: Math.round(duration),
      avgPitch,
      range: `${Math.min(...track.notes.map(n => n.pitch))} - ${Math.max(...track.notes.map(n => n.pitch))}`
    };
  };

  const pitchToNote = (pitch: number) => {
    const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const octave = Math.floor(pitch / 12) - 1;
    const noteIndex = pitch % 12;
    return `${notes[noteIndex]}${octave}`;
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="space-y-2">
          <Label htmlFor="melody-key">Key Signature</Label>
          <Select value={key} onValueChange={setKey}>
            <SelectTrigger id="melody-key">
              <SelectValue placeholder="Select key" />
            </SelectTrigger>
            <SelectContent>
              {KEYS.map((k) => (
                <SelectItem key={k} value={k}>
                  {k}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="melody-style">Musical Style</Label>
          <Select value={style} onValueChange={setStyle}>
            <SelectTrigger id="melody-style">
              <SelectValue placeholder="Select style" />
            </SelectTrigger>
            <SelectContent>
              {MUSIC_STYLES.map((s) => (
                <SelectItem key={s} value={s.toLowerCase()}>
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="melody-bpm">BPM (Tempo)</Label>
          <Input
            id="melody-bpm"
            type="number"
            placeholder="120"
            value={bpm}
            onChange={(e) => setBpm(e.target.value)}
            min="60"
            max="200"
          />
        </div>
      </div>

      {/* Arrangement Input Section */}
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-4">
            <h3 className="text-lg font-medium">Song Arrangement (Optional)</h3>
            <p className="text-sm text-muted-foreground">
              Use a saved arrangement or provide custom structure JSON to generate music that follows a specific song form.
            </p>

            {savedArrangements.length > 0 && (
              <div className="space-y-2">
                <Label htmlFor="saved-arrangement">Use Saved Arrangement</Label>
                <Select value={selectedArrangement} onValueChange={loadArrangement}>
                  <SelectTrigger id="saved-arrangement">
                    <SelectValue placeholder="Select saved arrangement" />
                  </SelectTrigger>
                  <SelectContent>
                    {savedArrangements.map((arr) => (
                      <SelectItem key={arr.id} value={arr.id}>
                        {arr.name} ({arr.songType}, {arr.bpm} BPM)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="arrangement-json">Or Paste Arrangement JSON</Label>
              <Textarea
                id="arrangement-json"
                placeholder="Paste exported arrangement JSON here..."
                value={arrangementJSON}
                onChange={(e) => {
                  setArrangementJSON(e.target.value);
                  if (e.target.value.trim()) {
                    setSelectedArrangement(""); // Clear saved selection when using manual JSON
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
          onClick={generateComposition} 
          disabled={isGenerating || !key || !style}
          className="flex items-center gap-2"
        >
          {isGenerating ? (
            <>
              <Sparkles className="w-4 h-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <MusicNote className="w-4 h-4" />
              Generate Composition
            </>
          )}
        </Button>

        {generatedTracks.length > 0 && (
          <>
            <Button 
              variant="secondary" 
              onClick={playComposition}
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
                  Preview
                </>
              )}
            </Button>

            <Button variant="outline" onClick={saveComposition} className="flex items-center gap-2">
              <Save className="w-4 h-4" />
              Save
            </Button>

            <Button variant="outline" onClick={exportMIDI} className="flex items-center gap-2">
              <Download className="w-4 h-4" />
              Export MIDI
            </Button>
          </>
        )}
      </div>

      {generatedTracks.length > 0 && (
        <>
          <Separator />
          <Card>
            <CardContent className="pt-6">
              <div className="space-y-6">
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <h3 className="text-lg font-semibold">Generated Composition</h3>
                  <div className="flex gap-2 flex-wrap">
                    <Badge variant="secondary">{key}</Badge>
                    <Badge variant="secondary">{style}</Badge>
                    <Badge variant="outline">{bpm || '120'} BPM</Badge>
                  </div>
                </div>

                {/* Track Overview */}
                <div className="space-y-4">
                  {generatedTracks.map((track, index) => {
                    const summary = getTrackSummary(track);
                    const trackIcons = {
                      'Melody': MusicNote,
                      'Chords': Piano,
                      'Bass': Music
                    };
                    const IconComponent = trackIcons[track.name as keyof typeof trackIcons] || Music;
                    
                    return (
                      <div
                        key={index}
                        className="p-4 bg-card border rounded-lg hover:bg-accent/5 transition-colors"
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center gap-3">
                            <div className="p-2 bg-accent/10 rounded-lg">
                              <IconComponent className="w-5 h-5 text-accent" />
                            </div>
                            <div>
                              <h4 className="font-medium">{track.name}</h4>
                              <p className="text-sm text-muted-foreground">
                                {track.instrument} • Channel {track.channel}
                              </p>
                            </div>
                          </div>
                          <div className="flex gap-2 flex-wrap">
                            <Badge variant="outline" className="text-xs">
                              {summary.noteCount} notes
                            </Badge>
                            <Badge variant="secondary" className="text-xs">
                              {summary.duration}s
                            </Badge>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <span className="text-muted-foreground">Average Pitch:</span>
                            <div className="font-medium">{pitchToNote(summary.avgPitch)}</div>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Range:</span>
                            <div className="font-medium">
                              {pitchToNote(Math.min(...track.notes.map(n => n.pitch)))} - {pitchToNote(Math.max(...track.notes.map(n => n.pitch)))}
                            </div>
                          </div>
                          <div>
                            <span className="text-muted-foreground">First Note:</span>
                            <div className="font-medium">
                              {track.notes.length > 0 ? pitchToNote(track.notes[0].pitch) : 'N/A'}
                            </div>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Chord Context:</span>
                            <div className="font-medium">
                              {track.notes.find(n => n.chord)?.chord || 'N/A'}
                            </div>
                          </div>
                        </div>

                        {/* Mini visualization of note pattern */}
                        <div className="mt-3 pt-3 border-t">
                          <div className="text-xs text-muted-foreground mb-2">Note Pattern (first 16 notes):</div>
                          <div className="flex gap-1 flex-wrap">
                            {track.notes.slice(0, 16).map((note, noteIndex) => (
                              <div
                                key={noteIndex}
                                className="w-3 h-6 bg-accent/20 rounded-sm"
                                style={{
                                  height: `${Math.max(12, (note.pitch - 40) * 0.8)}px`,
                                  opacity: note.velocity / 127
                                }}
                                title={`${pitchToNote(note.pitch)} (vel: ${note.velocity})`}
                              />
                            ))}
                            {track.notes.length > 16 && (
                              <span className="text-xs text-muted-foreground self-end">
                                +{track.notes.length - 16} more
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="p-4 bg-muted/50 rounded-lg">
                  <h4 className="font-medium mb-2">Composition Notes:</h4>
                  <div className="text-sm text-muted-foreground space-y-1">
                    <p>• This multi-track composition includes melody, harmonic support, and bass foundation</p>
                    <p>• Each track uses different MIDI channels for separate instrument control</p>
                    <p>• Chord symbols are embedded for harmonic reference and arrangement guidance</p>
                    <p>• Export as MIDI JSON to use with external music software or DAWs</p>
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
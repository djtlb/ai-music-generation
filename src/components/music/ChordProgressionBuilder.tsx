import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { useKV } from "@/hooks/useKV";
import { Music, Save, Play, Pause, Sparkles } from "@phosphor-icons/react";
import { toast } from "sonner";

interface ChordProgression {
  id: string;
  name: string;
  key: string;
  genre: string;
  chords: string[];
  pattern: string;
  timestamp: number;
}

const CHORD_PATTERNS = {
  "I-V-vi-IV": ["major", "major", "minor", "major"],
  "vi-IV-I-V": ["minor", "major", "major", "major"],
  "I-vi-IV-V": ["major", "minor", "major", "major"],
  "ii-V-I": ["minor", "major", "major"],
  "I-IV-V": ["major", "major", "major"],
  "vi-V-IV-V": ["minor", "major", "major", "major"]
};

export function ChordProgressionBuilder() {
  const [key, setKey] = useState("");
  const [genre, setGenre] = useState("");
  const [pattern, setPattern] = useState("");
  const [generatedChords, setGeneratedChords] = useState<string[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progressions, setProgressions] = useKV<ChordProgression[]>("chord-progressions", []);

  const keys = [
    "C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"
  ];

  const genres = [
    "Pop", "Rock", "Jazz", "Blues", "Country", "Folk", "Electronic", "R&B", "Indie", "Classical"
  ];

  const patterns = Object.keys(CHORD_PATTERNS);

  const generateProgression = async () => {
    if (!key || !genre || !pattern) {
      toast.error("Please select key, genre, and pattern");
      return;
    }

    setIsGenerating(true);
    try {
      const prompt = spark.llmPrompt`Generate a chord progression in the key of ${key} for ${genre} music using the ${pattern} pattern.

Return ONLY the chord names separated by commas, no explanations. 

For example, if the key is C major and pattern is I-V-vi-IV, return: C, G, Am, F

Make sure:
- Chords are appropriate for ${genre} style
- Chords fit the ${pattern} roman numeral pattern
- Use proper chord notation (major chords as letters, minor as letter+m, etc.)
- Include chord extensions if appropriate for ${genre} (7ths, 9ths, etc.)`;

      const result = await spark.llm(prompt);
      const chords = result.split(',').map(chord => chord.trim());
      setGeneratedChords(chords);
      toast.success("Chord progression generated!");
    } catch (error) {
      toast.error("Failed to generate progression. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const playProgression = () => {
    if (generatedChords.length === 0) {
      toast.error("No progression to play");
      return;
    }

    setIsPlaying(true);
    
    // Simulate audio playback with timeout
    setTimeout(() => {
      setIsPlaying(false);
      toast.success("Progression played!");
    }, 3000);
  };

  const saveProgression = () => {
    if (generatedChords.length === 0) {
      toast.error("No progression to save");
      return;
    }

    const newProgression: ChordProgression = {
      id: Date.now().toString(),
      name: `${key} ${genre} - ${pattern}`,
      key,
      genre,
      chords: generatedChords,
      pattern,
      timestamp: Date.now()
    };

    setProgressions(current => [newProgression, ...current]);
    toast.success("Progression saved to history!");
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="space-y-2">
          <Label htmlFor="key">Key</Label>
          <Select value={key} onValueChange={setKey}>
            <SelectTrigger id="key">
              <SelectValue placeholder="Select key" />
            </SelectTrigger>
            <SelectContent>
              {keys.map((k) => (
                <SelectItem key={k} value={k}>
                  {k}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="chord-genre">Genre</Label>
          <Select value={genre} onValueChange={setGenre}>
            <SelectTrigger id="chord-genre">
              <SelectValue placeholder="Select genre" />
            </SelectTrigger>
            <SelectContent>
              {genres.map((g) => (
                <SelectItem key={g} value={g.toLowerCase()}>
                  {g}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="pattern">Chord Pattern</Label>
          <Select value={pattern} onValueChange={setPattern}>
            <SelectTrigger id="pattern">
              <SelectValue placeholder="Select pattern" />
            </SelectTrigger>
            <SelectContent>
              {patterns.map((p) => (
                <SelectItem key={p} value={p}>
                  {p}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex gap-3">
        <Button 
          onClick={generateProgression} 
          disabled={isGenerating || !key || !genre || !pattern}
          className="flex items-center gap-2"
        >
          {isGenerating ? (
            <>
              <Sparkles className="w-4 h-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Music className="w-4 h-4" />
              Generate Progression
            </>
          )}
        </Button>

        {generatedChords.length > 0 && (
          <>
            <Button 
              variant="secondary" 
              onClick={playProgression}
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
                  Play Progression
                </>
              )}
            </Button>

            <Button variant="outline" onClick={saveProgression} className="flex items-center gap-2">
              <Save className="w-4 h-4" />
              Save
            </Button>
          </>
        )}
      </div>

      {generatedChords.length > 0 && (
        <>
          <Separator />
          <Card>
            <CardContent className="pt-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold">Generated Chord Progression</h3>
                  <div className="flex gap-2 text-sm">
                    <Badge variant="secondary">{key}</Badge>
                    <Badge variant="secondary">{genre}</Badge>
                    <Badge variant="outline">{pattern}</Badge>
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {generatedChords.map((chord, index) => (
                    <div
                      key={index}
                      className="flex flex-col items-center justify-center p-6 bg-accent/10 rounded-lg border-2 border-accent/20 hover:border-accent/40 transition-colors"
                    >
                      <div className="text-2xl font-bold text-accent mb-1">{chord}</div>
                      <div className="text-xs text-muted-foreground">
                        {pattern.split('-')[index]}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="p-4 bg-muted/50 rounded-lg">
                  <h4 className="font-medium mb-2">Progression Notes:</h4>
                  <p className="text-sm text-muted-foreground">
                    This {pattern} progression in {key} is commonly used in {genre} music. 
                    Try different strumming patterns or rhythms to match your desired style.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
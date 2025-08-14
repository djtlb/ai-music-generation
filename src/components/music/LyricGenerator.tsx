import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { useKV } from "@github/spark/hooks";
import { Wand2, Save, Sparkles } from "@phosphor-icons/react";
import { toast } from "sonner";

interface LyricComposition {
  id: string;
  title: string;
  lyrics: string;
  genre: string;
  mood: string;
  theme: string;
  timestamp: number;
}

// AI-powered lyric generation component
export function LyricGenerator() {
  const [genre, setGenre] = useState("");
  const [mood, setMood] = useState("");
  const [theme, setTheme] = useState("");
  const [generatedLyrics, setGeneratedLyrics] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [compositions, setCompositions] = useKV<LyricComposition[]>("lyric-compositions", []);

  const genres = [
    "Pop", "Rock", "Hip-Hop", "Country", "Folk", "R&B", "Electronic", "Jazz", "Blues", "Indie"
  ];

  const moods = [
    "Happy", "Sad", "Energetic", "Calm", "Romantic", "Angry", "Nostalgic", "Hopeful", "Melancholy", "Uplifting"
  ];

  const generateLyrics = async () => {
    if (!genre || !mood) {
      toast.error("Please select both genre and mood");
      return;
    }

    setIsGenerating(true);
    try {
      const prompt = spark.llmPrompt`Create original song lyrics in the ${genre} genre with a ${mood} mood${theme ? ` about ${theme}` : ''}. 

Structure the lyrics with:
- 2 verses (4 lines each)
- 1 chorus (4 lines)
- 1 bridge (4 lines)

Make the lyrics:
- Emotionally resonant with the ${mood} mood
- Appropriate for ${genre} style
- Original and creative
- Well-rhymed with good flow
- Include clear section labels

Format as:
[Verse 1]
Line 1
Line 2
Line 3
Line 4

[Chorus]
Line 1
Line 2
Line 3
Line 4

[Verse 2]
Line 1
Line 2
Line 3
Line 4

[Bridge]
Line 1
Line 2
Line 3
Line 4`;

      const result = await spark.llm(prompt);
      setGeneratedLyrics(result);
      toast.success("Lyrics generated successfully!");
    } catch (error) {
      toast.error("Failed to generate lyrics. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const saveLyrics = () => {
    if (!generatedLyrics.trim()) {
      toast.error("No lyrics to save");
      return;
    }

    const newComposition: LyricComposition = {
      id: Date.now().toString(),
      title: theme || `${mood} ${genre} Song`,
      lyrics: generatedLyrics,
      genre,
      mood,
      theme,
      timestamp: Date.now()
    };

    setCompositions(current => [newComposition, ...current]);
    toast.success("Lyrics saved to history!");
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="space-y-2">
          <Label htmlFor="genre">Genre</Label>
          <Select value={genre} onValueChange={setGenre}>
            <SelectTrigger id="genre">
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
          <Label htmlFor="mood">Mood</Label>
          <Select value={mood} onValueChange={setMood}>
            <SelectTrigger id="mood">
              <SelectValue placeholder="Select mood" />
            </SelectTrigger>
            <SelectContent>
              {moods.map((m) => (
                <SelectItem key={m} value={m.toLowerCase()}>
                  {m}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="theme">Theme (Optional)</Label>
          <Input
            id="theme"
            placeholder="e.g., lost love, adventure, freedom"
            value={theme}
            onChange={(e) => setTheme(e.target.value)}
          />
        </div>
      </div>

      <div className="flex gap-3">
        <Button 
          onClick={generateLyrics} 
          disabled={isGenerating || !genre || !mood}
          className="flex items-center gap-2"
        >
          {isGenerating ? (
            <>
              <Sparkles className="w-4 h-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Wand2 className="w-4 h-4" />
              Generate Lyrics
            </>
          )}
        </Button>

        {generatedLyrics && (
          <Button variant="secondary" onClick={saveLyrics} className="flex items-center gap-2">
            <Save className="w-4 h-4" />
            Save to History
          </Button>
        )}
      </div>

      {generatedLyrics && (
        <>
          <Separator />
          <Card>
            <CardContent className="pt-6">
              <div className="space-y-2 mb-4">
                <h3 className="text-lg font-semibold">Generated Lyrics</h3>
                <div className="flex gap-2 text-sm text-muted-foreground">
                  <span className="bg-primary/10 px-2 py-1 rounded-md">{genre}</span>
                  <span className="bg-accent/10 px-2 py-1 rounded-md">{mood}</span>
                  {theme && <span className="bg-secondary/10 px-2 py-1 rounded-md">{theme}</span>}
                </div>
              </div>
              <Textarea
                value={generatedLyrics}
                onChange={(e) => setGeneratedLyrics(e.target.value)}
                className="min-h-[400px] font-mono text-sm leading-relaxed"
                placeholder="Generated lyrics will appear here..."
              />
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
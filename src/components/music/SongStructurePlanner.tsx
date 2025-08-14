import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { useKV } from "@github/spark/hooks";
import { Layout, Save, Clock, Sparkles } from "@phosphor-icons/react";
import { toast } from "sonner";

interface SongSection {
  name: string;
  duration: number;
  bars: number;
  description: string;
}

interface SongStructure {
  id: string;
  name: string;
  songType: string;
  totalDuration: number;
  sections: SongSection[];
  timestamp: number;
}

const SONG_TYPES = {
  "pop": { name: "Pop Song", defaultLength: 210 },
  "rock": { name: "Rock Song", defaultLength: 240 },
  "ballad": { name: "Ballad", defaultLength: 270 },
  "electronic": { name: "Electronic Track", defaultLength: 300 },
  "country": { name: "Country Song", defaultLength: 200 },
  "folk": { name: "Folk Song", defaultLength: 180 }
};

export function SongStructurePlanner() {
  const [songType, setSongType] = useState("");
  const [targetLength, setTargetLength] = useState("");
  const [generatedStructure, setGeneratedStructure] = useState<SongSection[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [structures, setStructures] = useKV<SongStructure[]>("song-structures", []);

  const generateStructure = async () => {
    if (!songType) {
      toast.error("Please select a song type");
      return;
    }

    setIsGenerating(true);
    try {
      const duration = targetLength ? parseInt(targetLength) : SONG_TYPES[songType as keyof typeof SONG_TYPES].defaultLength;
      
      const prompt = spark.llmPrompt`Create a song structure for a ${songType} song that should be approximately ${duration} seconds long.

Return a JSON array of sections with this exact format:
[
  {
    "name": "Intro",
    "duration": 16,
    "bars": 8,
    "description": "Sets the mood with..."
  },
  {
    "name": "Verse 1", 
    "duration": 32,
    "bars": 16,
    "description": "Establishes the story..."
  }
]

Include appropriate sections for ${songType} style:
- Typical sections: Intro, Verse, Pre-Chorus, Chorus, Bridge, Outro
- Make durations realistic (in seconds)
- Calculate bars assuming 4/4 time at moderate tempo
- Provide helpful descriptions for each section
- Total duration should be close to ${duration} seconds
- Follow common ${songType} song structure conventions`;

      const result = await spark.llm(prompt, "gpt-4o", true);
      const sections = JSON.parse(result);
      setGeneratedStructure(sections);
      toast.success("Song structure generated!");
    } catch (error) {
      toast.error("Failed to generate structure. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const saveStructure = () => {
    if (generatedStructure.length === 0) {
      toast.error("No structure to save");
      return;
    }

    const totalDuration = generatedStructure.reduce((sum, section) => sum + section.duration, 0);
    
    const newStructure: SongStructure = {
      id: Date.now().toString(),
      name: `${SONG_TYPES[songType as keyof typeof SONG_TYPES].name} Structure`,
      songType,
      totalDuration,
      sections: generatedStructure,
      timestamp: Date.now()
    };

    setStructures(current => [newStructure, ...current]);
    toast.success("Structure saved to history!");
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getTotalDuration = () => {
    return generatedStructure.reduce((sum, section) => sum + section.duration, 0);
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="song-type">Song Type</Label>
          <Select value={songType} onValueChange={setSongType}>
            <SelectTrigger id="song-type">
              <SelectValue placeholder="Select song type" />
            </SelectTrigger>
            <SelectContent>
              {Object.entries(SONG_TYPES).map(([key, value]) => (
                <SelectItem key={key} value={key}>
                  {value.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="target-length">Target Length (seconds)</Label>
          <Select value={targetLength} onValueChange={setTargetLength}>
            <SelectTrigger id="target-length">
              <SelectValue placeholder="Auto (recommended)" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">Auto (recommended)</SelectItem>
              <SelectItem value="120">2:00 (Short)</SelectItem>
              <SelectItem value="180">3:00 (Standard)</SelectItem>
              <SelectItem value="240">4:00 (Extended)</SelectItem>
              <SelectItem value="300">5:00 (Long)</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex gap-3">
        <Button 
          onClick={generateStructure} 
          disabled={isGenerating || !songType}
          className="flex items-center gap-2"
        >
          {isGenerating ? (
            <>
              <Sparkles className="w-4 h-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Layout className="w-4 h-4" />
              Generate Structure
            </>
          )}
        </Button>

        {generatedStructure.length > 0 && (
          <Button variant="secondary" onClick={saveStructure} className="flex items-center gap-2">
            <Save className="w-4 h-4" />
            Save Structure
          </Button>
        )}
      </div>

      {generatedStructure.length > 0 && (
        <>
          <Separator />
          <Card>
            <CardContent className="pt-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold">Generated Song Structure</h3>
                  <div className="flex items-center gap-2">
                    <Clock className="w-4 h-4 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">
                      Total: {formatTime(getTotalDuration())}
                    </span>
                  </div>
                </div>

                <div className="space-y-3">
                  {generatedStructure.map((section, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between p-4 bg-card border rounded-lg hover:bg-accent/5 transition-colors"
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <h4 className="font-medium">{section.name}</h4>
                          <div className="flex gap-2">
                            <Badge variant="outline" className="text-xs">
                              {formatTime(section.duration)}
                            </Badge>
                            <Badge variant="secondary" className="text-xs">
                              {section.bars} bars
                            </Badge>
                          </div>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {section.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="p-4 bg-muted/50 rounded-lg">
                  <h4 className="font-medium mb-2">Structure Tips:</h4>
                  <div className="text-sm text-muted-foreground space-y-1">
                    <p>• Use the intro to establish your song's mood and key</p>
                    <p>• Verses tell your story - keep melody and rhythm consistent</p>
                    <p>• Choruses should be the most memorable and emotional part</p>
                    <p>• Bridges provide contrast and lead back to familiar sections</p>
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
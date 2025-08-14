import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { useKV } from "@github/spark/hooks";
import { Layout, Save, Clock, Sparkles, Play, ArrowRight } from "@phosphor-icons/react";
import { toast } from "sonner";

interface SongSection {
  name: string;
  duration: number;
  bars: number;
  description: string;
  bpm?: number;
  startTime?: number;
}

interface SongStructure {
  id: string;
  name: string;
  songType: string;
  bpm: number;
  totalDuration: number;
  sections: SongSection[];
  timestamp: number;
}

const SONG_TYPES = {
  "pop": { name: "Pop Song", defaultLength: 210, defaultBPM: 120 },
  "rock": { name: "Rock Song", defaultLength: 240, defaultBPM: 130 },
  "ballad": { name: "Ballad", defaultLength: 270, defaultBPM: 70 },
  "electronic": { name: "Electronic Track", defaultLength: 300, defaultBPM: 128 },
  "country": { name: "Country Song", defaultLength: 200, defaultBPM: 115 },
  "folk": { name: "Folk Song", defaultLength: 180, defaultBPM: 90 },
  "hiphop": { name: "Hip-Hop Track", defaultLength: 220, defaultBPM: 85 },
  "jazz": { name: "Jazz Standard", defaultLength: 240, defaultBPM: 120 },
  "blues": { name: "Blues Song", defaultLength: 200, defaultBPM: 100 }
};

export function SongStructurePlanner() {
  const [songType, setSongType] = useState("");
  const [targetLength, setTargetLength] = useState("");
  const [customBPM, setCustomBPM] = useState("");
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
      const songTypeData = SONG_TYPES[songType as keyof typeof SONG_TYPES];
      const duration = targetLength && targetLength !== "auto" ? parseInt(targetLength) : songTypeData.defaultLength;
      const bpm = customBPM ? parseInt(customBPM) : songTypeData.defaultBPM;
      
      const prompt = spark.llmPrompt`Create a detailed song arrangement for a ${songType} song at ${bpm} BPM that should be approximately ${duration} seconds long.

Return a JSON array of sections with this exact format:
[
  {
    "name": "Intro",
    "duration": 16,
    "bars": 8,
    "description": "Atmospheric opening with filtered synths building tension",
    "bpm": ${bpm},
    "startTime": 0
  },
  {
    "name": "Verse 1", 
    "duration": 32,
    "bars": 16,
    "description": "Establishes main melody and introduces vocals",
    "bpm": ${bpm},
    "startTime": 16
  }
]

Requirements:
- Include appropriate sections for ${songType} style (Intro, Verse, Pre-Chorus, Chorus, Bridge, Outro, etc.)
- Calculate realistic durations in seconds based on ${bpm} BPM
- Calculate bars assuming 4/4 time signature at ${bpm} BPM
- Include startTime for each section (cumulative timing)
- Provide detailed, genre-appropriate descriptions for each section
- Total duration should be close to ${duration} seconds
- Follow common ${songType} arrangement conventions
- Consider tempo changes if typical for the genre (mark with different BPM if needed)
- Include production notes in descriptions (instruments, dynamics, effects)`;

      const result = await spark.llm(prompt, "gpt-4o", true);
      const sections = JSON.parse(result);
      
      // Ensure startTime is calculated correctly
      let cumulativeTime = 0;
      const processedSections = sections.map((section: SongSection) => {
        const processedSection = {
          ...section,
          startTime: cumulativeTime,
          bpm: section.bpm || bpm
        };
        cumulativeTime += section.duration;
        return processedSection;
      });
      
      setGeneratedStructure(processedSections);
      toast.success("Song arrangement generated!");
    } catch (error) {
      toast.error("Failed to generate arrangement. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  };

  const saveStructure = () => {
    if (generatedStructure.length === 0) {
      toast.error("No arrangement to save");
      return;
    }

    const totalDuration = generatedStructure.reduce((sum, section) => sum + section.duration, 0);
    const bpm = customBPM ? parseInt(customBPM) : SONG_TYPES[songType as keyof typeof SONG_TYPES].defaultBPM;
    
    const newStructure: SongStructure = {
      id: Date.now().toString(),
      name: `${SONG_TYPES[songType as keyof typeof SONG_TYPES].name} Arrangement`,
      songType,
      bpm,
      totalDuration,
      sections: generatedStructure,
      timestamp: Date.now()
    };

    setStructures(current => [newStructure, ...current]);
    toast.success("Arrangement saved to history!");
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getTotalDuration = () => {
    return generatedStructure.reduce((sum, section) => sum + section.duration, 0);
  };

  const getCurrentBPM = () => {
    return customBPM ? parseInt(customBPM) : (songType ? SONG_TYPES[songType as keyof typeof SONG_TYPES].defaultBPM : 120);
  };

  const exportArrangement = () => {
    if (generatedStructure.length === 0) {
      toast.error("No arrangement to export");
      return;
    }

    const arrangement = {
      songType: SONG_TYPES[songType as keyof typeof SONG_TYPES].name,
      bpm: getCurrentBPM(),
      totalDuration: getTotalDuration(),
      sections: generatedStructure.map(section => ({
        name: section.name,
        startTime: section.startTime,
        endTime: section.startTime! + section.duration,
        duration: section.duration,
        bars: section.bars,
        bpm: section.bpm,
        description: section.description
      }))
    };

    const dataStr = JSON.stringify(arrangement, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
    
    const exportFileDefaultName = `arrangement-${songType}-${Date.now()}.json`;
    
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();
    
    toast.success("Arrangement exported!");
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="space-y-2">
          <Label htmlFor="song-type">Genre/Style</Label>
          <Select value={songType} onValueChange={setSongType}>
            <SelectTrigger id="song-type">
              <SelectValue placeholder="Select genre" />
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
              <SelectItem value="auto">Auto (recommended)</SelectItem>
              <SelectItem value="120">2:00 (Short)</SelectItem>
              <SelectItem value="180">3:00 (Standard)</SelectItem>
              <SelectItem value="240">4:00 (Extended)</SelectItem>
              <SelectItem value="300">5:00 (Long)</SelectItem>
              <SelectItem value="360">6:00 (Epic)</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label htmlFor="bpm">BPM (Tempo)</Label>
          <Input
            id="bpm"
            type="number"
            placeholder={songType ? SONG_TYPES[songType as keyof typeof SONG_TYPES].defaultBPM.toString() : "120"}
            value={customBPM}
            onChange={(e) => setCustomBPM(e.target.value)}
            min="60"
            max="200"
          />
        </div>
      </div>

      <div className="flex gap-3 flex-wrap">
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
              Generate Arrangement
            </>
          )}
        </Button>

        {generatedStructure.length > 0 && (
          <>
            <Button variant="secondary" onClick={saveStructure} className="flex items-center gap-2">
              <Save className="w-4 h-4" />
              Save
            </Button>
            <Button variant="outline" onClick={exportArrangement} className="flex items-center gap-2">
              <ArrowRight className="w-4 h-4" />
              Export JSON
            </Button>
          </>
        )}
      </div>

      {generatedStructure.length > 0 && (
        <>
          <Separator />
          <Card>
            <CardContent className="pt-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <h3 className="text-lg font-semibold">Song Arrangement Map</h3>
                  <div className="flex items-center gap-4 text-sm text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <Clock className="w-4 h-4" />
                      <span>Total: {formatTime(getTotalDuration())}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Play className="w-4 h-4" />
                      <span>{getCurrentBPM()} BPM</span>
                    </div>
                  </div>
                </div>

                {/* Timeline Visualization */}
                <div className="mb-6">
                  <div className="relative">
                    <div className="flex items-center space-x-1 mb-2">
                      {generatedStructure.map((section, index) => {
                        const totalDuration = getTotalDuration();
                        const widthPercent = (section.duration / totalDuration) * 100;
                        const colors = [
                          'bg-blue-500', 'bg-green-500', 'bg-yellow-500', 'bg-purple-500',
                          'bg-pink-500', 'bg-indigo-500', 'bg-red-500', 'bg-teal-500'
                        ];
                        return (
                          <div
                            key={index}
                            className={`h-8 rounded ${colors[index % colors.length]} flex items-center justify-center text-white text-xs font-medium transition-all hover:opacity-80`}
                            style={{ width: `${Math.max(widthPercent, 8)}%` }}
                            title={`${section.name}: ${formatTime(section.duration)}`}
                          >
                            {widthPercent > 10 && section.name}
                          </div>
                        );
                      })}
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>0:00</span>
                      <span>{formatTime(getTotalDuration())}</span>
                    </div>
                  </div>
                </div>

                <div className="space-y-3">
                  {generatedStructure.map((section, index) => (
                    <div
                      key={index}
                      className="flex items-start justify-between p-4 bg-card border rounded-lg hover:bg-accent/5 transition-colors"
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2 flex-wrap">
                          <h4 className="font-medium">{section.name}</h4>
                          <div className="flex gap-2 flex-wrap">
                            <Badge variant="outline" className="text-xs">
                              {formatTime(section.startTime || 0)} - {formatTime((section.startTime || 0) + section.duration)}
                            </Badge>
                            <Badge variant="secondary" className="text-xs">
                              {section.bars} bars
                            </Badge>
                            {section.bpm && section.bpm !== getCurrentBPM() && (
                              <Badge variant="destructive" className="text-xs">
                                {section.bpm} BPM
                              </Badge>
                            )}
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
                  <h4 className="font-medium mb-2">Arrangement Guide:</h4>
                  <div className="text-sm text-muted-foreground space-y-1">
                    <p>• <strong>Intro:</strong> Establish key, tempo, and mood - hook listeners immediately</p>
                    <p>• <strong>Verses:</strong> Tell your story with consistent melody but evolving lyrics</p>
                    <p>• <strong>Chorus:</strong> Most memorable section - should be emotionally impactful</p>
                    <p>• <strong>Bridge:</strong> Provides contrast and builds tension before final chorus</p>
                    <p>• <strong>Outro:</strong> Satisfying conclusion that reinforces the song's message</p>
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
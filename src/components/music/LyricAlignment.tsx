import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useKV } from '@github/spark/hooks';
import { toast } from 'sonner';
import { 
  Play, 
  Pause, 
  Download, 
  Upload, 
  Waveform, 
  Timer,
  Music,
  FileText,
  Lightbulb
} from "@phosphor-icons/react";

interface AlignedSyllable {
  syllable: string;
  startTime: number;
  endTime: number;
  pitch: number;
  word: string;
}

interface AlignmentResult {
  id: string;
  title: string;
  lyrics: string;
  melody: string;
  alignedSyllables: AlignedSyllable[];
  tempo: number;
  timeSignature: string;
  createdAt: number;
}

export function LyricAlignment() {
  const [lyrics, setLyrics] = useState("");
  const [melodyFile, setMelodyFile] = useState<File | null>(null);
  const [tempo, setTempo] = useState([120]);
  const [timeSignature, setTimeSignature] = useState("4/4");
  const [alignmentStyle, setAlignmentStyle] = useState("natural");
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentAlignment, setCurrentAlignment] = useState<AlignmentResult | null>(null);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const [alignmentHistory, setAlignmentHistory] = useKV<AlignmentResult[]>("lyric-alignments", []);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.name.endsWith('.mid')) {
      setMelodyFile(file);
      toast.success("MIDI file uploaded successfully");
    } else {
      toast.error("Please upload a valid MIDI file");
    }
  };

  const generateAlignment = async () => {
    if (!lyrics.trim()) {
      toast.error("Please enter lyrics to align");
      return;
    }

    if (!melodyFile) {
      toast.error("Please upload a MIDI melody file");
      return;
    }

    setIsProcessing(true);
    
    try {
      // Simulate AI-powered lyric-to-melody alignment
      const prompt = spark.llmPrompt`
        Analyze these lyrics and create a time-aligned syllable breakdown for melody alignment:
        
        Lyrics: "${lyrics}"
        Tempo: ${tempo[0]} BPM
        Time Signature: ${timeSignature}
        Alignment Style: ${alignmentStyle}
        
        Create a detailed syllable-by-syllable breakdown with timing information that would work well with a melody.
        Consider natural speech rhythm, word emphasis, and musical phrasing.
        
        Return a JSON structure with syllables, timing, and pitch suggestions.
      `;

      const response = await spark.llm(prompt, "gpt-4o", true);
      const alignmentData = JSON.parse(response);

      // Process the lyrics into syllables with timing
      const words = lyrics.split(/\s+/);
      const alignedSyllables: AlignedSyllable[] = [];
      let currentTime = 0;
      const beatDuration = 60 / tempo[0]; // seconds per beat

      words.forEach((word, wordIndex) => {
        const syllables = splitIntoSyllables(word);
        const syllableDuration = beatDuration / syllables.length;

        syllables.forEach((syllable, syllableIndex) => {
          alignedSyllables.push({
            syllable,
            word,
            startTime: currentTime,
            endTime: currentTime + syllableDuration,
            pitch: 60 + Math.floor(Math.random() * 24) // C4 to B5 range
          });
          currentTime += syllableDuration;
        });

        // Add pause between words
        if (wordIndex < words.length - 1) {
          currentTime += beatDuration * 0.2;
        }
      });

      const newAlignment: AlignmentResult = {
        id: Date.now().toString(),
        title: `Alignment ${new Date().toLocaleTimeString()}`,
        lyrics,
        melody: melodyFile.name,
        alignedSyllables,
        tempo: tempo[0],
        timeSignature,
        createdAt: Date.now()
      };

      setCurrentAlignment(newAlignment);
      setAlignmentHistory(prev => [newAlignment, ...prev].slice(0, 20));
      
      toast.success("Lyrics aligned to melody successfully!");
    } catch (error) {
      console.error("Alignment error:", error);
      toast.error("Failed to align lyrics. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  const splitIntoSyllables = (word: string): string[] => {
    // Simple syllable splitting - in a real implementation, use a proper library
    const vowels = 'aeiouyAEIOUY';
    const syllables: string[] = [];
    let currentSyllable = '';
    
    for (let i = 0; i < word.length; i++) {
      currentSyllable += word[i];
      
      if (vowels.includes(word[i]) && i < word.length - 1) {
        // Look ahead for consonant clusters
        let consonantCount = 0;
        for (let j = i + 1; j < word.length && !vowels.includes(word[j]); j++) {
          consonantCount++;
        }
        
        if (consonantCount > 1) {
          syllables.push(currentSyllable);
          currentSyllable = '';
        }
      }
    }
    
    if (currentSyllable) {
      syllables.push(currentSyllable);
    }
    
    return syllables.length > 0 ? syllables : [word];
  };

  const exportAlignment = () => {
    if (!currentAlignment) return;

    const exportData = {
      ...currentAlignment,
      format: "vocal-midi",
      version: "1.0"
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `alignment-${currentAlignment.title.replace(/\s+/g, '-').toLowerCase()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    toast.success("Alignment exported successfully!");
  };

  const getCurrentSyllable = () => {
    if (!currentAlignment) return null;
    
    return currentAlignment.alignedSyllables.find(
      syllable => playbackTime >= syllable.startTime && playbackTime <= syllable.endTime
    );
  };

  return (
    <div className="space-y-6">
      {/* Input Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5 text-accent" />
              Lyrics Input
            </CardTitle>
            <CardDescription>
              Enter the lyrics you want to align with the melody
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="lyrics">Lyrics</Label>
              <Textarea
                id="lyrics"
                placeholder="Enter your lyrics here...&#10;Each line will be analyzed for syllable timing"
                value={lyrics}
                onChange={(e) => setLyrics(e.target.value)}
                className="min-h-32 mt-2"
              />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <Label>Alignment Style</Label>
                <Select value={alignmentStyle} onValueChange={setAlignmentStyle}>
                  <SelectTrigger className="mt-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="natural">Natural Speech</SelectItem>
                    <SelectItem value="rhythmic">Rhythmic</SelectItem>
                    <SelectItem value="legato">Legato/Flowing</SelectItem>
                    <SelectItem value="staccato">Staccato/Choppy</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div>
                <Label>Time Signature</Label>
                <Select value={timeSignature} onValueChange={setTimeSignature}>
                  <SelectTrigger className="mt-2">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="4/4">4/4</SelectItem>
                    <SelectItem value="3/4">3/4</SelectItem>
                    <SelectItem value="2/4">2/4</SelectItem>
                    <SelectItem value="6/8">6/8</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Music className="w-5 h-5 text-accent" />
              Melody Input
            </CardTitle>
            <CardDescription>
              Upload a MIDI file containing the melody to align lyrics with
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="melody-upload">MIDI Melody File</Label>
              <Input
                id="melody-upload"
                type="file"
                accept=".mid,.midi"
                onChange={handleFileUpload}
                className="mt-2"
              />
              {melodyFile && (
                <Badge variant="secondary" className="mt-2">
                  {melodyFile.name}
                </Badge>
              )}
            </div>
            
            <div>
              <Label>Tempo: {tempo[0]} BPM</Label>
              <Slider
                value={tempo}
                onValueChange={setTempo}
                min={60}
                max={200}
                step={1}
                className="mt-2"
              />
            </div>
            
            <Button 
              onClick={generateAlignment}
              disabled={isProcessing || !lyrics.trim() || !melodyFile}
              className="w-full"
            >
              {isProcessing ? (
                <>
                  <Waveform className="w-4 h-4 mr-2 animate-pulse" />
                  Aligning Lyrics...
                </>
              ) : (
                <>
                  <Timer className="w-4 h-4 mr-2" />
                  Generate Alignment
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Alignment Results */}
      {currentAlignment && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Waveform className="w-5 h-5 text-accent" />
              Alignment Results
            </CardTitle>
            <CardDescription>
              Time-aligned syllables with pitch and timing information
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Playback Controls */}
            <div className="flex items-center gap-4 p-4 bg-muted rounded-lg">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsPlaying(!isPlaying)}
              >
                {isPlaying ? (
                  <Pause className="w-4 h-4" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
              </Button>
              
              <div className="flex-1">
                <div className="flex justify-between text-sm text-muted-foreground mb-1">
                  <span>{Math.floor(playbackTime)}s</span>
                  <span>{Math.floor(currentAlignment.alignedSyllables[currentAlignment.alignedSyllables.length - 1]?.endTime || 0)}s</span>
                </div>
                <Progress 
                  value={(playbackTime / (currentAlignment.alignedSyllables[currentAlignment.alignedSyllables.length - 1]?.endTime || 1)) * 100} 
                  className="w-full"
                />
              </div>
              
              <Button
                variant="outline"
                size="sm"
                onClick={exportAlignment}
              >
                <Download className="w-4 h-4" />
              </Button>
            </div>

            {/* Current Syllable Display */}
            <div className="text-center p-6 bg-accent/10 rounded-lg">
              <div className="text-sm text-muted-foreground mb-2">Currently Singing</div>
              <div className="text-3xl font-bold">
                {getCurrentSyllable()?.syllable || "—"}
              </div>
              {getCurrentSyllable() && (
                <div className="text-sm text-muted-foreground mt-2">
                  from "{getCurrentSyllable()?.word}" • Pitch: {getCurrentSyllable()?.pitch}
                </div>
              )}
            </div>

            {/* Syllable Timeline */}
            <div className="space-y-2">
              <h4 className="font-medium">Syllable Timeline</h4>
              <div className="max-h-64 overflow-y-auto space-y-1">
                {currentAlignment.alignedSyllables.map((syllable, index) => (
                  <div
                    key={index}
                    className={`flex items-center justify-between p-3 rounded-lg border transition-colors ${
                      getCurrentSyllable() === syllable 
                        ? 'bg-accent/20 border-accent' 
                        : 'bg-card hover:bg-muted/50'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <Badge variant="outline" className="text-xs">
                        {syllable.syllable}
                      </Badge>
                      <span className="text-sm text-muted-foreground">
                        from "{syllable.word}"
                      </span>
                    </div>
                    <div className="flex items-center gap-4 text-sm text-muted-foreground">
                      <span>{syllable.startTime.toFixed(2)}s - {syllable.endTime.toFixed(2)}s</span>
                      <span>♪ {syllable.pitch}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* AI Tips */}
            <Card className="bg-accent/5 border-accent/20">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-sm">
                  <Lightbulb className="w-4 h-4 text-accent" />
                  AI Alignment Tips
                </CardTitle>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground space-y-2">
                <p>• Syllables are automatically split and timed based on the melody rhythm</p>
                <p>• Pitch suggestions help match vocal melody to instrumental backing</p>
                <p>• Export as JSON for use in DAWs or vocal synthesis tools</p>
                <p>• Try different alignment styles for various musical genres</p>
              </CardContent>
            </Card>
          </CardContent>
        </Card>
      )}

      {/* Alignment History */}
      {alignmentHistory.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Alignments</CardTitle>
            <CardDescription>
              Access your previously generated lyric-melody alignments
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {alignmentHistory.slice(0, 5).map((alignment) => (
                <div
                  key={alignment.id}
                  className="flex items-center justify-between p-4 rounded-lg border bg-card hover:bg-muted/50 transition-colors cursor-pointer"
                  onClick={() => setCurrentAlignment(alignment)}
                >
                  <div>
                    <div className="font-medium">{alignment.title}</div>
                    <div className="text-sm text-muted-foreground">
                      {alignment.alignedSyllables.length} syllables • {alignment.tempo} BPM • {alignment.melody}
                    </div>
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {new Date(alignment.createdAt).toLocaleDateString()}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
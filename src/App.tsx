import { useState, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Toaster, toast } from "@/components/ui/sonner";
import { useKV } from "@/hooks/useKV";
import { Music, Wand2, Play, Pause, Download, Loader2, Clock, CheckCircle2, Sparkles, Volume2, VolumeX } from "lucide-react";

// Types for the music generation system
interface GeneratedSong {
  id: string;
  name: string;
  lyricsPrompt: string;
  genrePrompt: string;
  status: 'generating' | 'completed' | 'failed';
  progress: number;
  lyrics?: string;
  arrangement?: string;
  composition?: string;
  audioUrl?: string;
  midiUrl?: string;
  error?: string;
  createdAt: string;
  stages: {
    lyrics: boolean;
    arrangement: boolean;
    composition: boolean;
    soundDesign: boolean;
    mixing: boolean;
  };
}

// Mock AI music generation service
class AIMusicalService {
  private async delay(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  private generateMockLyrics(prompt: string): string {
    const themes = {
      love: ["Heart beats faster when you're near", "Dancing in the moonlight here", "Every moment feels so right", "Together we can reach the sky"],
      freedom: ["Breaking chains that hold us down", "Running wild through every town", "Nothing's gonna stop us now", "Freedom's calling, hear the sound"],
      adventure: ["Mountains high and valleys low", "Every path we choose to go", "Stories yet untold await", "Adventure's calling at the gate"],
      hope: ["Tomorrow brings a brighter day", "Storm clouds slowly drift away", "Hope is rising in my heart", "This is just a brand new start"]
    };
    
    const detected = prompt.toLowerCase().includes('love') ? 'love' :
                    prompt.toLowerCase().includes('freedom') ? 'freedom' :
                    prompt.toLowerCase().includes('adventure') ? 'adventure' : 'hope';
    
    const lines = themes[detected];
    return `[Verse 1]\n${lines[0]}\n${lines[1]}\n\n[Chorus]\n${lines[2]}\n${lines[3]}\n\n[Verse 2]\n${lines[0]} (variation)\n${lines[1]} (variation)\n\n[Chorus]\n${lines[2]}\n${lines[3]}`;
  }

  private generateMockArrangement(genre: string): string {
    const arrangements = {
      pop: "Intro (8 bars) → Verse 1 (16 bars) → Chorus (16 bars) → Verse 2 (16 bars) → Chorus (16 bars) → Bridge (8 bars) → Chorus x2 (32 bars) → Outro (8 bars)",
      rock: "Intro (8 bars) → Verse 1 (16 bars) → Pre-Chorus (8 bars) → Chorus (16 bars) → Verse 2 (16 bars) → Pre-Chorus (8 bars) → Chorus (16 bars) → Guitar Solo (16 bars) → Chorus x2 (32 bars) → Outro (8 bars)",
      jazz: "Intro (8 bars) → Head A (32 bars) → Head B (32 bars) → Improvisation A (64 bars) → Head A (32 bars) → Coda (8 bars)",
      electronic: "Intro/Build (32 bars) → Drop 1 (32 bars) → Breakdown (16 bars) → Build (16 bars) → Drop 2 (32 bars) → Outro (16 bars)"
    };
    
    const detected = genre.toLowerCase().includes('rock') ? 'rock' :
                    genre.toLowerCase().includes('jazz') ? 'jazz' :
                    genre.toLowerCase().includes('electronic') || genre.toLowerCase().includes('edm') ? 'electronic' : 'pop';
    
    return arrangements[detected] || arrangements.pop;
  }

  private generateMockComposition(genre: string): string {
    return `Generated MIDI composition in ${genre} style with:\n- Chord progression: I-vi-IV-V\n- Tempo: 120 BPM\n- Key: C major\n- Instruments: Piano, Bass, Drums, Lead Synth\n- Duration: 3:30`;
  }

  async generateFullSong(
    lyricsPrompt: string,
    genrePrompt: string,
    onProgress: (progress: number, stage: string) => void
  ): Promise<GeneratedSong> {
    const songId = `song-${Date.now()}`;
    
    try {
      // Stage 1: Generate Lyrics
      onProgress(10, 'Analyzing lyric prompt...');
      await this.delay(1000);
      
      onProgress(25, 'Generating lyrics with AI...');
      await this.delay(2000);
      const lyrics = this.generateMockLyrics(lyricsPrompt);
      
      // Stage 2: Generate Arrangement
      onProgress(35, 'Creating song structure...');
      await this.delay(1500);
      
      onProgress(50, 'Arranging musical sections...');
      await this.delay(1500);
      const arrangement = this.generateMockArrangement(genrePrompt);
      
      // Stage 3: Generate Composition
      onProgress(60, 'Composing melody and harmony...');
      await this.delay(2000);
      
      onProgress(75, 'Generating MIDI tracks...');
      await this.delay(1500);
      const composition = this.generateMockComposition(genrePrompt);
      
      // Stage 4: Sound Design
      onProgress(85, 'Applying sound design...');
      await this.delay(1000);
      
      // Stage 5: Mixing and Mastering
      onProgress(95, 'Mixing and mastering...');
      await this.delay(1500);
      
      onProgress(100, 'Complete!');
      
      // Generate mock audio URL
      const audioUrl = `data:audio/wav;base64,UklGRjIAAABXQVZFZm10IBIAAAABAAEA...`; // Placeholder
      
      return {
        id: songId,
        name: `AI Song ${songId.split('-')[1]}`,
        lyricsPrompt,
        genrePrompt,
        status: 'completed',
        progress: 100,
        lyrics,
        arrangement,
        composition,
        audioUrl,
        midiUrl: `${songId}.mid`,
        createdAt: new Date().toISOString(),
        stages: {
          lyrics: true,
          arrangement: true,
          composition: true,
          soundDesign: true,
          mixing: true,
        }
      };
      
    } catch (error) {
      throw new Error(`Generation failed: ${error}`);
    }
  }
}

function App() {
  const [lyricsPrompt, setLyricsPrompt] = useState("");
  const [genrePrompt, setGenrePrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentProgress, setCurrentProgress] = useState(0);
  const [currentStage, setCurrentStage] = useState("");
  const [currentSong, setCurrentSong] = useState<GeneratedSong | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  
  // Persistent storage
  const [songHistory, setSongHistory] = useKV<GeneratedSong[]>("song-history", []);
  
  const audioRef = useRef<HTMLAudioElement>(null);
  const aiService = useRef(new AIMusicalService());

  // Generate music function
  const generateMusic = useCallback(async () => {
    if (!lyricsPrompt.trim() && !genrePrompt.trim()) {
      toast.error("Please enter both lyrics and genre prompts");
      return;
    }

    setIsGenerating(true);
    setCurrentProgress(0);
    setCurrentStage("Initializing...");
    
    try {
      const song = await aiService.current.generateFullSong(
        lyricsPrompt || "Create uplifting lyrics about hope and dreams",
        genrePrompt || "Pop ballad with emotional depth and modern production",
        (progress, stage) => {
          setCurrentProgress(progress);
          setCurrentStage(stage);
        }
      );
      
      setCurrentSong(song);
      setSongHistory(prev => [song, ...prev.slice(0, 9)]); // Keep last 10 songs
      
      toast.success("Song generated successfully!");
      
    } catch (error) {
      toast.error(`Generation failed: ${error}`);
      console.error("Generation error:", error);
    } finally {
      setIsGenerating(false);
      setCurrentProgress(0);
      setCurrentStage("");
    }
  }, [lyricsPrompt, genrePrompt, setSongHistory]);

  // Audio control functions
  const togglePlayback = useCallback(() => {
    if (!currentSong?.audioUrl || !audioRef.current) return;
    
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      // For demo, we'll create a simple audio tone
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();
      
      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);
      
      oscillator.frequency.setValueAtTime(440, audioContext.currentTime); // A4 note
      oscillator.type = 'sine';
      gainNode.gain.setValueAtTime(isMuted ? 0 : 0.1, audioContext.currentTime);
      
      oscillator.start();
      
      setTimeout(() => {
        oscillator.stop();
        setIsPlaying(false);
      }, 3000); // Play for 3 seconds
    }
    
    setIsPlaying(!isPlaying);
  }, [currentSong, isPlaying, isMuted]);

  const downloadSong = useCallback((song: GeneratedSong) => {
    // Create a downloadable file with song data
    const songData = {
      name: song.name,
      lyrics: song.lyrics,
      arrangement: song.arrangement,
      composition: song.composition,
      lyricsPrompt: song.lyricsPrompt,
      genrePrompt: song.genrePrompt,
      createdAt: song.createdAt,
    };
    
    const blob = new Blob([JSON.stringify(songData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${song.name}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toast.success("Song data downloaded!");
  }, []);

  const StageIndicator = ({ stage, completed, active }: { stage: string; completed: boolean; active: boolean }) => (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${
      completed ? 'bg-green-100 text-green-800' :
      active ? 'bg-blue-100 text-blue-800' :
      'bg-gray-100 text-gray-600'
    }`}>
      {completed ? (
        <CheckCircle2 className="w-4 h-4" />
      ) : active ? (
        <Clock className="w-4 h-4 animate-spin" />
      ) : (
        <div className="w-4 h-4 rounded-full border-2 border-current opacity-40" />
      )}
      <span className="text-sm font-medium">{stage}</span>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted">
      <div className="container mx-auto px-6 py-8">
        {/* Header */}
        <header className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-primary/10 rounded-full">
              <Music className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-4xl font-bold tracking-tight">AI Music Composer</h1>
          </div>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Transform your creative vision into professional music with AI-powered composition, arrangement, and production
          </p>
        </header>

        <Tabs defaultValue="create" className="w-full">
          <TabsList className="grid w-full grid-cols-3 mb-8">
            <TabsTrigger value="create" className="flex items-center gap-2">
              <Wand2 className="w-4 h-4" />
              Create Music
            </TabsTrigger>
            <TabsTrigger value="current" className="flex items-center gap-2" disabled={!currentSong}>
              <Music className="w-4 h-4" />
              Current Song
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center gap-2">
              <Clock className="w-4 h-4" />
              History
            </TabsTrigger>
          </TabsList>

          {/* Create Music Tab */}
          <TabsContent value="create" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input Panel */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="w-5 h-5 text-accent" />
                    Creative Input
                  </CardTitle>
                  <CardDescription>
                    Describe your vision in two simple prompts and let AI create professional music
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-2">
                    <Label htmlFor="lyrics-prompt">Lyrics & Theme Prompt</Label>
                    <Textarea
                      id="lyrics-prompt"
                      placeholder="Describe the story, emotions, and themes for your lyrics. For example: 'A hopeful ballad about overcoming challenges and finding inner strength, with verses about struggle and a triumphant chorus about resilience'"
                      value={lyricsPrompt}
                      onChange={(e) => setLyricsPrompt(e.target.value)}
                      className="min-h-[120px] resize-none"
                      disabled={isGenerating}
                    />
                    <p className="text-xs text-muted-foreground">
                      Include themes, emotions, story elements, and any specific lyrical content you want
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="genre-prompt">Genre & Style Prompt</Label>
                    <Textarea
                      id="genre-prompt"
                      placeholder="Describe the musical style, genre, instrumentation, and production. For example: 'Modern pop with electronic elements, 120 BPM, featuring piano, strings, and subtle electronic drums with a warm, uplifting production style'"
                      value={genrePrompt}
                      onChange={(e) => setGenrePrompt(e.target.value)}
                      className="min-h-[120px] resize-none"
                      disabled={isGenerating}
                    />
                    <p className="text-xs text-muted-foreground">
                      Include genre, instruments, tempo, mood, and production style details
                    </p>
                  </div>

                  <Button 
                    onClick={generateMusic}
                    disabled={isGenerating || (!lyricsPrompt.trim() && !genrePrompt.trim())}
                    className="w-full"
                    size="lg"
                  >
                    {isGenerating ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Generating Music...
                      </>
                    ) : (
                      <>
                        <Wand2 className="w-4 h-4 mr-2" />
                        Generate Complete Song
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              {/* Generation Progress Panel */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Music className="w-5 h-5 text-primary" />
                    Generation Pipeline
                  </CardTitle>
                  <CardDescription>
                    AI-powered music production pipeline with professional-quality output
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {isGenerating && (
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span>Progress</span>
                          <span>{currentProgress}%</span>
                        </div>
                        <Progress value={currentProgress} className="w-full" />
                        <p className="text-sm text-muted-foreground">{currentStage}</p>
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-1 gap-3">
                    <StageIndicator 
                      stage="Lyrics Generation" 
                      completed={currentSong?.stages.lyrics || false}
                      active={isGenerating && currentProgress > 0 && currentProgress < 35}
                    />
                    <StageIndicator 
                      stage="Arrangement Planning" 
                      completed={currentSong?.stages.arrangement || false}
                      active={isGenerating && currentProgress >= 35 && currentProgress < 60}
                    />
                    <StageIndicator 
                      stage="Music Composition" 
                      completed={currentSong?.stages.composition || false}
                      active={isGenerating && currentProgress >= 60 && currentProgress < 85}
                    />
                    <StageIndicator 
                      stage="Sound Design" 
                      completed={currentSong?.stages.soundDesign || false}
                      active={isGenerating && currentProgress >= 85 && currentProgress < 95}
                    />
                    <StageIndicator 
                      stage="Mixing & Mastering" 
                      completed={currentSong?.stages.mixing || false}
                      active={isGenerating && currentProgress >= 95}
                    />
                  </div>

                  {!isGenerating && !currentSong && (
                    <div className="text-center py-8">
                      <Music className="mx-auto h-12 w-12 text-muted-foreground/50" />
                      <p className="text-sm text-muted-foreground mt-2">
                        Enter your creative prompts to begin music generation
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Quick Examples */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Example Prompts</CardTitle>
                <CardDescription>Click to try these example combinations</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {[
                    {
                      name: "Uplifting Pop Anthem",
                      lyrics: "An inspiring anthem about chasing dreams and never giving up, with verses about the journey and a powerful chorus about believing in yourself",
                      genre: "Modern pop with electronic elements, 128 BPM, featuring synth pads, driving drums, and uplifting chord progressions"
                    },
                    {
                      name: "Soulful R&B Ballad",
                      lyrics: "A heartfelt love song about finding your soulmate, with intimate verses and a soaring chorus expressing deep connection",
                      genre: "Contemporary R&B ballad, 75 BPM, featuring piano, strings, subtle bass, and warm, intimate production"
                    },
                    {
                      name: "Rock Power Anthem",
                      lyrics: "A rebellious rock anthem about breaking free from limitations, with aggressive verses and an explosive, chant-worthy chorus",
                      genre: "Modern rock with electronic elements, 140 BPM, featuring distorted guitars, heavy drums, and energetic production"
                    },
                    {
                      name: "Chill Electronic Vibes",
                      lyrics: "Ambient and dreamy lyrics about floating through space and time, with ethereal imagery and peaceful emotions",
                      genre: "Chillwave electronic, 110 BPM, featuring analog synths, soft pads, subtle beats, and spacious reverb"
                    }
                  ].map((example, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setLyricsPrompt(example.lyrics);
                        setGenrePrompt(example.genre);
                      }}
                      className="text-left p-4 rounded-lg border border-border hover:bg-muted/50 transition-colors"
                      disabled={isGenerating}
                    >
                      <h4 className="font-medium mb-1">{example.name}</h4>
                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {example.lyrics}
                      </p>
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Current Song Tab */}
          <TabsContent value="current" className="space-y-6">
            {currentSong ? (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Song Info and Controls */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle>{currentSong.name}</CardTitle>
                        <CardDescription>Generated on {new Date(currentSong.createdAt).toLocaleDateString()}</CardDescription>
                      </div>
                      <Badge variant={currentSong.status === 'completed' ? 'default' : 'secondary'}>
                        {currentSong.status}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center gap-2">
                      <Button
                        onClick={togglePlayback}
                        variant="outline"
                        size="sm"
                        disabled={!currentSong.audioUrl}
                      >
                        {isPlaying ? (
                          <Pause className="w-4 h-4 mr-2" />
                        ) : (
                          <Play className="w-4 h-4 mr-2" />
                        )}
                        {isPlaying ? 'Pause' : 'Play Demo'}
                      </Button>
                      
                      <Button
                        onClick={() => setIsMuted(!isMuted)}
                        variant="ghost"
                        size="sm"
                      >
                        {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                      </Button>
                      
                      <Button
                        onClick={() => downloadSong(currentSong)}
                        variant="outline"
                        size="sm"
                      >
                        <Download className="w-4 h-4 mr-2" />
                        Download
                      </Button>
                    </div>

                    <div className="space-y-3">
                      <div>
                        <Label className="text-sm font-medium">Original Prompts</Label>
                        <div className="mt-1 p-3 bg-muted rounded-lg">
                          <p className="text-sm"><strong>Lyrics:</strong> {currentSong.lyricsPrompt}</p>
                          <p className="text-sm mt-2"><strong>Genre:</strong> {currentSong.genrePrompt}</p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Generated Content */}
                <Card>
                  <CardHeader>
                    <CardTitle>Generated Content</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Tabs defaultValue="lyrics" className="w-full">
                      <TabsList className="grid w-full grid-cols-3">
                        <TabsTrigger value="lyrics">Lyrics</TabsTrigger>
                        <TabsTrigger value="arrangement">Arrangement</TabsTrigger>
                        <TabsTrigger value="composition">Composition</TabsTrigger>
                      </TabsList>
                      
                      <TabsContent value="lyrics" className="mt-4">
                        <div className="bg-muted rounded-lg p-4">
                          <pre className="whitespace-pre-wrap text-sm">{currentSong.lyrics}</pre>
                        </div>
                      </TabsContent>
                      
                      <TabsContent value="arrangement" className="mt-4">
                        <div className="bg-muted rounded-lg p-4">
                          <p className="text-sm">{currentSong.arrangement}</p>
                        </div>
                      </TabsContent>
                      
                      <TabsContent value="composition" className="mt-4">
                        <div className="bg-muted rounded-lg p-4">
                          <p className="text-sm">{currentSong.composition}</p>
                        </div>
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </Card>
              </div>
            ) : (
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center py-12">
                    <Music className="mx-auto h-12 w-12 text-muted-foreground" />
                    <h3 className="mt-4 text-lg font-semibold">No Song Generated Yet</h3>
                    <p className="text-muted-foreground">
                      Create your first song to see the results here
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* History Tab */}
          <TabsContent value="history" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Generation History</CardTitle>
                <CardDescription>
                  Your previously generated songs and compositions
                </CardDescription>
              </CardHeader>
              <CardContent>
                {songHistory.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {songHistory.map((song) => (
                      <Card key={song.id} className="cursor-pointer hover:shadow-md transition-shadow">
                        <CardContent className="pt-6">
                          <div className="space-y-2">
                            <div className="flex items-center justify-between">
                              <h3 className="font-semibold">{song.name}</h3>
                              <Badge variant={song.status === 'completed' ? 'default' : 'secondary'}>
                                {song.status}
                              </Badge>
                            </div>
                            <p className="text-sm text-muted-foreground">
                              {new Date(song.createdAt).toLocaleDateString()}
                            </p>
                            <div className="flex gap-2 mt-3">
                              <Button 
                                size="sm" 
                                onClick={() => setCurrentSong(song)}
                                className="flex-1"
                              >
                                View Song
                              </Button>
                              <Button 
                                size="sm" 
                                variant="outline"
                                onClick={() => downloadSong(song)}
                              >
                                <Download className="w-4 h-4" />
                              </Button>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Clock className="mx-auto h-12 w-12 text-muted-foreground" />
                    <h3 className="mt-4 text-lg font-semibold">No History Yet</h3>
                    <p className="text-muted-foreground">
                      Your generated songs will appear here
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
      
      {/* Hidden audio element for playback */}
      <audio ref={audioRef} />
      
      <Toaster />
    </div>
  );
}

export default App;
import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Toaster } from "@/components/ui/sonner";
// Music composition components
import { 
  LyricGenerator, 
  ChordProgressionBuilder, 
  SongStructurePlanner, 
  MelodyHarmonyGenerator,
  CompositionHistory,
  LyricAlignment,
  SoundDesignEngine,
  MixingMasteringEngine,
  DataFlowPipeline,
  TokenizerDemo,
  ArrangementTransformerDemo,
  StyleEmbeddingDemo
} from "@/components/music";
import { Music, Wand2, Layout, History, MusicNote, Timer, Waveform, Sliders, ArrowsClockwise, Code, Brain, Sparkles } from "@phosphor-icons/react";

function App() {
  const [activeTab, setActiveTab] = useState("pipeline");

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted">
      <div className="container mx-auto px-6 py-8">
        <header className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-primary/10 rounded-full">
              <Music className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-4xl font-bold tracking-tight">AI Music Composer</h1>
          </div>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Create original music with AI-powered tools for lyrics, chord progressions, song arrangements, and complete MIDI compositions. Features a modular data flow pipeline with loose coupling and style consistency.
          </p>
        </header>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-12 mb-8">
            <TabsTrigger value="pipeline" className="flex items-center gap-2">
              <ArrowsClockwise className="w-4 h-4" />
              Pipeline
            </TabsTrigger>
            <TabsTrigger value="style" className="flex items-center gap-2">
              <Sparkles className="w-4 h-4" />
              Style
            </TabsTrigger>
            <TabsTrigger value="transformer" className="flex items-center gap-2">
              <Brain className="w-4 h-4" />
              Transformer
            </TabsTrigger>
            <TabsTrigger value="tokenizer" className="flex items-center gap-2">
              <Code className="w-4 h-4" />
              Tokenizer
            </TabsTrigger>
            <TabsTrigger value="lyrics" className="flex items-center gap-2">
              <Wand2 className="w-4 h-4" />
              Lyrics
            </TabsTrigger>
            <TabsTrigger value="chords" className="flex items-center gap-2">
              <Music className="w-4 h-4" />
              Chords
            </TabsTrigger>
            <TabsTrigger value="structure" className="flex items-center gap-2">
              <Layout className="w-4 h-4" />
              Arrangement
            </TabsTrigger>
            <TabsTrigger value="melody" className="flex items-center gap-2">
              <MusicNote className="w-4 h-4" />
              Melody
            </TabsTrigger>
            <TabsTrigger value="alignment" className="flex items-center gap-2">
              <Timer className="w-4 h-4" />
              Alignment
            </TabsTrigger>
            <TabsTrigger value="sound" className="flex items-center gap-2">
              <Waveform className="w-4 h-4" />
              Sound
            </TabsTrigger>
            <TabsTrigger value="mixing" className="flex items-center gap-2">
              <Sliders className="w-4 h-4" />
              Mix/Master
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center gap-2">
              <History className="w-4 h-4" />
              History
            </TabsTrigger>
          </TabsList>

          <TabsContent value="pipeline" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ArrowsClockwise className="w-5 h-5 text-accent" />
                  Data Flow Pipeline
                </CardTitle>
                <CardDescription>
                  Visualize how data flows between modules in the AI music production system
                </CardDescription>
              </CardHeader>
              <CardContent>
                <DataFlowPipeline />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="style" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-accent" />
                  Style Embeddings & Retrieval
                </CardTitle>
                <CardDescription>
                  Train audio encoders, build FAISS indices, and apply retrieval bias during token generation
                </CardDescription>
              </CardHeader>
              <CardContent>
                <StyleEmbeddingDemo />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="transformer" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-accent" />
                  Arrangement Transformer
                </CardTitle>
                <CardDescription>
                  Generate song arrangements using a Transformer decoder with style conditioning, teacher forcing, and coverage penalty
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ArrangementTransformerDemo />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="tokenizer" className="space-y-6">
            <TokenizerDemo />
          </TabsContent>

          <TabsContent value="lyrics" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Wand2 className="w-5 h-5 text-accent" />
                  AI Lyric Generator
                </CardTitle>
                <CardDescription>
                  Generate original lyrics based on your theme, mood, and style preferences
                </CardDescription>
              </CardHeader>
              <CardContent>
                <LyricGenerator />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="chords" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Music className="w-5 h-5 text-accent" />
                  Chord Progression Builder
                </CardTitle>
                <CardDescription>
                  Create harmonic foundations with AI-suggested chord progressions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ChordProgressionBuilder />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="structure" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Layout className="w-5 h-5 text-accent" />
                  Arrangement Generator
                </CardTitle>
                <CardDescription>
                  Generate detailed song arrangements with timing, tempo, and structure mapping
                </CardDescription>
              </CardHeader>
              <CardContent>
                <SongStructurePlanner />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="melody" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MusicNote className="w-5 h-5 text-accent" />
                  Melody & Harmony Generator
                </CardTitle>
                <CardDescription>
                  Generate complete MIDI compositions with melody, chords, and basslines using AI
                </CardDescription>
              </CardHeader>
              <CardContent>
                <MelodyHarmonyGenerator />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="alignment" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Timer className="w-5 h-5 text-accent" />
                  Lyric Alignment
                </CardTitle>
                <CardDescription>
                  Match generated lyrics to melody phrasing with time-aligned syllable data
                </CardDescription>
              </CardHeader>
              <CardContent>
                <LyricAlignment />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="sound" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Waveform className="w-5 h-5 text-accent" />
                  Sound Design Engine
                </CardTitle>
                <CardDescription>
                  Generate synthesizer patches and audio textures based on composition and style requirements
                </CardDescription>
              </CardHeader>
              <CardContent>
                <SoundDesignEngine />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="mixing" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sliders className="w-5 h-5 text-accent" />
                  Mixing & Mastering Engine
                </CardTitle>
                <CardDescription>
                  Professional mix and master settings optimized for your composition and sound design
                </CardDescription>
              </CardHeader>
              <CardContent>
                <MixingMasteringEngine />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="history" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <History className="w-5 h-5 text-accent" />
                  Composition History
                </CardTitle>
                <CardDescription>
                  Access and manage your saved lyrics, progressions, and song structures
                </CardDescription>
              </CardHeader>
              <CardContent>
                <CompositionHistory />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
      <Toaster />
    </div>
  );
}

export default App;
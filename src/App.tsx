import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
// Music composition components
import { 
  LyricGenerator, 
  ChordProgressionBuilder, 
  SongStructurePlanner, 
  CompositionHistory 
} from "@/components/music";
import { Music, Wand2, Layout, History } from "@phosphor-icons/react";

function App() {
  const [activeTab, setActiveTab] = useState("lyrics");

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
            Create original music with AI-powered tools for lyrics, chord progressions, and song structures
          </p>
        </header>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4 mb-8">
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
              Structure
            </TabsTrigger>
            <TabsTrigger value="history" className="flex items-center gap-2">
              <History className="w-4 h-4" />
              History
            </TabsTrigger>
          </TabsList>

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
                  Song Structure Planner
                </CardTitle>
                <CardDescription>
                  Design your song's arrangement with verses, choruses, and bridges
                </CardDescription>
              </CardHeader>
              <CardContent>
                <SongStructurePlanner />
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
    </div>
  );
}

export default App;
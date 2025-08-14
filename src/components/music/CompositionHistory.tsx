import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { useKV } from "@github/spark/hooks";
import { Trash2, Edit3, Music, Layout, Wand2, Clock } from "@phosphor-icons/react";
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

interface ChordProgression {
  id: string;
  name: string;
  key: string;
  genre: string;
  chords: string[];
  pattern: string;
  timestamp: number;
}

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

export function CompositionHistory() {
  const [lyrics, setLyrics] = useKV<LyricComposition[]>("lyric-compositions", []);
  const [progressions, setProgressions] = useKV<ChordProgression[]>("chord-progressions", []);
  const [structures, setStructures] = useKV<SongStructure[]>("song-structures", []);
  const [editingLyrics, setEditingLyrics] = useState<string | null>(null);
  const [editedText, setEditedText] = useState("");

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const deleteLyrics = (id: string) => {
    setLyrics(current => current.filter(item => item.id !== id));
    toast.success("Lyrics deleted");
  };

  const deleteProgression = (id: string) => {
    setProgressions(current => current.filter(item => item.id !== id));
    toast.success("Progression deleted");
  };

  const deleteStructure = (id: string) => {
    setStructures(current => current.filter(item => item.id !== id));
    toast.success("Structure deleted");
  };

  const startEditingLyrics = (composition: LyricComposition) => {
    setEditingLyrics(composition.id);
    setEditedText(composition.lyrics);
  };

  const saveEditedLyrics = () => {
    if (!editingLyrics) return;
    
    setLyrics(current => 
      current.map(item => 
        item.id === editingLyrics 
          ? { ...item, lyrics: editedText }
          : item
      )
    );
    setEditingLyrics(null);
    setEditedText("");
    toast.success("Lyrics updated");
  };

  const cancelEdit = () => {
    setEditingLyrics(null);
    setEditedText("");
  };

  return (
    <div className="space-y-6">
      {lyrics.length === 0 && progressions.length === 0 && structures.length === 0 ? (
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
            <Music className="w-8 h-8 text-muted-foreground" />
          </div>
          <h3 className="text-lg font-medium mb-2">No compositions yet</h3>
          <p className="text-muted-foreground">
            Start creating lyrics, chord progressions, or song structures to see them here.
          </p>
        </div>
      ) : (
        <Tabs defaultValue="lyrics" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="lyrics" className="flex items-center gap-2">
              <Wand2 className="w-4 h-4" />
              Lyrics ({lyrics.length})
            </TabsTrigger>
            <TabsTrigger value="chords" className="flex items-center gap-2">
              <Music className="w-4 h-4" />
              Chords ({progressions.length})
            </TabsTrigger>
            <TabsTrigger value="structures" className="flex items-center gap-2">
              <Layout className="w-4 h-4" />
              Structures ({structures.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="lyrics" className="space-y-4">
            {lyrics.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No saved lyrics yet
              </div>
            ) : (
              lyrics.map((composition) => (
                <Card key={composition.id}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-lg">{composition.title}</CardTitle>
                        <div className="flex gap-2 mt-2">
                          <Badge variant="secondary">{composition.genre}</Badge>
                          <Badge variant="outline">{composition.mood}</Badge>
                          {composition.theme && (
                            <Badge variant="secondary">{composition.theme}</Badge>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">
                          {formatDate(composition.timestamp)}
                        </span>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => startEditingLyrics(composition)}
                        >
                          <Edit3 className="w-4 h-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => deleteLyrics(composition.id)}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {editingLyrics === composition.id ? (
                      <div className="space-y-4">
                        <Textarea
                          value={editedText}
                          onChange={(e) => setEditedText(e.target.value)}
                          className="min-h-[200px] font-mono text-sm"
                        />
                        <div className="flex gap-2">
                          <Button size="sm" onClick={saveEditedLyrics}>
                            Save Changes
                          </Button>
                          <Button size="sm" variant="outline" onClick={cancelEdit}>
                            Cancel
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <pre className="whitespace-pre-wrap font-mono text-sm bg-muted/50 p-4 rounded-md">
                        {composition.lyrics}
                      </pre>
                    )}
                  </CardContent>
                </Card>
              ))
            )}
          </TabsContent>

          <TabsContent value="chords" className="space-y-4">
            {progressions.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No saved chord progressions yet
              </div>
            ) : (
              progressions.map((progression) => (
                <Card key={progression.id}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-lg">{progression.name}</CardTitle>
                        <div className="flex gap-2 mt-2">
                          <Badge variant="secondary">{progression.key}</Badge>
                          <Badge variant="outline">{progression.genre}</Badge>
                          <Badge variant="secondary">{progression.pattern}</Badge>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">
                          {formatDate(progression.timestamp)}
                        </span>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => deleteProgression(progression.id)}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-2">
                      {progression.chords.map((chord, index) => (
                        <div
                          key={index}
                          className="px-4 py-2 bg-accent/10 rounded-lg border border-accent/20 font-mono font-medium"
                        >
                          {chord}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </TabsContent>

          <TabsContent value="structures" className="space-y-4">
            {structures.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No saved song structures yet
              </div>
            ) : (
              structures.map((structure) => (
                <Card key={structure.id}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-lg">{structure.name}</CardTitle>
                        <div className="flex gap-2 mt-2">
                          <Badge variant="secondary">{structure.songType}</Badge>
                          <Badge variant="outline" className="flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {formatTime(structure.totalDuration)}
                          </Badge>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">
                          {formatDate(structure.timestamp)}
                        </span>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => deleteStructure(structure.id)}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {structure.sections.map((section, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-muted/30 rounded-md">
                          <div>
                            <span className="font-medium">{section.name}</span>
                            <span className="text-sm text-muted-foreground ml-2">
                              {section.description}
                            </span>
                          </div>
                          <div className="flex gap-2 text-xs">
                            <Badge variant="outline">{formatTime(section.duration)}</Badge>
                            <Badge variant="outline">{section.bars} bars</Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}
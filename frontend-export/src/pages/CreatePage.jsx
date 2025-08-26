
import React, { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { motion, AnimatePresence } from 'framer-motion';
import { Music, Play, Pause, Save, Loader2, Wand2, Type, Download, Share2, Mic, Music2, Pencil } from 'lucide-react';
import { useToast } from '@/components/ui/use-toast';
import { useModel } from '@/contexts/ModelContext';
import { supabase } from '@/lib/customSupabaseClient';
import { useAuth } from '@/contexts/SupabaseAuthContext';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { useSubscription } from '@/contexts/SubscriptionContext';
import { useNavigate } from 'react-router-dom';
import { ToastAction } from "@/components/ui/toast";

const TrackCard = ({ song, onSave }) => {
  const { toast } = useToast();
  const navigate = useNavigate();
  const { subscription } = useSubscription();
  const isPro = !!subscription;
  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);

  const handlePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleDownload = () => {
    if (!isPro) {
      toast({
        title: 'Upgrade to Pro to Download',
        description: 'Get high-quality .wav downloads with a Pro subscription.',
        action: <ToastAction altText="Upgrade" onClick={() => navigate('/pricing')}>Upgrade</ToastAction>,
        variant: 'destructive'
      });
      return;
    }
    if (song.audio_url) {
      const link = document.createElement('a');
      link.href = song.audio_url;
      link.download = `${song.title}.wav`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else {
      toast({ title: 'Audio not available for download.', variant: 'destructive' });
    }
  };

  const handleShare = () => {
     toast({
        title: '🚧 This feature isn\'t implemented yet—but don\'t worry! You can request it in your next prompt! 🚀',
        description: 'Save to library first to enable sharing.',
    });
  }

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 50, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -50, scale: 0.9 }}
      transition={{ duration: 0.5, type: 'spring' }}
    >
      <Card className="overflow-hidden bg-black/30 border-white/10 hover:border-primary/50 transition-all duration-300 group">
        <CardHeader>
          <CardTitle className="flex items-center gap-3 text-xl font-bold text-white">
            <Music className="text-primary" />
            <span>{song.title}</span>
          </CardTitle>
          <p className="text-xs text-slate-400">Generated with: {song.model}</p>
        </CardHeader>
        <CardContent>
          <div className="w-full h-24 bg-black/20 rounded-md flex items-center justify-center">
            {song.audio_url ? (
              <audio ref={audioRef} src={song.audio_url} onEnded={() => setIsPlaying(false)} />
            ) : (
              <p className="text-slate-500 text-sm">No audio preview</p>
            )}
          </div>
        </CardContent>
        <CardFooter className="flex justify-between items-center bg-black/20 p-3">
          <Button size="icon" variant="ghost" onClick={handlePlay} disabled={!song.audio_url}>
            {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
          </Button>
          <div className="flex items-center gap-1">
            <Button size="icon" variant="ghost" onClick={onSave}>
              <Save className="h-4 w-4" />
            </Button>
            <Button size="icon" variant="ghost" onClick={handleDownload} disabled={!song.audio_url}>
              <Download className="h-4 w-4" />
            </Button>
            <Button size="icon" variant="ghost" onClick={handleShare}>
              <Share2 className="h-4 w-4" />
            </Button>
          </div>
        </CardFooter>
      </Card>
    </motion.div>
  );
};

const predefinedStyles = ["electronic", "classical", "punk", "gospel pop", "indie rock", "breakbeat"];

export function CreatePage() {
  const [prompt, setPrompt] = useState('An energetic, old-school breakbeat track with classic drum loops and a funky bassline.');
  const [title, setTitle] = useState('');
  const [lyrics, setLyrics] = useState('');
  const [lyricsMode, setLyricsMode] = useState('auto');
  const [isInstrumental, setIsInstrumental] = useState(false);
  const [styles, setStyles] = useState(['breakbeat']);
  
  const [generatedSongs, setGeneratedSongs] = useState([]);
  const { toast } = useToast();
  const { models, selectedModel, isGenerating, generateSong } = useModel();
  const { user, session } = useAuth();

  const handleGenerate = async () => {
    let finalPrompt = prompt;
    if (styles.length > 0) {
      finalPrompt = `${prompt} in the style of ${styles.join(', ')}`;
    } else if (prompt.trim()) {
      finalPrompt = prompt;
    } else {
      toast({ title: 'Missing Style', description: 'Please add at least one style tag to guide the AI.', variant: 'destructive' });
      return;
    }
    
    if (!finalPrompt.trim() && lyricsMode === 'auto' && !isInstrumental) {
        toast({ title: 'Missing details', description: 'Please describe your song idea or provide lyrics.', variant: 'destructive' });
        return;
    }

    if (!finalPrompt.trim() && lyricsMode === 'write' && !lyrics.trim()) {
        toast({ title: 'Missing details', description: 'Please describe the style and add your lyrics.', variant: 'destructive' });
        return;
    }

    const songDetails = {
      userProvidedTitle: title.trim() || null,
      genrePrompt: finalPrompt,
      lyricsValue: isInstrumental ? '[Instrumental]' : (lyricsMode === 'auto' ? finalPrompt : lyrics),
      useAILyrics: !isInstrumental && lyricsMode === 'auto',
    };
    
    const newSong = await generateSong(songDetails, session.access_token);

    if (newSong) {
      setGeneratedSongs(prevSongs => [newSong, ...prevSongs]);
    }
  };

  const handleSaveToLibrary = async (songToSave) => {
    if (!songToSave || !user) return;
    
    const { error } = await supabase
      .from('songs')
      .insert({
        user_id: user.id,
        prompt: prompt,
        title: songToSave.title,
        lyrics: songToSave.lyrics,
        model: songToSave.model,
        audio_url: songToSave.audio_url,
      });

    if (error) {
        toast({ title: 'Error Saving Song', description: error.message, variant: 'destructive' });
    } else {
        toast({ title: 'Song Saved!', description: `'${songToSave.title}' has been saved to your library.` });
    }
  };

  const toggleStyle = (style) => {
    setStyles(prev => 
      prev.includes(style) 
        ? prev.filter(s => s !== style) 
        : [...prev, style]
    );
  };


  return (
    <div className="h-full flex flex-col items-center justify-start py-8">
      <div className="w-full max-w-6xl mx-auto flex flex-col flex-grow">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl md:text-6xl font-extrabold tracking-tighter bg-clip-text text-transparent bg-gradient-to-b from-white to-slate-400">
            Create Your Sound
          </h1>
          <p className="text-slate-400 mt-4 text-lg md:text-xl max-w-2xl mx-auto">
            Use our advanced AI to craft songs from your ideas, lyrics, and styles.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full max-w-6xl mx-auto">
          {/* Left Column: Lyrics & Title */}
          <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }} className="flex flex-col gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Music2 className="text-primary w-5 h-5" /> Song Details</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                 <div>
                    <Label htmlFor="title" className="mb-2 block font-semibold text-slate-300">Title (Optional)</Label>
                    <Input id="title" placeholder="Enter a title for your song" value={title} onChange={(e) => setTitle(e.target.value)} />
                </div>
                 <div>
                    <Label htmlFor="prompt" className="mb-2 block font-semibold text-slate-300">Main Idea / Prompt</Label>
                    <Input id="prompt" placeholder="e.g., A hopeful song about new beginnings" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Mic className="text-primary w-5 h-5" /> Lyrics</CardTitle>
                <div className="flex items-center justify-between pt-2 flex-wrap gap-2">
                    <p className="text-sm text-slate-400">Generate with AI, write your own, or go instrumental.</p>
                     <div className="flex items-center space-x-2">
                        <Label htmlFor="instrumental-switch" className="text-slate-400 font-medium">Instrumental</Label>
                        <Switch id="instrumental-switch" checked={isInstrumental} onCheckedChange={setIsInstrumental}/>
                    </div>
                </div>
              </CardHeader>
              <CardContent>
                <AnimatePresence mode="wait">
                  <motion.div
                    key={isInstrumental ? 'instrumental' : 'lyrics'}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -10 }}
                    transition={{ duration: 0.2 }}
                  >
                  {isInstrumental ? (
                    <div className="text-center py-10 border-2 border-dashed border-border rounded-lg bg-black/20">
                      <p className="text-slate-400">Generating an instrumental track.</p>
                    </div>
                  ) : (
                    <Tabs value={lyricsMode} onValueChange={setLyricsMode} className="w-full">
                      <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="auto" className="flex items-center gap-2"><Wand2 className="w-4 h-4"/> Auto</TabsTrigger>
                        <TabsTrigger value="write" className="flex items-center gap-2"><Type className="w-4 h-4"/> Write Lyrics</TabsTrigger>
                      </TabsList>
                      <TabsContent value="auto">
                        <div className="text-center py-10 border-2 border-dashed border-border rounded-lg bg-black/20">
                          <p className="text-slate-400">AI will generate lyrics based on your prompt.</p>
                        </div>
                      </TabsContent>
                      <TabsContent value="write">
                        <Textarea 
                          placeholder="Add your own lyrics here..."
                          value={lyrics}
                          onChange={(e) => setLyrics(e.target.value)}
                          className="min-h-[120px] bg-black/20"
                        />
                      </TabsContent>
                    </Tabs>
                  )}
                  </motion.div>
                </AnimatePresence>
              </CardContent>
            </Card>
          </motion.div>

          {/* Right Column: Style & Generation */}
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }} className="flex flex-col gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Pencil className="text-primary w-5 h-5" /> Styles</CardTitle>
                <p className="text-sm text-slate-400 pt-2">Define the musical genre for the AI.</p>
              </CardHeader>
              <CardContent>
                <Input placeholder="Enter custom style tags..." onKeyDown={(e) => {
                  if (e.key === 'Enter' && e.currentTarget.value.trim()) {
                    toggleStyle(e.currentTarget.value.trim());
                    e.currentTarget.value = '';
                    e.preventDefault();
                  }
                }}/>
                <div className="flex flex-wrap gap-2 mt-4">
                  {predefinedStyles.map(style => (
                    <Badge
                      key={style}
                      variant={styles.includes(style) ? 'default' : 'secondary'}
                      onClick={() => toggleStyle(style)}
                      className="cursor-pointer transition-all duration-200"
                    >
                      {style}
                    </Badge>
                  ))}
                   {styles.filter(s => !predefinedStyles.includes(s)).map(style => (
                     <Badge key={style} variant="default" className="transition-all duration-200">
                      {style}
                      <button onClick={() => setStyles(styles.filter(s => s !== style))} className="ml-2 text-primary-foreground/70 hover:text-primary-foreground">x</button>
                    </Badge>
                   ))}
                </div>
              </CardContent>
            </Card>
            
            <div className="sticky top-4">
                <Button onClick={handleGenerate} disabled={isGenerating} size="lg" className="w-full h-16 text-xl font-bold button-glow bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 shadow-lg shadow-primary/30">
                  {isGenerating ? <Loader2 className="h-6 w-6 animate-spin" /> : 'Generate Music'}
                </Button>
            </div>
          </motion.div>
        </div>

        {/* Track Feed */}
        <div className="w-full max-w-6xl mx-auto mt-12">
          <AnimatePresence>
            {isGenerating && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="flex items-center justify-center gap-3 text-lg text-primary p-4 rounded-lg mb-8"
              >
                <Loader2 className="h-6 w-6 animate-spin" />
                <span>Generating with Beat Addicts...</span>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            <AnimatePresence>
              {generatedSongs.map((song, index) => (
                <TrackCard 
                  key={song.title + index} 
                  song={song}
                  onSave={() => handleSaveToLibrary(song)}
                />
              ))}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}

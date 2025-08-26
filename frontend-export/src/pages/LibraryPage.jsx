
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Card, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { motion, AnimatePresence } from 'framer-motion';
import { Library, Music, Search, Loader2, Trash2, Play, Pause, Download, Share2, GripVertical, List } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { supabase } from '@/lib/customSupabaseClient';
import { useAuth } from '@/contexts/SupabaseAuthContext';
import { useToast } from '@/components/ui/use-toast';
import { Button } from '@/components/ui/button';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useSubscription } from '@/contexts/SubscriptionContext';
import { useNavigate } from 'react-router-dom';
import { ToastAction } from "@/components/ui/toast";
import { Badge } from '@/components/ui/badge';

const Waveform = ({ isPlaying }) => {
    const bars = useMemo(() => Array.from({ length: 40 }, () => Math.random()), []);
    return (
        <div className="w-full h-24 flex items-center justify-center gap-0.5 overflow-hidden">
            {bars.map((height, i) => (
                <motion.div
                    key={i}
                    className="w-1 bg-primary/40"
                    style={{ height: `${height * 100}%` }}
                    initial={{ height: '2%' }}
                    animate={isPlaying ? { height: `${height * 100}%` } : { height: '2%' }}
                    transition={{ duration: isPlaying ? (0.2 + Math.random() * 0.3) : 0.5, delay: i * 0.01, repeat: isPlaying ? Infinity : 0, repeatType: 'mirror', ease: 'easeInOut' }}
                />
            ))}
        </div>
    );
};

const SongCard = ({ song, onDelete, onShare, layout, isPro, onPlay, currentlyPlaying, setCurrentlyPlaying }) => {
  const { toast } = useToast();
  const navigate = useNavigate();
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const audioRef = useRef(null);
  const isPlaying = currentlyPlaying === song.id;

  const handlePlayClick = (e) => {
    e.stopPropagation();
    if (isPlaying) {
      setCurrentlyPlaying(null);
    } else {
      if (audioRef.current) {
        onPlay(song.id, audioRef);
      }
    }
  };

  useEffect(() => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.play().catch(e => console.error("Audio play failed:", e));
      } else {
        audioRef.current.pause();
        audioRef.current.currentTime = 0;
      }
    }
  }, [isPlaying]);

  const handleDeleteConfirm = (e) => {
    e.stopPropagation();
    onDelete();
    setIsDeleteDialogOpen(false);
  }

  const handleShareClick = (e) => {
    e.stopPropagation();
    onShare();
  };

  const handleDownloadClick = (e) => {
    e.stopPropagation();
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

  const creationDate = new Date(song.created_at).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  if (layout === 'grid') {
    return (
      <>
        <motion.div 
            layout 
            initial={{ opacity: 0, scale: 0.8 }} 
            animate={{ opacity: 1, scale: 1 }} 
            exit={{ opacity: 0, scale: 0.8 }}
            className="w-full cursor-pointer group"
            onClick={handlePlayClick}
        >
          <Card className="bg-black/30 border-white/10 hover:border-primary/50 transition-all duration-300 flex flex-col h-full">
            <div className="relative overflow-hidden rounded-t-2xl">
                <Waveform isPlaying={isPlaying} />
                <audio ref={audioRef} src={song.audio_url} onEnded={() => setCurrentlyPlaying(null)} preload="metadata" />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent" />
                 {song.is_public && <Badge variant="secondary" className="absolute top-2 right-2 bg-primary/20 text-primary border-primary/30 text-xs">Public</Badge>}
                <Button size="icon" variant="default" className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-14 h-14 rounded-full opacity-0 group-hover:opacity-100 transition-opacity shadow-lg shadow-primary/30">
                    {isPlaying ? <Pause className="h-6 w-6 fill-white" /> : <Play className="h-6 w-6 fill-white ml-1" />}
                </Button>
            </div>
            <CardHeader className="flex-grow pt-4">
              <CardTitle className="text-lg font-bold text-white truncate">{song.title}</CardTitle>
              <p className="text-sm text-slate-400">{creationDate}</p>
            </CardHeader>
            <CardFooter className="p-2 border-t border-border">
              <Button size="icon" variant="ghost" className={`hover:text-white ${song.is_public ? 'text-primary' : 'text-slate-400'}`} onClick={handleShareClick}>
                <Share2 className="h-4 w-4" />
              </Button>
              <Button size="icon" variant="ghost" className="text-slate-400 hover:text-white" onClick={handleDownloadClick}>
                <Download className="h-4 w-4" />
              </Button>
              <Button size="icon" variant="ghost" className="ml-auto text-slate-500 hover:bg-red-500/10 hover:text-red-400" onClick={(e) => {e.stopPropagation(); setIsDeleteDialogOpen(true)}}>
                <Trash2 className="h-4 w-4" />
              </Button>
            </CardFooter>
          </Card>
        </motion.div>

        <AlertDialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
            <AlertDialogContent>
                <AlertDialogHeader>
                    <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                    <AlertDialogDescription>
                        This action cannot be undone. This will permanently delete your song "{song.title}".
                    </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                    <AlertDialogCancel onClick={(e) => e.stopPropagation()}>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={handleDeleteConfirm} className="bg-destructive hover:bg-destructive/90">Delete</AlertDialogAction>
                </AlertDialogFooter>
            </AlertDialogContent>
        </AlertDialog>
      </>
    );
  }
  
  return (
    <>
      <motion.div 
          layout 
          initial={{ opacity: 0, y: 20 }} 
          animate={{ opacity: 1, y: 0 }} 
          exit={{ opacity: 0, y: -20 }}
          className="w-full"
      >
          <Card className="bg-black/30 border-white/10 hover:border-primary/50 transition-all duration-300 flex items-center p-3 gap-4">
              <Button size="icon" variant="ghost" className='flex-shrink-0' onClick={handlePlayClick}>
                  {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
              </Button>
              <div className="flex-grow">
                  <p className="font-bold text-white truncate">{song.title}</p>
                  <p className="text-xs text-slate-400">{creationDate}</p>
              </div>
              {song.is_public && <Badge variant="secondary" className="bg-primary/20 text-primary border-primary/30 text-xs flex-shrink-0">Public</Badge>}
              <div className="flex items-center gap-1 flex-shrink-0">
                  <Button size="icon" variant="ghost" className={`hover:text-white ${song.is_public ? 'text-primary' : 'text-slate-400'}`} onClick={handleShareClick}>
                      <Share2 className="h-4 w-4" />
                  </Button>
                  <Button size="icon" variant="ghost" className="text-slate-400 hover:text-white" onClick={handleDownloadClick}>
                      <Download className="h-4 w-4" />
                  </Button>
                  <Button size="icon" variant="ghost" className="text-slate-500 hover:bg-red-500/10 hover:text-red-400" onClick={() => setIsDeleteDialogOpen(true)}>
                      <Trash2 className="h-4 w-4" />
                  </Button>
              </div>
          </Card>
      </motion.div>
      <AlertDialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
          <AlertDialogContent>
              <AlertDialogHeader>
                  <AlertDialogTitle>Are you absolutely sure?</AlertDialogTitle>
                  <AlertDialogDescription>
                      This action cannot be undone. This will permanently delete your song "{song.title}".
                  </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={() => { onDelete(); setIsDeleteDialogOpen(false); }} className="bg-destructive hover:bg-destructive/90">Delete</AlertDialogAction>
              </AlertDialogFooter>
          </AlertDialogContent>
      </AlertDialog>
    </>
  );
};

export function LibraryPage() {
  const [songs, setSongs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [layout, setLayout] = useState('grid');
  const { user } = useAuth();
  const { toast } = useToast();
  const { subscription } = useSubscription();
  const isPro = !!subscription;
  const [currentlyPlaying, setCurrentlyPlaying] = useState(null);
  const audioRefs = useRef({});

  const handlePlay = (songId, audioRef) => {
    if (currentlyPlaying && currentlyPlaying !== songId) {
      const currentAudioRef = audioRefs.current[currentlyPlaying];
      if (currentAudioRef && currentAudioRef.current) {
        currentAudioRef.current.pause();
      }
    }
    audioRefs.current[songId] = audioRef;
    setCurrentlyPlaying(songId);
  };

  useEffect(() => {
    const fetchSongs = async () => {
      if (!user) return;
      setLoading(true);
      const { data, error } = await supabase
        .from('songs')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (error) {
        toast({ title: 'Error fetching songs', description: error.message, variant: 'destructive' });
      } else {
        setSongs(data);
      }
      setLoading(false);
    };
    fetchSongs();
  }, [user, toast]);
  
  const handleShare = async (song) => {
    const newIsPublic = !song.is_public;
    const { data, error } = await supabase
      .from('songs')
      .update({ is_public: newIsPublic })
      .eq('id', song.id)
      .select()
      .single();

    if (error) {
      toast({ title: 'Error updating song', description: error.message, variant: 'destructive' });
    } else {
      setSongs(songs.map(s => (s.id === song.id ? data : s)));
      toast({
        title: `Song is now ${newIsPublic ? 'public' : 'private'}`,
        description: newIsPublic ? 'Your song is now visible in the Collab area.' : 'Your song is no longer visible to others.',
      });
    }
  };

  const handleDelete = async (songId) => {
    const originalSongs = [...songs];
    setSongs(songs.filter(s => s.id !== songId));

    const { error } = await supabase.from('songs').delete().eq('id', songId);
    
    if (error) {
      setSongs(originalSongs);
      toast({ title: 'Error deleting song', description: error.message, variant: 'destructive' });
    } else {
      toast({ title: 'Song deleted successfully' });
    }
  };

  const filteredSongs = songs.filter(song => 
    song.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (song.prompt && song.prompt.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div className="container mx-auto max-w-7xl">
        <div className="flex justify-between items-center mb-8 flex-wrap gap-4">
            <h1 className="text-4xl lg:text-5xl font-extrabold tracking-tighter flex items-center gap-3 bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
                <Library className="w-8 h-8 lg:w-10 lg:h-10 text-primary" />
                My Library
            </h1>
            <div className="flex items-center gap-2 w-full sm:w-auto">
                 <div className="relative flex-grow sm:flex-grow-0 sm:w-auto sm:min-w-[300px]">
                    <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500" />
                    <Input 
                        placeholder="Search library..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="bg-card border-border pl-11 focus:border-primary w-full"
                    />
                </div>
                <div className="bg-card p-1 rounded-lg flex items-center border border-border">
                    <Button variant={layout === 'grid' ? 'secondary' : 'ghost'} size="icon" onClick={() => setLayout('grid')} className={`${layout === 'grid' ? 'bg-primary/20 text-primary' : 'text-slate-400'}`}>
                        <GripVertical className='h-5 w-5' />
                    </Button>
                     <Button variant={layout === 'list' ? 'secondary' : 'ghost'} size="icon" onClick={() => setLayout('list')} className={`${layout === 'list' ? 'bg-primary/20 text-primary' : 'text-slate-400'}`}>
                        <List className='h-5 w-5' />
                    </Button>
                </div>
            </div>
        </div>

      {loading ? (
        <div className="flex justify-center items-center py-20">
          <Loader2 className="h-12 w-12 animate-spin text-primary"/>
        </div>
      ) : filteredSongs.length > 0 ? (
        <motion.div 
            layout
            className={`transition-all duration-500 ${layout === 'grid' ? 'grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6' : 'space-y-3'}`}
        >
            <AnimatePresence>
                {filteredSongs.map(song => (
                    <SongCard 
                      key={song.id} 
                      song={song} 
                      onDelete={() => handleDelete(song.id)} 
                      onShare={() => handleShare(song)} 
                      layout={layout} 
                      isPro={isPro}
                      onPlay={handlePlay}
                      currentlyPlaying={currentlyPlaying}
                      setCurrentlyPlaying={setCurrentlyPlaying}
                    />
                ))}
            </AnimatePresence>
        </motion.div>
      ) : (
        <div className="text-center py-20 px-6 bg-black/20 border-2 border-dashed border-border rounded-xl mt-12">
            <Music className="mx-auto h-16 w-16 text-slate-600" />
            <h3 className="mt-4 text-2xl font-semibold text-white">Your library is waiting</h3>
            <p className="mt-2 text-slate-400">
              Songs you create will appear here. Start by describing an idea on the Home page.
            </p>
        </div>
      )}
    </div>
  );
}

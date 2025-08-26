
import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { motion, AnimatePresence } from 'framer-motion';
import { Users, Loader2, Play, Download, Share2, Music } from 'lucide-react';
import { supabase } from '@/lib/customSupabaseClient';
import { useToast } from '@/components/ui/use-toast';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useSubscription } from '@/contexts/SubscriptionContext';
import { useNavigate } from 'react-router-dom';
import { ToastAction } from "@/components/ui/toast";

const Waveform = () => {
    const bars = useMemo(() => Array.from({ length: 40 }, () => Math.random()), []);
    return (
        <div className="w-full h-24 flex items-center justify-center gap-0.5 overflow-hidden">
            {bars.map((height, i) => (
                <motion.div
                    key={i}
                    className="w-1 bg-primary/40"
                    style={{ height: `${height * 100}%` }}
                    initial={{ height: '2%' }}
                    animate={{ height: `${height * 100}%`}}
                    transition={{ duration: 0.5 + Math.random(), delay: i * 0.01, repeat: Infinity, repeatType: 'mirror', ease: 'easeInOut' }}
                />
            ))}
        </div>
    );
};

const CollabSongCard = ({ song, isPro }) => {
  const { toast } = useToast();
  const navigate = useNavigate();

  const showNotImplementedToast = () => {
    toast({
        title: '🚧 This feature isn\'t implemented yet—but don\'t worry! You can request it in your next prompt! 🚀',
    });
  }

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
    toast({
        title: '🚧 Download starting soon!',
        description: 'Your .wav file is being prepared.',
    });
  };

  const creationDate = new Date(song.created_at).toLocaleDateString('en-US', {
    month: 'long',
    day: 'numeric',
  });

  return (
    <motion.div 
        layout 
        initial={{ opacity: 0, scale: 0.8 }} 
        animate={{ opacity: 1, scale: 1 }} 
        exit={{ opacity: 0, scale: 0.8 }}
        className="w-full cursor-pointer group"
        onClick={showNotImplementedToast}
    >
      <Card className="bg-black/30 border-white/10 hover:border-primary/50 transition-all duration-300 flex flex-col h-full">
        <div className="relative overflow-hidden rounded-t-2xl">
            <Waveform />
            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent" />
            <Button size="icon" variant="default" className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-14 h-14 rounded-full opacity-0 group-hover:opacity-100 transition-opacity shadow-lg shadow-primary/30">
                <Play className="h-6 w-6 fill-white ml-1" />
            </Button>
        </div>
        <CardHeader className="flex-grow pt-4">
          <CardTitle className="text-lg font-bold text-white truncate">{song.title}</CardTitle>
          <p className="text-sm text-slate-400">Created {creationDate}</p>
        </CardHeader>
        <CardFooter className="p-2 border-t border-border">
          <Button size="icon" variant="ghost" className="text-slate-400 hover:text-white" onClick={(e) => { e.stopPropagation(); showNotImplementedToast(); }}>
            <Share2 className="h-4 w-4" />
          </Button>
          <Button size="icon" variant="ghost" className="text-slate-400 hover:text-white" onClick={handleDownloadClick}>
            <Download className="h-4 w-4" />
          </Button>
        </CardFooter>
      </Card>
    </motion.div>
  );
};


export function ExplorePage() {
    const [publicSongs, setPublicSongs] = useState([]);
    const [loading, setLoading] = useState(true);
    const { toast } = useToast();
    const { subscription } = useSubscription();
    const isPro = !!subscription;

    useEffect(() => {
        const fetchPublicSongs = async () => {
            setLoading(true);
            const { data, error } = await supabase
                .from('songs')
                .select('*')
                .eq('is_public', true)
                .order('created_at', { ascending: false });
            
            if (error) {
                toast({ title: 'Error fetching public songs', description: error.message, variant: 'destructive' });
            } else {
                setPublicSongs(data);
            }
            setLoading(false);
        };
        fetchPublicSongs();
    }, [toast]);

    return (
        <div className="container mx-auto max-w-7xl">
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center md:text-left mb-12"
            >
                <h1 className="text-5xl md:text-6xl font-extrabold tracking-tighter flex items-center justify-center md:justify-start gap-4 bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent">
                    <Users className="w-12 h-12" />
                    Collab Area
                </h1>
                <p className="text-slate-400 mt-4 text-lg max-w-2xl mx-auto md:mx-0">
                    Discover, share, and get inspired by music from the BeatAddicts community.
                </p>
            </motion.div>
            
            {loading ? (
                <div className="flex justify-center items-center py-20">
                    <Loader2 className="h-12 w-12 animate-spin text-primary"/>
                </div>
            ) : publicSongs.length > 0 ? (
                <motion.div
                    layout
                    className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6"
                >
                    <AnimatePresence>
                        {publicSongs.map(song => (
                           <CollabSongCard key={song.id} song={song} isPro={isPro} />
                        ))}
                    </AnimatePresence>
                </motion.div>
            ) : (
                <div className="text-center py-20 px-6 bg-black/20 border-2 border-dashed border-border rounded-xl mt-12">
                    <Music className="mx-auto h-16 w-16 text-slate-600" />
                    <h3 className="mt-4 text-2xl font-semibold text-white">The Stage is Empty</h3>
                    <p className="mt-2 text-slate-400">
                        No public songs have been shared yet. Be the first! Share a track from your library to get the party started.
                    </p>
                </div>
            )}
        </div>
    );
}

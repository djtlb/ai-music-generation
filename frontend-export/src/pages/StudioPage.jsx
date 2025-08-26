
import React from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { MicOff as MicVocal, Zap, Lock } from 'lucide-react';
import { useToast } from "@/components/ui/use-toast";

export function StudioPage() {
    const { toast } = useToast();

    const handleUpgrade = () => {
        toast({
            title: '🚀 Coming Soon!',
            description: "The Studio tier and production tools are in development. Stay tuned!",
        });
    }

  return (
    <div className="space-y-8">
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        <h1 className="text-4xl font-bold tracking-tighter text-white flex items-center gap-3">
            <MicVocal className="w-10 h-10 text-purple-400"/>
            Production Studio
        </h1>
        <p className="text-slate-400 mt-2">Unlock advanced tools and fine-tune your creations.</p>
      </motion.div>

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center py-20 px-6 bg-gradient-to-br from-purple-900/30 to-slate-900/30 rounded-lg border-2 border-purple-500/30 relative overflow-hidden"
    >
        <div className="absolute top-0 left-0 w-full h-full bg-grid-slate-800/40 [mask-image:radial-gradient(ellipse_at_center,transparent_10%,black)] z-0"></div>
        <div className="relative z-10">
            <Zap className="mx-auto h-12 w-12 text-yellow-300" />
            <h3 className="mt-4 text-3xl font-extrabold text-white">Upgrade to the Studio Tier</h3>
            <p className="mt-2 text-lg max-w-2xl mx-auto text-slate-300">
                Gain access to our professional-grade production suite, including multi-track editing, stem separation, advanced vocal synthesis, and more.
            </p>
            <Button onClick={handleUpgrade} size="lg" className="mt-8 bg-purple-600 hover:bg-purple-700 font-bold text-lg">
                <Lock className="mr-2 h-5 w-5"/>
                Unlock Studio Access
            </Button>
        </div>
    </motion.div>
    </div>
  );
}

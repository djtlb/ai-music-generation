
import React, { useState } from 'react';
import { useAuth } from '@/contexts/SupabaseAuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { useToast } from '@/components/ui/use-toast';
import { motion, AnimatePresence } from 'framer-motion';
import { Music, Loader2 } from 'lucide-react';

const AuthForm = ({ isSignUp, onSubmit, loading }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isSignUp) {
      onSubmit(email, password, { data: { username } });
    } else {
      onSubmit(email, password);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <Label htmlFor="email">Email</Label>
        <Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required placeholder="you@example.com" />
      </motion.div>
      
      <AnimatePresence>
      {isSignUp && (
        <motion.div 
            initial={{ opacity: 0, height: 0, y: 10 }} 
            animate={{ opacity: 1, height: 'auto', y: 0 }} 
            exit={{ opacity: 0, height: 0, y: 10 }}
            transition={{ delay: 0.2 }}
            className="space-y-2 overflow-hidden">
            <Label htmlFor="username">Username</Label>
            <Input id="username" type="text" value={username} onChange={(e) => setUsername(e.target.value)} required placeholder="your_username" />
        </motion.div>
      )}
      </AnimatePresence>

      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
        <Label htmlFor="password">Password</Label>
        <Input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} required placeholder="••••••••" />
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
        <Button type="submit" disabled={loading} size="lg" className="w-full bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-lg font-bold py-6 button-glow shadow-lg shadow-primary/30">
          {loading ? <Loader2 className="h-5 w-5 animate-spin" /> : (isSignUp ? 'Create Account' : 'Sign In')}
        </Button>
      </motion.div>
    </form>
  );
};

export function Auth() {
  const [isSignUp, setIsSignUp] = useState(false);
  const [loading, setLoading] = useState(false);
  const { signUp, signIn } = useAuth();
  const { toast } = useToast();

  const handleAuthAction = async (email, password, options) => {
    setLoading(true);
    const { error } = isSignUp ? await signUp(email, password, options) : await signIn(email, password);
    if (!error && isSignUp) {
      toast({ title: 'Success!', description: 'Check your email for a confirmation link.' });
      setIsSignUp(false);
    }
    if (error) {
        toast({ title: 'Authentication Error', description: error.message, variant: 'destructive' });
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <motion.div 
        initial={{ opacity: 0, scale: 0.9, y: -20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="w-full max-w-md mx-auto bg-black/40 backdrop-blur-2xl rounded-2xl shadow-2xl shadow-black/50 p-8 border border-white/10 text-white"
      >
        <div className="text-center mb-8">
            <div className="flex justify-center items-center gap-3 mb-4">
              <div className="w-12 h-12 bg-gradient-to-br from-primary to-accent rounded-xl flex items-center justify-center shadow-lg shadow-primary/30">
                  <Music className="w-7 h-7 text-white"/>
              </div>
              <h1 className="text-4xl font-black tracking-tighter bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">BeatAddicts</h1>
            </div>
            <p className="text-slate-400">
                {isSignUp ? 'Create an account to start generating music.' : 'Sign in to access your studio.'}
            </p>
        </div>

        <div className="flex justify-center mb-6">
            <div className="bg-black/30 border border-white/10 p-1 rounded-lg flex gap-1">
                <Button onClick={() => setIsSignUp(false)} variant="ghost" className={`px-6 transition-all rounded-md relative ${!isSignUp ? 'text-white' : 'text-slate-400 hover:text-white'}`}>
                  Sign In
                  {!isSignUp && <motion.div className="absolute inset-0 bg-primary/20 rounded-md -z-10" layoutId="auth-tab" />}
                </Button>
                <Button onClick={() => setIsSignUp(true)} variant="ghost" className={`px-6 transition-all rounded-md relative ${isSignUp ? 'text-white' : 'text-slate-400 hover:text-white'}`}>
                  Sign Up
                  {isSignUp && <motion.div className="absolute inset-0 bg-primary/20 rounded-md -z-10" layoutId="auth-tab" />}
                </Button>
            </div>
        </div>

        <AnimatePresence mode="wait">
            <motion.div
                key={isSignUp ? 'signup' : 'signin'}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.2, ease: 'easeInOut' }}
            >
                <AuthForm isSignUp={isSignUp} onSubmit={handleAuthAction} loading={loading} />
            </motion.div>
        </AnimatePresence>
      </motion.div>
    </div>
  );
}


import React, { useState, useEffect } from 'react';
import { useAuth } from '@/contexts/SupabaseAuthContext';
import { useProfile } from '@/contexts/ProfileContext';
import { Auth } from '@/components/Auth';
import { Button } from '@/components/ui/button';
import { LogOut, Music, Library, Home, ShieldCheck, Gem, Loader2, UserCircle, Users } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { Routes, Route, NavLink, useLocation, Navigate } from 'react-router-dom';
import { CreatePage } from '@/pages/CreatePage';
import { LibraryPage } from '@/pages/LibraryPage';
import { ExplorePage } from '@/pages/ExplorePage';
import { AdminPage } from '@/pages/AdminPage';
import { PricingPage } from '@/pages/PricingPage';
import { AccountPage } from '@/pages/AccountPage';
import { Toaster } from "@/components/ui/toaster";

const navItems = [
  { path: '/', label: 'Home', icon: Home },
  { path: '/library', label: 'My Library', icon: Library },
  { path: '/collab', label: 'Collab', icon: Users },
  { path: '/pricing', label: 'Pricing', icon: Gem },
];

const adminNavItems = [
  { path: '/admin', label: 'Admin', icon: ShieldCheck },
];

const useIsMobile = () => {
    const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
    useEffect(() => {
        const handleResize = () => setIsMobile(window.innerWidth < 768);
        window.addEventListener('resize', handleResize);
        return () => window.removeEventListener('resize', handleResize);
    }, []);
    return isMobile;
};

const Sidebar = ({ isAdmin, onSignOut, profile }) => (
    <motion.aside
        initial={{ x: -250 }}
        animate={{ x: 0 }}
        transition={{ duration: 0.7, ease: [0.25, 1, 0.5, 1] }}
        className="fixed top-0 left-0 h-full w-64 bg-black/30 backdrop-blur-xl p-4 flex-col justify-between border-r border-border hidden md:flex"
    >
        <div>
            <div className="flex items-center gap-3 mb-12 p-2">
                <motion.div
                    whileHover={{ rotate: [0, 15, -10, 0], scale: 1.1 }}
                    className="w-10 h-10 bg-gradient-to-br from-primary to-accent rounded-lg flex items-center justify-center shadow-lg shadow-primary/20"
                >
                    <Music className="text-white w-6 h-6" />
                </motion.div>
                <span className="text-3xl font-black tracking-tighter bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">BeatAddicts</span>
            </div>
            <nav className="flex flex-col gap-1">
                {navItems.map((item) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        end={item.path === '/'}
                        className={({ isActive }) => `nav-link group ${isActive ? 'active' : ''}`}
                    >
                        <item.icon className="w-5 h-5 transition-transform duration-300" />
                        <span>{item.label}</span>
                        <div className="nav-link-glow" />
                    </NavLink>
                ))}
            </nav>
            {isAdmin && (
                <div className='mt-8'>
                    <p className='px-3 text-xs font-semibold uppercase text-slate-500 mb-2'>Admin</p>
                    <nav className="flex flex-col gap-1">
                        {adminNavItems.map((item) => (
                            <NavLink
                                key={item.path}
                                to={item.path}
                                className={({ isActive }) => `nav-link group ${isActive ? 'active' : ''}`}
                            >
                                <item.icon className="w-5 h-5 transition-transform duration-300" />
                                <span>{item.label}</span>
                                <div className="nav-link-glow" />
                            </NavLink>
                        ))}
                    </nav>
                </div>
            )}
        </div>
        <div className="flex flex-col gap-1 border-t border-border pt-4">
            <NavLink to="/account" className={({ isActive }) => `flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-slate-300 hover:bg-white/5 hover:text-white ${isActive ? 'bg-white/5 text-white' : ''}`}>
                <div className='w-8 h-8 bg-slate-700 rounded-full flex items-center justify-center'>
                    <UserCircle className="w-5 h-5" />
                </div>
                <div className='flex flex-col'>
                    <span className='font-semibold text-sm leading-tight'>{profile?.username || 'User'}</span>
                    <span className='text-xs text-slate-400 leading-tight'>Settings</span>
                </div>
            </NavLink>
            <Button onClick={onSignOut} variant="ghost" className="justify-start gap-3 text-slate-400 hover:bg-red-500/10 hover:text-red-400 h-auto py-2.5 px-3">
                <LogOut className="w-5 h-5" /> Sign Out
            </Button>
        </div>
    </motion.aside>
);

const MobileNav = ({ onSignOut }) => {
    const location = useLocation();
    return (
        <motion.div
            initial={{ y: 100 }}
            animate={{ y: 0 }}
            exit={{ y: 100 }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
            className="fixed bottom-0 left-0 right-0 h-20 bg-black/50 backdrop-blur-xl border-t border-border z-50 flex items-center justify-around px-2 md:hidden"
        >
            {navItems.map((item) => (
                <NavLink
                    key={item.path}
                    to={item.path}
                    end={item.path === '/'}
                    className={({ isActive }) => `flex flex-col items-center justify-center w-16 h-16 rounded-lg transition-colors duration-300 ${isActive ? 'text-primary' : 'text-slate-400 hover:text-white'}`}
                >
                    <item.icon className="w-6 h-6 mb-1" />
                    <span className="text-xs font-medium">{item.label}</span>
                </NavLink>
            ))}
            <NavLink
                to="/account"
                className={({ isActive }) => `flex flex-col items-center justify-center w-16 h-16 rounded-lg transition-colors duration-300 ${isActive ? 'text-primary' : 'text-slate-400 hover:text-white'}`}
            >
                <UserCircle className="w-6 h-6 mb-1" />
                <span className="text-xs font-medium">Account</span>
            </NavLink>
        </motion.div>
    );
};


function App() {
  const { session, signOut, isLoading } = useAuth();
  const { profile, loading: profileLoading } = useProfile();
  const location = useLocation();
  const isMobile = useIsMobile();
  const loading = isLoading || profileLoading;

  if (loading) {
    return (
      <div className="min-h-screen w-full flex items-center justify-center bg-background">
        <Loader2 className="h-10 w-10 animate-spin text-primary" />
      </div>
    );
  }

  if (!session) {
    return (
      <>
        <Auth />
        <Toaster />
      </>
    );
  }

  const isAdmin = profile?.role === 'admin';

  return (
    <div className='min-h-screen w-full flex bg-background text-white font-sans'>
      {!isMobile && <Sidebar isAdmin={isAdmin} onSignOut={signOut} profile={profile} />}

      <div className="flex-1 flex flex-col overflow-hidden md:ml-64">
        <main className="flex-1 overflow-y-auto pb-20 md:pb-0">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.4, ease: "easeInOut" }}
              className="h-full p-4 sm:p-6 lg:p-8"
            >
              <Routes>
                <Route path="/" element={<CreatePage />} />
                <Route path="/library" element={<LibraryPage />} />
                <Route path="/collab" element={<ExplorePage />} />
                <Route path="/pricing" element={<PricingPage />} />
                <Route path="/account" element={<AccountPage />} />
                {isAdmin && <Route path="/admin" element={<AdminPage />} />}
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
      
      {isMobile && <MobileNav onSignOut={signOut} />}
      <Toaster />
    </div>
  );
}

export default App;

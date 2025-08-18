import React from 'react';
import { CollabLab } from './components/CollabLab';
import { AuthProvider } from './lib/auth';
import { ToastProvider } from './components/ui/toast';

function App() {
  return (
    <AuthProvider>
      <ToastProvider>
        <main className="min-h-screen bg-background text-foreground">
          <CollabLab />
        </main>
      </ToastProvider>
    </AuthProvider>
  );
}

export default App;

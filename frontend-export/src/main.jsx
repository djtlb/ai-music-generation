
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from '@/App';
import '@/index.css';
import { AuthProvider } from '@/contexts/SupabaseAuthContext';
import { ProfileProvider } from '@/contexts/ProfileContext';
import { ModelProvider } from '@/contexts/ModelContext';
import { SubscriptionProvider } from '@/contexts/SubscriptionContext';
import { Toaster } from "@/components/ui/toaster";

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <AuthProvider>
        <SubscriptionProvider>
          <ProfileProvider>
            <ModelProvider>
              <App />
            </ModelProvider>
          </ProfileProvider>
        </SubscriptionProvider>
      </AuthProvider>
    </BrowserRouter>
  </React.StrictMode>
);

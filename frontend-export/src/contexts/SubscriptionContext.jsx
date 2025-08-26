
import React, { createContext, useState, useEffect, useContext, useMemo, useCallback } from 'react';
import { useAuth } from '@/contexts/SupabaseAuthContext';
import { supabase } from '@/lib/customSupabaseClient';

const SubscriptionContext = createContext(undefined);

export function SubscriptionProvider({ children }) {
  const { user, isLoading: authLoading, session } = useAuth();
  const [subscription, setSubscription] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchSubscription = useCallback(async () => {
    if (!user) {
      setSubscription(null);
      setLoading(false);
      return;
    }
  
    setLoading(true);
    try {
      const { data, error } = await supabase
        .from('subscriptions')
        .select('*, prices(*, products(*))')
        .in('status', ['trialing', 'active'])
        .eq('user_id', user.id)
        .maybeSingle();
  
      if (error) {
        console.error('Error fetching subscription:', error);
      } else {
        setSubscription(data);
      }
    } catch (e) {
      console.error('An unexpected error occurred fetching subscription:', e);
    } finally {
      setLoading(false);
    }
  }, [user]);
  
  useEffect(() => {
    if (!authLoading) {
      fetchSubscription();
    }
  }, [user, authLoading, fetchSubscription, session]);

  const value = useMemo(() => ({
    subscription,
    loading,
  }), [subscription, loading]);

  return (
    <SubscriptionContext.Provider value={value}>
      {children}
    </SubscriptionContext.Provider>
  );
}

export const useSubscription = () => {
  const context = useContext(SubscriptionContext);
  if (context === undefined) {
    throw new Error('useSubscription must be used within a SubscriptionProvider');
  }
  return context;
};

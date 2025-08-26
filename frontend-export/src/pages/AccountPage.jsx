
import React from 'react';
import { useAuth } from '@/contexts/SupabaseAuthContext';
import { useProfile } from '@/contexts/ProfileContext';
import { useSubscription } from '@/contexts/SubscriptionContext';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { useToast } from '@/components/ui/use-toast';

export function AccountPage() {
    const { user } = useAuth();
    const { profile, loading: profileLoading } = useProfile();
    const { subscription, loading: subscriptionLoading } = useSubscription();
    const { toast } = useToast();
    
    const loading = profileLoading || subscriptionLoading;

    const handleManageSubscription = () => {
        toast({
            title: '🚧 Coming Soon!',
            description: "Managing subscriptions will be enabled when you set up Stripe webhooks.",
        });
    };
    
    const PlanDetails = () => {
        if (subscriptionLoading) {
            return (
                <div className="flex items-center justify-center p-8">
                    <Loader2 className="h-8 w-8 animate-spin text-purple-400"/>
                </div>
            );
        }

        if (!subscription) {
            return (
                <div>
                    <p className="text-slate-400">You are on the Free plan.</p>
                    <p className="text-slate-500 mt-2">Upgrade to a paid plan to unlock more features.</p>
                </div>
            );
        }
        
        const price = subscription.prices;
        const product = price?.products;

        return (
            <div className="space-y-4">
                <div>
                    <h3 className="font-semibold text-white">Current Plan</h3>
                    <p className="text-2xl font-bold text-purple-400">{product?.name || 'Unknown Plan'}</p>
                </div>
                <div>
                    <h3 className="font-semibold text-white">Status</h3>
                    <p className="capitalize text-green-400">{subscription.status}</p>
                </div>
                <Button onClick={handleManageSubscription}>
                    Manage Subscription
                </Button>
            </div>
        );
    };

    if (loading && !profile) {
        return (
             <div className="min-h-screen w-full flex items-center justify-center bg-background">
                <Loader2 className="h-10 w-10 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="container mx-auto max-w-4xl py-8">
             <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <h1 className="text-4xl font-bold tracking-tighter text-white mb-8">
                    Your Account
                </h1>
            </motion.div>
             <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
                <Card className="bg-slate-950/30 border-slate-800">
                    <CardHeader>
                        <CardTitle>Account Details</CardTitle>
                        <CardDescription>This is the information associated with your account.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-2">
                        <div>
                            <p className="text-sm font-medium text-slate-400">Email</p>
                            <p className="text-lg text-slate-200">{user?.email}</p>
                        </div>
                         <div>
                            <p className="text-sm font-medium text-slate-400">Username</p>
                            <p className="text-lg text-slate-200">{profile?.username || 'User'}</p>
                        </div>
                    </CardContent>
                </Card>
             </motion.div>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
                <Card className="mt-8 bg-slate-950/30 border-slate-800">
                    <CardHeader>
                        <CardTitle>Subscription</CardTitle>
                        <CardDescription>View and manage your current subscription plan.</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <PlanDetails />
                    </CardContent>
                </Card>
            </motion.div>
        </div>
    );
}

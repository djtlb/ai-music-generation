import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { motion } from 'framer-motion';
import { Check, Zap, Loader2, Star } from 'lucide-react';
import { useToast } from '@/components/ui/use-toast';
import { loadStripe } from '@stripe/stripe-js';
import { useAuth } from '@/contexts/SupabaseAuthContext';
import { supabase } from '@/lib/customSupabaseClient';
import { Badge } from '@/components/ui/badge';
import { useSubscription } from '@/contexts/SubscriptionContext';

const STRIPE_PUBLISHABLE_KEY = import.meta.env.VITE_STRIPE_PUBLISHABLE_KEY;

export function PricingPage() {
    const { toast } = useToast();
    const { user } = useAuth();
    const { subscription, loading: subscriptionLoading } = useSubscription();
    const [loadingPriceId, setLoadingPriceId] = useState(null);
    const [products, setProducts] = useState([]);
    const [loadingProducts, setLoadingProducts] = useState(true);

    useEffect(() => {
        const fetchProducts = async () => {
            setLoadingProducts(true);
            const { data, error } = await supabase
                .from('products')
                .select('*, prices(*)')
                .eq('active', true)
                .eq('prices.active', true)
                .order('metadata->priority');
            
            if (error) {
                toast({ title: 'Error fetching plans', description: error.message, variant: 'destructive' });
            } else {
                const freeProduct = {
                    id: 'prod_free',
                    name: 'Free',
                    description: 'For getting started & having fun.',
                    metadata: { features: 'Generate up to 10 songs,Community access' }
                };
                const paidProducts = data.filter(p => p.id !== 'prod_free');
                setProducts([freeProduct, ...paidProducts]);
            }
            setLoadingProducts(false);
        };
        fetchProducts();
    }, [toast]);

    const handleCheckout = async (price) => {
        setLoadingPriceId(price.id);
        
        if (!STRIPE_PUBLISHABLE_KEY) {
            toast({
                title: '🚧 Stripe Not Configured',
                description: "This feature requires a Stripe Publishable Key. Please set VITE_STRIPE_PUBLISHABLE_KEY in your environment variables.",
                variant: 'destructive',
            });
            setLoadingPriceId(null);
            return;
        }

        try {
            const stripe = await loadStripe(STRIPE_PUBLISHABLE_KEY);
            if (!stripe) throw new Error('Stripe.js not loaded');
            
            const { error } = await stripe.redirectToCheckout({
                lineItems: [{ price: price.id, quantity: 1 }],
                mode: 'subscription',
                successUrl: `${window.location.origin}/library`,
                cancelUrl: `${window.location.origin}/pricing`,
                customerEmail: user?.email,
            });

            if (error) {
                toast({ title: 'Checkout Error', description: error.message, variant: 'destructive' });
            }
        } catch (error) {
             toast({ title: 'Error', description: 'Could not proceed to checkout.', variant: 'destructive' });
        }
        setLoadingPriceId(null);
    };

    if (loadingProducts || subscriptionLoading) {
        return (
            <div className="flex items-center justify-center h-full">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
        );
    }

    return (
        <div className="container mx-auto max-w-5xl py-8">
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center mb-16"
            >
                <h1 className="text-5xl md:text-6xl font-extrabold tracking-tighter bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent">
                    Choose Your Plan
                </h1>
                <p className="text-slate-400 mt-4 text-lg max-w-2xl mx-auto">
                    Unlock your creative potential. Start for free, upgrade when you're ready.
                </p>
            </motion.div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-stretch justify-center">
                {products.map((product, index) => {
                    const price = product.prices?.[0];
                    const isHighlighted = product.metadata?.highlighted === 'true';
                    const isCurrentPlan = (!subscription && product.id === 'prod_free') || (subscription && subscription.price_id === price?.id);

                    return (
                        <motion.div key={product.id} initial={{ opacity: 0, y: 50 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.1 }}>
                            <Card className={`relative h-full flex flex-col transition-all duration-300 ${isHighlighted ? 'bg-black/30 border-primary shadow-2xl shadow-primary/20' : 'bg-black/20 border-border'}`}>
                                <CardHeader className="text-center pt-8">
                                    {isHighlighted && (
                                        <Badge variant="secondary" className="mx-auto mb-4 bg-primary/20 text-primary border-primary/30">
                                            <Star className="w-4 h-4 mr-2" /> Most Popular
                                        </Badge>
                                    )}
                                    <CardTitle className="text-4xl font-bold">{product.name}</CardTitle>
                                    <CardDescription className="mt-2 text-lg text-slate-400">{product.description}</CardDescription>
                                </CardHeader>
                                <CardContent className="flex-grow flex flex-col justify-between">
                                    <div className="text-center my-8">
                                        {price ? (
                                            <>
                                                <span className="text-6xl font-extrabold text-white">${price.unit_amount / 100}</span>
                                                <span className="text-slate-400 text-lg">/ {price.interval}</span>
                                            </>
                                        ) : (
                                            <span className="text-6xl font-extrabold text-white">Free</span>
                                        )}
                                    </div>
                                    <ul className="space-y-4 text-lg">
                                        {product.metadata?.features?.split(',').map((feature) => (
                                            <li key={feature} className="flex items-start gap-3">
                                                <Check className="w-5 h-5 text-accent flex-shrink-0 mt-1.5"/>
                                                <span className="text-slate-300">{feature.trim()}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </CardContent>
                                <CardFooter className="p-6 mt-8">
                                    <Button
                                        onClick={() => price ? handleCheckout(price) : null}
                                        disabled={loadingPriceId === price?.id || isCurrentPlan}
                                        size="lg"
                                        className={`w-full text-lg font-bold py-7 transition-all duration-300 ${isHighlighted ? 'bg-primary hover:bg-primary/90 button-glow' : 'bg-white/10 hover:bg-white/20'}`}
                                    >
                                        {loadingPriceId === price?.id ? <Loader2 className="animate-spin" /> : (isCurrentPlan ? 'Your Current Plan' : (price ? 'Choose Plan' : 'Get Started'))}
                                    </Button>
                                </CardFooter>
                            </Card>
                        </motion.div>
                    )
                })}
            </div>
        </div>
    );
}
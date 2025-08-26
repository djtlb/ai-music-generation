/**
 * Stripe payment integration for AI Music Generation
 * This library handles interactions with our Stripe payment functions
 */

// Supabase functions base URL
const FUNCTIONS_URL = 'https://emecscbwfcvbkvxbztfa.supabase.co/functions/v1';

// Stripe public key
export const STRIPE_PUBLIC_KEY = 'pk_live_51RdlxXP2t49TLtZwLCBbfbDWXqro6nwA779ETkPT2NRT0TO4421Ot0g9ostCOor9OIop5m4JnXGdJKKrRtBpdZ2P00rUfHRja2';

// Supabase anon key for authorization
export const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWNzY2J3ZmN2Ymt2eGJ6dGZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUyOTc4NTYsImV4cCI6MjA3MDg3Mzg1Nn0.gKtvk5fiqjBToyy9w0_-DF4EmZdBQwUJCpY07XM7QJ0';

/**
 * Create a new customer in Stripe
 */
export async function createCustomer(email, name) {
  try {
    const response = await fetch(`${FUNCTIONS_URL}/create-customer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
      },
      body: JSON.stringify({ email, name })
    });
    
    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error || 'Failed to create customer');
    }
    
    return data;
  } catch (error) {
    console.error('Error creating customer:', error);
    throw error;
  }
}

/**
 * Create a subscription for a customer
 */
export async function createSubscription(customerId, priceId = 'price_1RzsuQP2t49TLtZwBGCR42bt', addressInfo = null) {
  try {
    const response = await fetch(`${FUNCTIONS_URL}/create-subscription`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
      },
      body: JSON.stringify({ 
        customerId, 
        priceId,
        addressInfo
      })
    });
    
    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error || 'Failed to create subscription');
    }
    
    return data;
  } catch (error) {
    console.error('Error creating subscription:', error);
    throw error;
  }
}

/**
 * Create a payment intent for one-time charges
 */
export async function createPayment(customerId, amount, currency = 'usd', description) {
  try {
    const response = await fetch(`${FUNCTIONS_URL}/create-payment`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
      },
      body: JSON.stringify({ customerId, amount, currency, description })
    });
    
    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error || 'Failed to create payment');
    }
    
    return data;
  } catch (error) {
    console.error('Error creating payment:', error);
    throw error;
  }
}

/**
 * Create a setup intent for attaching a payment method to a customer
 */
export async function createSetupIntent(customerId) {
  try {
    const response = await fetch(`${FUNCTIONS_URL}/create-setup-intent`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
      },
      body: JSON.stringify({ customerId })
    });
    
    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error || 'Failed to create setup intent');
    }
    
    return data;
  } catch (error) {
    console.error('Error creating setup intent:', error);
    throw error;
  }
}

/**
 * Initialize Stripe.js and return the Stripe object
 * Requires the Stripe.js script to be loaded:
 * <script src="https://js.stripe.com/v3/"></script>
 */
export function initStripe() {
  if (!window.Stripe) {
    throw new Error('Stripe.js is not loaded. Add <script src="https://js.stripe.com/v3/"></script> to your HTML.');
  }
  
  return window.Stripe(STRIPE_PUBLIC_KEY);
}

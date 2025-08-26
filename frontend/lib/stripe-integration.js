// Example of integrating Stripe with your music generation page

// Include this script in your HTML page
// <script src="https://js.stripe.com/v3/"></script>
// <script src="/path/to/this/file.js"></script>

// Initialize Stripe with your publishable key
const stripe = Stripe('pk_live_51RdlxXP2t49TLtZwLCBbfbDWXqro6nwA779ETkPT2NRT0TO4421Ot0g9ostCOor9OIop5m4JnXGdJKKrRtBpdZ2P00rUfHRja2');

// Supabase function URLs
const CREATE_CUSTOMER_URL = 'https://emecscbwfcvbkvxbztfa.supabase.co/functions/v1/create-customer';
const CREATE_SUBSCRIPTION_URL = 'https://emecscbwfcvbkvxbztfa.supabase.co/functions/v1/create-subscription';
const CREATE_PAYMENT_URL = 'https://emecscbwfcvbkvxbztfa.supabase.co/functions/v1/create-payment';
const CREATE_SETUP_INTENT_URL = 'https://emecscbwfcvbkvxbztfa.supabase.co/functions/v1/create-setup-intent';

// Supabase anon key for authorization
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWNzY2J3ZmN2Ymt2eGJ6dGZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUyOTc4NTYsImV4cCI6MjA3MDg3Mzg1Nn0.gKtvk5fiqjBToyy9w0_-DF4EmZdBQwUJCpY07XM7QJ0';

// Function to create a customer
async function createCustomer(email, name) {
  try {
    const response = await fetch(CREATE_CUSTOMER_URL, {
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

// Function to create a subscription
async function createSubscription(customerId, addressInfo) {
  try {
    const response = await fetch(CREATE_SUBSCRIPTION_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
      },
      body: JSON.stringify({ 
        customerId,
        priceId: 'price_1RzsuQP2t49TLtZwBGCR42bt', // Pro subscription price ID
        addressInfo: addressInfo
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

// Function to create a one-time payment
async function createPayment(customerId, amount) {
  try {
    const response = await fetch(CREATE_PAYMENT_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
      },
      body: JSON.stringify({ 
        customerId,
        amount,
        currency: 'usd',
        description: 'AI Music Generation'
      })
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

// Example function to handle subscription sign-up
async function handleSubscriptionSignup(event) {
  event.preventDefault();
  
  const submitButton = document.getElementById('subscribe-button');
  const statusElement = document.getElementById('subscription-status');
  
  submitButton.disabled = true;
  submitButton.textContent = 'Processing...';
  
  try {
    // Get user details from form
    const email = document.getElementById('email').value;
    const name = document.getElementById('name').value;
    
    // Get address information
    const addressInfo = {
      line1: document.getElementById('address-line1').value,
      line2: document.getElementById('address-line2')?.value || '',
      city: document.getElementById('city').value,
      state: document.getElementById('state').value,
      postal_code: document.getElementById('postal-code').value,
      country: document.getElementById('country').value
    };
    
    // 1. Create customer
    statusElement.textContent = 'Creating customer...';
    const customerData = await createCustomer(email, name);
    
    // 2. Create subscription
    statusElement.textContent = 'Setting up subscription...';
    const subscriptionData = await createSubscription(customerData.customerId, addressInfo);
    
    // 3. Success
    statusElement.textContent = 'Subscription active! You now have access to all Pro features.';
    statusElement.className = 'success';
    
    // 4. Store subscription details in localStorage or your backend
    localStorage.setItem('subscription', JSON.stringify({
      customerId: customerData.customerId,
      subscriptionId: subscriptionData.subscriptionId,
      status: subscriptionData.status,
      expiresAt: subscriptionData.currentPeriodEnd
    }));
    
    // 5. Redirect to the app or refresh the page
    setTimeout(() => {
      window.location.href = '/app.html?upgraded=true';
    }, 2000);
    
  } catch (error) {
    statusElement.textContent = `Error: ${error.message}`;
    statusElement.className = 'error';
    console.error('Subscription error:', error);
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = 'Subscribe';
  }
}

// Example function to check if user has an active subscription
function checkSubscriptionStatus() {
  const subscriptionData = localStorage.getItem('subscription');
  
  if (!subscriptionData) {
    return false;
  }
  
  const subscription = JSON.parse(subscriptionData);
  const now = Math.floor(Date.now() / 1000);
  
  // Check if subscription is active and not expired
  return subscription.status === 'active' && subscription.expiresAt > now;
}

// Example function to handle one-time payment for a song
async function handleSongPurchase(songId, price) {
  const statusElement = document.getElementById('payment-status');
  
  try {
    // Get customer ID from localStorage or create a new customer
    let customerId;
    const subscriptionData = localStorage.getItem('subscription');
    
    if (subscriptionData) {
      customerId = JSON.parse(subscriptionData).customerId;
    } else {
      // Get user details from form or prompt
      const email = prompt('Enter your email to continue:');
      if (!email) return;
      
      const customerData = await createCustomer(email);
      customerId = customerData.customerId;
    }
    
    // Create payment intent
    statusElement.textContent = 'Processing payment...';
    const paymentData = await createPayment(customerId, price);
    
    // Handle the payment on the client
    const { error } = await stripe.confirmCardPayment(paymentData.clientSecret, {
      payment_method: {
        card: elements.getElement('card'),
        billing_details: {
          name: document.getElementById('name').value,
        },
      }
    });
    
    if (error) {
      throw new Error(error.message);
    } else {
      statusElement.textContent = 'Payment successful! Your song is ready to download.';
      statusElement.className = 'success';
      
      // Enable download button or redirect to download page
      document.getElementById('download-button').disabled = false;
    }
    
  } catch (error) {
    statusElement.textContent = `Payment error: ${error.message}`;
    statusElement.className = 'error';
    console.error('Payment error:', error);
  }
}

// Example of how to use these functions in your HTML
document.addEventListener('DOMContentLoaded', () => {
  // For subscription page
  const subscribeForm = document.getElementById('subscribe-form');
  if (subscribeForm) {
    subscribeForm.addEventListener('submit', handleSubscriptionSignup);
  }
  
  // For song purchase
  const purchaseButtons = document.querySelectorAll('.purchase-button');
  if (purchaseButtons.length > 0) {
    purchaseButtons.forEach(button => {
      button.addEventListener('click', () => {
        const songId = button.dataset.songId;
        const price = parseInt(button.dataset.price, 10);
        handleSongPurchase(songId, price);
      });
    });
  }
  
  // Check subscription status and update UI
  const isSubscribed = checkSubscriptionStatus();
  if (isSubscribed) {
    // Show pro features
    document.querySelectorAll('.pro-feature').forEach(el => {
      el.style.display = 'block';
    });
    
    // Hide subscribe buttons/prompts
    document.querySelectorAll('.subscribe-prompt').forEach(el => {
      el.style.display = 'none';
    });
  }
});

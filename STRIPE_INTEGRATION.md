# Stripe Integration Guide

This document provides instructions for integrating Stripe payments into the AI Music Generation application.

## Overview

The integration supports three main payment operations:

1. Creating Stripe customers
2. Setting up subscriptions for the Pro tier
3. Processing one-time payments for individual songs

## API Endpoints

All payment functionality is implemented as Supabase Edge Functions:

1. `create-customer`: Creates a new Stripe customer
2. `create-subscription`: Sets up a recurring subscription for a customer
3. `create-payment`: Creates a one-time payment intent
4. `create-setup-intent`: Creates a setup intent for attaching a payment method to a customer

## Client-Side Integration

### Installation

Include the Stripe.js script in your HTML:

```html
<script src="https://js.stripe.com/v3/"></script>
```

### Basic Usage

```javascript
import { 
  createCustomer, 
  createSubscription, 
  createPayment, 
  initStripe 
} from '/frontend/lib/stripe.js';

// Initialize Stripe with your public key
const stripe = initStripe();

// Create a customer
const customerData = await createCustomer('user@example.com', 'User Name');
const customerId = customerData.customerId;

// Create a subscription (requires attached payment method)
// Note: Address information is required for tax calculation
const addressInfo = {
  line1: '123 Main St',
  city: 'San Francisco',
  state: 'CA',
  postal_code: '94111',
  country: 'US'
};
const subscriptionData = await createSubscription(customerId, 'price_1RzsuQP2t49TLtZwBGCR42bt', addressInfo);

// Create a one-time payment
const paymentData = await createPayment(customerId, 1999, 'usd', 'AI Music Generation');

// Confirm the payment on the client side
const { error } = await stripe.confirmCardPayment(paymentData.clientSecret, {
  payment_method: {
    card: elements.getElement('card'),
    billing_details: {
      name: 'User Name',
    },
  }
});
```

## Setting Up a Complete Subscription Flow

For subscriptions to work properly, you need to:

1. Create a customer with `createCustomer()`
2. Collect payment method information using Stripe Elements
3. Attach the payment method to the customer using `stripe.setupIntent()`
4. Create a subscription with `createSubscription()` including address information

### Example Subscription Flow

```javascript
// 1. Create customer
const customerData = await createCustomer(email, name);
const customerId = customerData.customerId;

// 2. Create setup intent for attaching a payment method
const setupIntentData = await createSetupIntent(customerId);

// 3. Confirm card setup with Stripe Elements
const result = await stripe.confirmCardSetup(setupIntentData.clientSecret, {
  payment_method: {
    card: cardElement,
    billing_details: {
      name: name,
      email: email,
      address: {
        line1: addressLine1,
        city: city,
        state: state,
        postal_code: postalCode,
        country: country
      }
    }
  }
});

if (result.error) {
  // Handle error
} else {
  // Payment method attached successfully
  
  // 4. Create subscription with address information
  const addressInfo = {
    line1: addressLine1,
    city: city,
    state: state,
    postal_code: postalCode,
    country: country
  };
  
  const subscriptionData = await createSubscription(customerId, undefined, addressInfo);
  
  // 5. Handle successful subscription
  console.log('Subscription active!', subscriptionData);
}
```

## Testing

You can test the Stripe integration using the provided test script:

```bash
bash test_stripe.sh
```

The script tests all three endpoints: create-customer, create-subscription, and create-payment.

## Important Notes

1. **Address Information**: Stripe requires valid address information for tax calculation on subscriptions.
2. **Payment Methods**: Customers must have a payment method attached before creating a subscription.
3. **Authorization**: All requests to Supabase functions require proper authorization headers.
4. **Error Handling**: Always implement proper error handling for payment operations.

## Configuration

The integration uses the following configuration:

- **Stripe Public Key**: `pk_live_51RdlxXP2t49TLtZwLCBbfbDWXqro6nwA779ETkPT2NRT0TO4421Ot0g9ostCOor9OIop5m4JnXGdJKKrRtBpdZ2P00rUfHRja2`
- **Stripe Secret Key**: Stored in Supabase secrets
- **Pro Subscription Price ID**: `price_1RzsuQP2t49TLtZwBGCR42bt`

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { corsHeaders } from '../_shared/cors.ts'

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Get the Stripe secret key from the Supabase secrets
    const stripeSecretKey = Deno.env.get('STRIPE_SECRET_KEY')
    if (!stripeSecretKey) {
      throw new Error('STRIPE_SECRET_KEY is not set in Supabase secrets.')
    }

    // Parse request body
    const { customerId, priceId = 'price_1RzsuQP2t49TLtZwBGCR42bt', addressInfo } = await req.json()
    
    if (!customerId) {
      throw new Error('Customer ID is required')
    }

    console.log(`Creating subscription for customer: ${customerId} with price: ${priceId}`)

    // First, update the customer with address information if provided
    if (addressInfo) {
      console.log(`Updating customer ${customerId} with address information`)
      
      const customerUpdateParams = new URLSearchParams();
      
      if (addressInfo.line1) customerUpdateParams.append('address[line1]', addressInfo.line1);
      if (addressInfo.line2) customerUpdateParams.append('address[line2]', addressInfo.line2);
      if (addressInfo.city) customerUpdateParams.append('address[city]', addressInfo.city);
      if (addressInfo.state) customerUpdateParams.append('address[state]', addressInfo.state);
      if (addressInfo.postal_code) customerUpdateParams.append('address[postal_code]', addressInfo.postal_code);
      if (addressInfo.country) customerUpdateParams.append('address[country]', addressInfo.country);
      
      // Set tax exempt status if provided
      if (addressInfo.tax_exempt) customerUpdateParams.append('tax_exempt', addressInfo.tax_exempt);
      
      const customerResponse = await fetch(`https://api.stripe.com/v1/customers/${customerId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'Authorization': `Bearer ${stripeSecretKey}`
        },
        body: customerUpdateParams
      });
      
      if (!customerResponse.ok) {
        const errorData = await customerResponse.json();
        console.error('Error updating customer with address:', errorData);
        throw new Error(`Failed to update customer: ${errorData.error?.message}`);
      }
      
      console.log(`Customer ${customerId} updated successfully with address`);
    }

    // Make the request to Stripe API
    const response = await fetch('https://api.stripe.com/v1/subscriptions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': `Bearer ${stripeSecretKey}`
      },
      body: new URLSearchParams({
        'automatic_tax[enabled]': 'true',
        'currency': 'usd',
        'customer': customerId,
        'items[0][price]': priceId,
        'items[0][quantity]': '1',
        'off_session': 'true',
        'payment_behavior': 'error_if_incomplete',
        'proration_behavior': 'none'
      })
    })

    // Parse Stripe response
    const responseData = await response.json()
    
    if (!response.ok) {
      console.error('Stripe API error:', responseData)
      return new Response(JSON.stringify({
        success: false,
        error: responseData.error?.message || 'Failed to create subscription',
        details: responseData.error
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 400
      })
    }

    // Return the successful subscription data
    return new Response(JSON.stringify({
      success: true,
      subscriptionId: responseData.id,
      status: responseData.status,
      currentPeriodEnd: responseData.current_period_end,
      customerId: responseData.customer
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200
    })
    
  } catch (error) {
    console.error('Function error:', error)
    return new Response(JSON.stringify({ 
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 500
    })
  }
})

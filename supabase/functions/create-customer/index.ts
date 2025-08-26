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
    const { email, name } = await req.json()
    
    if (!email) {
      throw new Error('Email is required')
    }

    console.log(`Creating customer for email: ${email}`)

    // Make the request to Stripe API
    const response = await fetch('https://api.stripe.com/v1/customers', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': `Bearer ${stripeSecretKey}`
      },
      body: new URLSearchParams({
        'email': email,
        'name': name || email.split('@')[0],
        'metadata[source]': 'ai_music_app'
      })
    })

    // Parse Stripe response
    const responseData = await response.json()
    
    if (!response.ok) {
      console.error('Stripe API error:', responseData)
      return new Response(JSON.stringify({
        success: false,
        error: responseData.error?.message || 'Failed to create customer',
        details: responseData.error
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 400
      })
    }

    // Return the successful customer data
    return new Response(JSON.stringify({
      success: true,
      customerId: responseData.id,
      email: responseData.email,
      name: responseData.name
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

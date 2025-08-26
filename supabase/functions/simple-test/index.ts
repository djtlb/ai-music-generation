import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'

// Simple CORS headers without the need for a shared file
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, GET, OPTIONS'
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // This is a simple test function that doesn't need authentication
    // It just checks if we can reach the backend
    return new Response(JSON.stringify({
      success: true,
      message: 'Test function is working!',
      timestamp: new Date().toISOString(),
      testInfo: 'This is a simple test that confirms the Edge Function is running correctly.'
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    })
    
  } catch (error) {
    return new Response(JSON.stringify({ 
      success: false, 
      error: error.message 
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    })
  }
})

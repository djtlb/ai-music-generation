/// <reference types="https://esm.sh/@supabase/functions-js/src/edge-runtime.d.ts" />

// Follow this setup guide to integrate the Deno language server with your editor:
// https://deno.land/manual/getting_started/setup_your_environment
// This enables autocomplete, go to definition, etc.

// Setup type definitions for built-in Supabase Runtime APIs
import "jsr:@supabase/functions-js/edge-runtime.d.ts"

// Define CORS headers to allow cross-origin requests
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, GET, OPTIONS'
}

console.log("Final test function called!")

Deno.serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Try to connect to the backend
    const backendUrl = Deno.env.get('BACKEND_URL') || 'http://localhost:8000';
    const backendResponse = await fetch(`${backendUrl}/healthz`)
    const backendStatus = backendResponse.ok ? 'reachable' : 'unreachable'
    const backendText = await backendResponse.text()
    
    // Return a success response with test data and backend connectivity info
    return new Response(
      JSON.stringify({
        success: true,
        message: 'Final test function working!',
        timestamp: new Date().toISOString(),
        test_project_id: 'test-123',
        audio_url: 'https://example.com/test.mp3',
        backend_status: backendStatus,
        backend_response: backendText
      }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200
      }
    )
  } catch (error) {
    // Return an error response
    return new Response(
      JSON.stringify({ 
        success: false, 
        error: error.message,
        timestamp: new Date().toISOString()
      }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500
      }
    )
  }
})

/// <reference types="https://esm.sh/@supabase/functions-js/src/edge-runtime.d.ts" />
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { corsHeaders } from '../_shared/cors.ts'

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Simply check if we can reach the backend
    const BACKEND_URL = Deno.env.get('BACKEND_URL') || 'http://localhost:8000';
    const response = await fetch(`${BACKEND_URL}/healthz`, {
      method: 'GET'
    })
    
    if (!response.ok) {
      return new Response(JSON.stringify({
        success: false,
        message: 'Cannot connect to backend',
        status: response.status,
        statusText: response.statusText
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      })
    }
    
    const data = await response.text()
    
    return new Response(JSON.stringify({
      success: true,
      message: 'Backend connection successful',
      response: data
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

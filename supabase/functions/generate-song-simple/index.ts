// @deno-types
import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'

// Simple CORS headers
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, GET, OPTIONS'
}

// Helper function to handle timeouts
const withTimeout = (promise, timeoutMs) => {
  let timeoutHandle;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutHandle = setTimeout(() => {
      reject(new Error(`Operation timed out after ${timeoutMs} ms`));
    }, timeoutMs);
  });

  return Promise.race([
    promise,
    timeoutPromise
  ]).finally(() => {
    clearTimeout(timeoutHandle);
  });
};

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Get the backend API key from the Supabase secrets
    const backendApiKey = Deno.env.get('BACKEND_API_KEY')
    if (!backendApiKey) {
      throw new Error('BACKEND_API_KEY is not set in Supabase secrets.')
    }

    // Get the user's prompt from the request
    let prompt = "Default prompt"
    let userId = "website-user"
    
    try {
      const body = await req.json()
      prompt = body.prompt || prompt
      if (body.user_id) {
        userId = body.user_id
      }
    } catch (e) {
      console.log("Error parsing request body:", e)
    }

    console.log(`Starting music generation for prompt: "${prompt.substring(0, 30)}..." and user: ${userId}`)

    try {
      // Get the backend URL from environment or use the hardcoded one
      const backendUrl = Deno.env.get('BACKEND_URL') || 'http://localhost:8000'
      console.log(`Using backend URL: ${backendUrl}`)

      // Call the backend with a 30-second timeout to start generation
      const backendResponse = await withTimeout(
        fetch(`${backendUrl}/api/v1/generate/full-song`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-api-key': backendApiKey,
          },
          body: JSON.stringify({ 
            prompt: prompt, 
            user_id: userId, 
            style_config: { 
              style: prompt.toLowerCase().includes('rock') ? 'rock_punk' : 
                     prompt.toLowerCase().includes('pop') ? 'country_pop' : 
                     prompt.toLowerCase().includes('jazz') ? 'jazz' : 
                     prompt.toLowerCase().includes('hiphop') || prompt.toLowerCase().includes('hip hop') ? 'hiphop' : 
                     prompt.toLowerCase().includes('ballad') || prompt.toLowerCase().includes('rnb') ? 'rnb_ballad' : 
                     'country_pop',
              duration: prompt.toLowerCase().includes('short') ? 60 : 120,
              generate_real_audio: true,
              lyrics_text: prompt
            },
            advanced_options: {
              render_stems: true,
              apply_mastering: true,
              use_high_quality: true
            }
          }),
        }),
        30000 // 30 second timeout for real audio generation
      )

      if (!backendResponse.ok) {
        const errorText = await backendResponse.text()
        throw new Error(`Backend error: ${backendResponse.status} - ${errorText}`)
      }

      const startData = await backendResponse.json()
      console.log('Backend response received:', JSON.stringify(startData))
      
      // If we get a project_id, return it immediately without polling
      if (startData.project_id) {
        const projectId = startData.project_id
        
        return new Response(JSON.stringify({
          success: true,
          project_id: projectId,
          message: 'Song generation started successfully',
          status: 'processing',
          estimated_completion: '1-2 minutes',
          audio_url: startData.audio_url || null
        }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 200,
        })
      } else {
        // If no project_id, return the response directly
        return new Response(JSON.stringify({
          ...startData,
          success: true,
          message: 'Request processed but no project_id was returned'
        }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 200,
        })
      }
    } catch (error) {
      // If there's an error connecting to the backend, return a more helpful error
      console.error('Backend connection error:', error)
      
      if (error.message.includes('timed out')) {
        return new Response(JSON.stringify({
          success: false,
          error: 'Backend connection timed out. The server might be busy or unreachable.',
          message: 'Please try again in a few minutes.',
          timestamp: new Date().toISOString()
        }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 200, // Return 200 even for errors to make debugging easier
        })
      } else {
        return new Response(JSON.stringify({
          success: false,
          error: `Backend connection error: ${error.message}`,
          message: 'There was a problem connecting to the AI music generation service.',
          timestamp: new Date().toISOString()
        }), {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 200,
        })
      }
    }
  } catch (error) {
    // Return an error response for any other errors
    console.error('Function error:', error)
    return new Response(JSON.stringify({ 
      success: false,
      error: error.message,
      timestamp: new Date().toISOString()
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200, // Return 200 to make debugging easier
    })
  }
})

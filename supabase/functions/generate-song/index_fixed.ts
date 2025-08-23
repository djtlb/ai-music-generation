import { serve } from "https://deno.land/std@0.208.0/http/server.ts"

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS'
}

// Simple interface for the request
interface GenerateRequest {
  prompt: string
  genre?: string
  duration?: number
  user_id?: string
}

// Function to call your VPS backend
async function generateMusic(prompt: string, genre: string, duration: number): Promise<{
  success: boolean
  project_id?: string
  message?: string
  error?: string
}> {
  try {
    // Call your VPS FastAPI backend
    const FASTAPI_URL = 'http://168.231.67.14:8000/generate'
    
    const response = await fetch(FASTAPI_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt,
        genre,
        duration
      })
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('FastAPI backend error:', errorText)
      return {
        success: false,
        error: `FastAPI backend error: ${response.status} ${response.statusText}`
      }
    }

    const result = await response.json()
    
    // Handle the response from your FastAPI backend
    if (result.project_id || result.status === 'completed') {
      return {
        success: true,
        project_id: result.project_id,
        message: result.message || 'Music generated successfully!'
      }
    } else {
      return {
        success: false,
        error: result.error || 'Unknown error from FastAPI backend'
      }
    }
    
  } catch (error) {
    console.error('Music generation error:', error)
    return {
      success: false,
      error: error instanceof Error ? error.message : String(error)
    }
  }
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders })
  }

  if (req.method !== 'POST') {
    return new Response(
      JSON.stringify({ error: 'Method not allowed. Use POST.' }), 
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 405 }
    )
  }

  try {
    // Parse request body
    const { prompt, genre = 'electronic', duration = 30, user_id }: GenerateRequest = await req.json()

    if (!prompt) {
      return new Response(
        JSON.stringify({ error: 'Missing required field: prompt' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }
    
    console.log(`Generating music: ${prompt} (genre: ${genre}, duration: ${duration}s)`)
    
    // Generate music using your VPS backend
    const result = await generateMusic(prompt, genre, duration)
    
    if (!result.success) {
      return new Response(
        JSON.stringify({ error: result.error }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Return success response
    return new Response(
      JSON.stringify({
        success: true,
        project_id: result.project_id,
        message: result.message
      }),
      { status: 200, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )

  } catch (error) {
    console.error('Edge Function error:', error)
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
    )
  }
})

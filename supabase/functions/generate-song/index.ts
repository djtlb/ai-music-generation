/// <reference types="https://esm.sh/@supabase/functions-js/src/edge-runtime.d.ts" />

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { corsHeaders } from '../_shared/cors.ts' // Import our new shared headers

serve(async (req) => {
  // This block is new! It handles the browser's permission check.
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Get the backend API key from the Supabase secrets we set earlier.
    const backendApiKey = Deno.env.get('BACKEND_API_KEY')
    if (!backendApiKey) {
      throw new Error('BACKEND_API_KEY is not set in Supabase secrets.')
    }

    // Get the user's prompt from the request.
    const { prompt } = await req.json()
    if (!prompt) {
      throw new Error('Prompt is required')
    }

    // Call your actual backend on the VPS to start generation.
    const FASTAPI_URL = (Deno.env.get('BACKEND_URL') || 'http://localhost:8000') + '/api/v1/generate/full-song'
    const backendResponse = await fetch(FASTAPI_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': backendApiKey, // Use the secret key to authenticate with your backend.
      },
      body: JSON.stringify({ prompt: prompt, user_id: 'website-user' }), // Send the prompt and a user ID.
    })

    if (!backendResponse.ok) {
      const errorBody = await backendResponse.text()
      throw new Error(`Backend error: ${errorBody}`)
    }

    const startData = await backendResponse.json()
    
    // If we get a project_id, poll for completion
    if (startData.project_id) {
      const projectId = startData.project_id
      let attempts = 0
      const maxAttempts = 60 // Wait up to 5 minutes (60 attempts * 5 seconds)
      
      while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 5000)) // Wait 5 seconds
        
        // Check project status
        const status_url = (Deno.env.get('BACKEND_URL') || 'http://localhost:8000') + `/api/v1/project/${projectId}/status`
        const statusResponse = await fetch(status_url, {
          method: 'GET',
          headers: {
            'x-api-key': backendApiKey,
          },
        })
        
        if (statusResponse.ok) {
          const statusData = await statusResponse.json()
          
          if (statusData.status === 'completed') {
            // Return the completed song - check different possible audio fields
            const stages = statusData.stages || {}
            const mixMasterStage = stages.mix_master || {}
            const audioUrl = statusData.audio_url || mixMasterStage.final_audio_url || statusData.result_url
            
            return new Response(JSON.stringify({
              success: true,
              audio_url: audioUrl ? (Deno.env.get('BACKEND_URL') || 'http://localhost:8000') + audioUrl : null,
              project_id: projectId,
              message: 'Song generation completed',
              status: statusData
            }), {
              headers: { ...corsHeaders, 'Content-Type': 'application/json' },
              status: 200,
            })
          } else if (statusData.status === 'failed' || statusData.status === 'error') {
            throw new Error('Song generation failed: ' + (statusData.error || 'Unknown error'))
          }
          // If still processing, continue polling
        }
        
        attempts++
      }
      
      // If we reach here, it timed out
      throw new Error('Song generation timed out')
    }

    // Fallback: return the original response if no project_id
    return new Response(JSON.stringify(startData), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    })
  } catch (error) {
    // Return an error response to the browser, including the CORS headers.
    return new Response(JSON.stringify({ error: error.message }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 500,
    })
  }
})

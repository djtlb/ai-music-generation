import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { corsHeaders } from '../_shared/cors.ts'

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

    // Get the project ID from the request
    const { project_id } = await req.json()
    if (!project_id) {
      throw new Error('Project ID is required')
    }

    // Get the backend URL from environment or use the hardcoded one
    const backendUrl = Deno.env.get('BACKEND_URL') || 'http://localhost:8000'
    console.log(`Using backend URL: ${backendUrl}`)
    
    // Call the backend API to check project status
    const statusResponse = await fetch(`${backendUrl}/api/v1/project/${project_id}/status`, {
      method: 'GET',
      headers: {
        'x-api-key': backendApiKey,
      },
    })
    
    if (!statusResponse.ok) {
      const errorBody = await statusResponse.text()
      throw new Error(`Backend error: ${errorBody}`)
    }
    
    const statusData = await statusResponse.json()
    
    // If the status is completed, include the audio URL
    if (statusData.status === 'completed') {
      // Get the audio URL from the appropriate field
      const stages = statusData.stages || {}
      const mixMasterStage = stages.mix_master || {}
      let audioUrl = statusData.audio_url || mixMasterStage.final_audio_url || statusData.result_url
      
      // Ensure the URL is properly formatted (absolute URL)
      if (audioUrl) {
        // If the URL doesn't start with http or https, make it absolute
        if (!audioUrl.startsWith('http://') && !audioUrl.startsWith('https://')) {
          // If it starts with a slash, it's a path on the backend
          if (audioUrl.startsWith('/')) {
            audioUrl = `${backendUrl}${audioUrl}`
          } else {
            audioUrl = `${backendUrl}/${audioUrl}`
          }
        }
      }
      
      return new Response(JSON.stringify({
        success: true,
        status: 'completed',
        audio_url: audioUrl,
        project_id: project_id,
        message: 'Song generation completed',
        progress: 100,
        details: statusData // Include full details for debugging
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      })
    }
    
    // If the status is failed or error, return the error
    if (statusData.status === 'failed' || statusData.status === 'error') {
      return new Response(JSON.stringify({
        success: false,
        status: statusData.status,
        error: statusData.error || 'Unknown error',
        project_id: project_id
      }), {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200,
      })
    }
    
    // If still processing, return the progress
    return new Response(JSON.stringify({
      success: true,
      status: 'processing',
      project_id: project_id,
      progress: statusData.progress || 0,
      message: 'Song generation in progress',
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 200,
    })
    
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      status: 500,
    })
  }
})

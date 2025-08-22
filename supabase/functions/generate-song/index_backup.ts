import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

interface GenerateRequest {
  prompt: string
  user_id: string
  project_id?: string
  genre?: string
  duration?: number
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Initialize Supabase client
    const supabaseUrl = Deno.env.get('SUPABASE_URL')!
    const supabaseAnonKey = Deno.env.get('SUPABASE_ANON_KEY')!
    const supabase = createClient(supabaseUrl, supabaseAnonKey)

    // Parse request body
    const { prompt, user_id, project_id, genre = 'electronic', duration = 30 }: GenerateRequest = await req.json()

    if (!prompt) {
      return new Response(
        JSON.stringify({ error: 'Missing required field: prompt' }),
        { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }
    
    // Use provided user_id or default to 'anonymous'
    const effectiveUserId = user_id || 'anonymous'

    // Create generation request record
    const { data: generation, error: genError } = await supabase
      .rpc('create_generation_request', {
        user_id_in: effectiveUserId,
        project_id_in: project_id,
        prompt_in: prompt
      })

    if (genError) {
      console.error('Error creating generation request:', genError)
      return new Response(
        JSON.stringify({ error: 'Failed to create generation request' }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Update status to processing
    await supabase.rpc('update_generation_status', {
      generation_id_in: generation,
      status_in: 'processing'
    })

    // Simulate music generation (replace with actual AI model call)
    const generatedMusic = await generateMusic(prompt, genre, duration)

    if (!generatedMusic.success) {
      // Update status to failed
      await supabase.rpc('update_generation_status', {
        generation_id_in: generation,
        status_in: 'failed'
      })

      return new Response(
        JSON.stringify({ error: 'Music generation failed', details: generatedMusic.error }),
        { status: 500, headers: { ...corsHeaders, 'Content-Type': 'application/json' } }
      )
    }

    // Create audio file record
    const { data: audioFile, error: audioError } = await supabase
      .from('audio_files')
      .insert({
        project_id: project_id,
        file_path: generatedMusic.filePath,
        file_name: generatedMusic.fileName,
        duration_seconds: duration,
        file_size_bytes: generatedMusic.fileSize
      })
      .select()
      .single()

    if (audioError) {
      console.error('Error creating audio file record:', audioError)
    }

    // Update generation status to completed
    await supabase.rpc('update_generation_status', {
      generation_id_in: generation,
      status_in: 'completed',
      generated_audio_id_in: audioFile?.id
    })

    return new Response(
      JSON.stringify({
        success: true,
        generation_id: generation,
        audio_file: audioFile,
        audio_url: generatedMusic.audioUrl,
        message: 'Music generated successfully'
      }),
      { 
        status: 200, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    )

  } catch (error) {
    console.error('Error in generate-song function:', error)
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error', 
        details: error instanceof Error ? error.message : String(error) 
      }),
      { 
        status: 500, 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      }
    )
  }
})

async function generateMusic(prompt: string, genre: string, duration: number): Promise<{
  success: boolean
  filePath?: string
  fileName?: string
  fileSize?: number
  audioUrl?: string
  error?: string
}> {
  try {
    // Call your VPS FastAPI backend
    const FASTAPI_URL = 'http://168.231.67.14:8000/generate' // Updated to match your backend endpoint
    
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
    if (result.success || result.audio_url) {
      return {
        success: true,
        filePath: result.file_path || `/generated_audio/${Date.now()}.wav`,
        fileName: result.file_name || `generated_${Date.now()}.wav`,
        fileSize: result.file_size || duration * 44100 * 2,
        audioUrl: result.audio_url || result.url
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

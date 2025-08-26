// Test script for the music generation endpoint via Supabase Edge Function
import { createClient } from '@supabase/supabase-js';

// Replace with your Supabase project URL and anon key
const SUPABASE_URL = 'https://emecscbwfcvbkvxbztfa.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWNzY2J3ZmN2Ymt2eGJ6dGZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTI4MjI4MTYsImV4cCI6MjAwODM5ODgxNn0.TzRWJh1v53LQOIKtbhGUHj6dXwZGnA1XFEKZWQQT7H8';

// Initialize Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

async function testMusicGeneration() {
  console.log('🎵 Testing music generation via Supabase Edge Function...');
  console.log('Sending request to generate-song function...');
  
  try {
    // Call the Edge Function
    const { data, error } = await supabase.functions.invoke('generate-song', {
      body: {
        prompt: 'Create an upbeat pop song with catchy melody and a strong chorus'
      }
    });

    if (error) {
      console.error('❌ Error calling function:', error);
      return;
    }

    console.log('✅ Function response received!');
    console.log('Response data:', JSON.stringify(data, null, 2));
    
    if (data.project_id) {
      console.log(`📋 Project ID: ${data.project_id}`);
      console.log('⏳ This project will be processed in the background.');
      
      if (data.audio_url) {
        console.log(`🎧 Audio URL: ${data.audio_url}`);
        console.log('The music has been generated successfully!');
      } else {
        console.log('🔍 No audio URL received yet. You may need to poll for status.');
      }
    }
    
  } catch (err) {
    console.error('❌ Unexpected error:', err);
  }
}

// Execute the test
testMusicGeneration();

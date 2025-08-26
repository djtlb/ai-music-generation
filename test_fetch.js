// Test script for the music generation endpoint via Supabase Edge Function
// This script uses fetch API to test the function directly

const SUPABASE_URL = 'https://emecscbwfcvbkvxbztfa.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWNzY2J3ZmN2Ymt2eGJ6dGZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTI4MjI4MTYsImV4cCI6MjAwODM5ODgxNn0.TzRWJh1v53LQOIKtbhGUHj6dXwZGnA1XFEKZWQQT7H8';

async function testMusicGeneration() {
  console.log('🎵 Testing music generation via Supabase Edge Function...');
  console.log('Sending request to generate-song function...');
  
  try {
    // Call the Edge Function
    const response = await fetch(`${SUPABASE_URL}/functions/v1/generate-song`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${SUPABASE_ANON_KEY}`
      },
      body: JSON.stringify({
        prompt: 'Create an upbeat pop song with catchy melody and a strong chorus'
      })
    });

    if (!response.ok) {
      console.error(`❌ Error: ${response.status}`);
      console.error(await response.text());
      return;
    }

    const data = await response.json();
    console.log('✅ Function response received!');
    console.log('Response data:', JSON.stringify(data, null, 2));
    
    if (data.project_id) {
      console.log(`📋 Project ID: ${data.project_id}`);
      console.log('⏳ This project will be processed in the background.');
      
      if (data.audio_url) {
        console.log(`🎧 Audio URL: ${data.audio_url}`);
        console.log('The music has been generated successfully!');
      } else {
        console.log('🔍 No audio URL received yet. The generation is still in progress.');
        console.log('You would need to poll for status with the project-status function.');
      }
    }
    
  } catch (err) {
    console.error('❌ Unexpected error:', err);
  }
}

// Execute the test
testMusicGeneration();

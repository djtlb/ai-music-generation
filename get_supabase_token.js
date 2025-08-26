// Script to get a valid Supabase JWT token
import { createClient } from '@supabase/supabase-js'

// These are the credentials we used previously
const supabaseUrl = 'https://emecscbwfcvbkvxbztfa.supabase.co'
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWNzY2J3ZmN2Ymt2eGJ6dGZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTI4MjI4MTYsImV4cCI6MjAwODM5ODgxNn0.TzRWJh1v53LQOIKtbhGUHj6dXwZGnA1XFEKZWQQT7H8'

const supabase = createClient(supabaseUrl, supabaseKey)

// Get the current session
async function getSession() {
  const { data, error } = await supabase.auth.getSession()
  
  if (error) {
    console.error('Error getting session:', error.message)
    return null
  }
  
  if (!data.session) {
    console.log('No active session. Let\'s create an anonymous session.')
    // If no session, you can either create an anonymous session or sign in
    // For this example, we'll try to create an anonymous session
    const { data: signInData, error: signInError } = await supabase.auth.signInAnonymously()
    
    if (signInError) {
      console.error('Error signing in anonymously:', signInError.message)
      return null
    }
    
    console.log('Anonymous session created!')
    return signInData.session
  }
  
  return data.session
}

// Main function
async function main() {
  const session = await getSession()
  
  if (session) {
    console.log('Session found!')
    console.log('Access Token:', session.access_token)
    
    // Test a function call
    console.log('\nTesting test-connection function...')
    const { data, error } = await supabase.functions.invoke('test-connection', {
      body: {}
    })
    
    if (error) {
      console.error('Error calling function:', error)
    } else {
      console.log('Function response:', data)
    }
  } else {
    console.log('No session available. Please check your Supabase configuration.')
  }
}

main()

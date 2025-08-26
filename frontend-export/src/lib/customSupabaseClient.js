import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://emecscbwfcvbkvxbztfa.supabase.co';
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWNzY2J3ZmN2Ymt2eGJ6dGZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTUyOTc4NTYsImV4cCI6MjA3MDg3Mzg1Nn0.gKtvk5fiqjBToyy9w0_-DF4EmZdBQwUJCpY07XM7QJ0';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
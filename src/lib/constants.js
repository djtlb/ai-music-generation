/**
 * Constants for Beat Addicts application
 */

// Genre options for music generation
export const GENRE_OPTIONS = [
  { id: 'pop', name: 'Pop' },
  { id: 'rock', name: 'Rock' },
  { id: 'hiphop', name: 'Hip Hop' },
  { id: 'rnb', name: 'R&B' },
  { id: 'electronic', name: 'Electronic' },
  { id: 'jazz', name: 'Jazz' },
  { id: 'classical', name: 'Classical' },
  { id: 'folk', name: 'Folk' },
  { id: 'country', name: 'Country' },
  { id: 'metal', name: 'Metal' },
  { id: 'blues', name: 'Blues' },
  { id: 'funk', name: 'Funk' },
  { id: 'reggae', name: 'Reggae' },
  { id: 'ambient', name: 'Ambient' },
  { id: 'cinematic', name: 'Cinematic' },
  { id: 'indie', name: 'Indie' },
  { id: 'latin', name: 'Latin' },
  { id: 'afrobeat', name: 'Afrobeat' },
  { id: 'trap', name: 'Trap' },
  { id: 'drill', name: 'Drill' },
  { id: 'kpop', name: 'K-Pop' },
];

// Voice profile options
export const VOICE_PROFILES = [
  { id: 'male_tenor', name: 'Male Tenor' },
  { id: 'male_baritone', name: 'Male Baritone' },
  { id: 'male_bass', name: 'Male Bass' },
  { id: 'female_soprano', name: 'Female Soprano' },
  { id: 'female_alto', name: 'Female Alto' },
  { id: 'female_contralto', name: 'Female Contralto' },
  { id: 'androgynous', name: 'Androgynous' },
  { id: 'robotic', name: 'Robotic' },
  { id: 'synthetic', name: 'Synthetic' },
  { id: 'choir', name: 'Choir' },
  { id: 'ensemble', name: 'Ensemble' },
];

// Mood options
export const MOOD_OPTIONS = [
  { id: 'happy', name: 'Happy' },
  { id: 'sad', name: 'Sad' },
  { id: 'energetic', name: 'Energetic' },
  { id: 'calm', name: 'Calm' },
  { id: 'aggressive', name: 'Aggressive' },
  { id: 'romantic', name: 'Romantic' },
  { id: 'melancholic', name: 'Melancholic' },
  { id: 'nostalgic', name: 'Nostalgic' },
  { id: 'dark', name: 'Dark' },
  { id: 'triumphant', name: 'Triumphant' },
  { id: 'inspirational', name: 'Inspirational' },
  { id: 'playful', name: 'Playful' },
  { id: 'dreamy', name: 'Dreamy' },
  { id: 'tense', name: 'Tense' },
  { id: 'relaxed', name: 'Relaxed' },
];

// Instrument options
export const INSTRUMENT_OPTIONS = [
  { id: 'guitar', name: 'Guitar' },
  { id: 'piano', name: 'Piano' },
  { id: 'drums', name: 'Drums' },
  { id: 'bass', name: 'Bass' },
  { id: 'synth', name: 'Synthesizer' },
  { id: 'strings', name: 'Strings' },
  { id: 'brass', name: 'Brass' },
  { id: 'woodwinds', name: 'Woodwinds' },
  { id: 'percussion', name: 'Percussion' },
  { id: 'violin', name: 'Violin' },
  { id: 'cello', name: 'Cello' },
  { id: 'trumpet', name: 'Trumpet' },
  { id: 'saxophone', name: 'Saxophone' },
  { id: 'flute', name: 'Flute' },
  { id: 'harp', name: 'Harp' },
  { id: 'accordion', name: 'Accordion' },
  { id: 'organ', name: 'Organ' },
  { id: 'banjo', name: 'Banjo' },
  { id: 'mandolin', name: 'Mandolin' },
  { id: 'ukulele', name: 'Ukulele' },
  { id: 'electric_guitar', name: 'Electric Guitar' },
  { id: 'acoustic_guitar', name: 'Acoustic Guitar' },
  { id: 'electric_bass', name: 'Electric Bass' },
  { id: 'upright_bass', name: 'Upright Bass' },
  { id: 'grand_piano', name: 'Grand Piano' },
  { id: 'electric_piano', name: 'Electric Piano' },
];

// Project status options
export const PROJECT_STATUS = {
  DRAFT: 'draft',
  GENERATING: 'generating',
  COMPLETE: 'complete',
  FAILED: 'failed',
  ARCHIVED: 'archived',
};

// Audio quality options
export const AUDIO_QUALITY = {
  STANDARD: 'standard', // 128kbps
  HIGH: 'high',         // 256kbps
  PREMIUM: 'premium',   // 320kbps
  LOSSLESS: 'lossless', // FLAC
};

// Subscription tiers
export const SUBSCRIPTION_TIERS = {
  FREE: 'free',
  BASIC: 'basic',
  PRO: 'pro',
  ENTERPRISE: 'enterprise',
};

// API endpoints
export const API_ENDPOINTS = {
  MUSIC: '/api/v1/music',
  PROJECTS: '/api/v1/projects',
  COLLABORATION: '/api/v1/collaboration',
  COLLAB_LAB: '/api/v1/collab-lab',
  AUTH: '/api/v1/auth',
  MARKETPLACE: '/api/v1/marketplace',
  NFT: '/api/v1/nft',
  PAYMENTS: '/api/v1/payments',
};

// Default track settings
export const DEFAULT_TRACK_SETTINGS = {
  tempo: 120,
  key: 'C',
  timeSignature: '4/4',
  duration: 180, // 3 minutes in seconds
};

// Maximum file upload sizes
export const MAX_UPLOAD_SIZES = {
  AUDIO: 50 * 1024 * 1024, // 50MB
  IMAGE: 5 * 1024 * 1024,  // 5MB
  LYRICS: 50 * 1024,       // 50KB
};

// Permissions
export const PERMISSIONS = {
  VIEW_PROJECT: 'view_project',
  EDIT_PROJECT: 'edit_project',
  DELETE_PROJECT: 'delete_project',
  GENERATE_MUSIC: 'generate_music',
  COLLABORATE: 'collaborate',
  MINT_NFT: 'mint_nft',
  PUBLISH: 'publish',
};

// WebSocket event types
export const WS_EVENT_TYPES = {
  CONNECT: 'connect',
  DISCONNECT: 'disconnect',
  ERROR: 'error',
  MESSAGE: 'message',
  GENERATION_STARTED: 'generation_started',
  GENERATION_PROGRESS: 'generation_progress',
  GENERATION_COMPLETE: 'generation_complete',
  GENERATION_FAILED: 'generation_failed',
  COLLABORATION_UPDATE: 'collaboration_update',
  USER_JOINED: 'user_joined',
  USER_LEFT: 'user_left',
};

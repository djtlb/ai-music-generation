/**
 * Simple tokenizer demonstration script
 * Shows basic encode/decode functionality
 */

import { MidiTokenizer, MultiTrackMidi } from './tokenizer';

// Create tokenizer instance
const tokenizer = new MidiTokenizer();

// Simple test MIDI
const testMidi: MultiTrackMidi = {
  style: 'rock_punk',
  tempo: 120,
  key: 'Em',
  sections: [
    { type: 'VERSE', start: 0, length: 4 }
  ],
  tracks: {
    KICK: [
      { pitch: 36, velocity: 110, start: 0, duration: 1, track: 'KICK' },
      { pitch: 36, velocity: 110, start: 4, duration: 1, track: 'KICK' }
    ],
    SNARE: [
      { pitch: 38, velocity: 100, start: 2, duration: 1, track: 'SNARE' }
    ]
  }
};

console.log('=== MIDI Tokenizer Demo ===');
console.log('Vocabulary size:', tokenizer.getVocabSize());

console.log('\nOriginal MIDI:');
console.log(JSON.stringify(testMidi, null, 2));

// Encode
const tokens = tokenizer.encode(testMidi);
console.log(`\nEncoded to ${tokens.length} tokens:`, tokens.slice(0, 10), '...');

// Decode
const decoded = tokenizer.decode(tokens);
console.log('\nDecoded MIDI:');
console.log(JSON.stringify(decoded, null, 2));

// Verify round-trip
const reEncoded = tokenizer.encode(decoded);
console.log('\nRound-trip verification:');
console.log('Original tokens:', tokens.length);
console.log('Re-encoded tokens:', reEncoded.length);
console.log('Tokens match:', JSON.stringify(tokens) === JSON.stringify(reEncoded));

export { tokenizer, testMidi };
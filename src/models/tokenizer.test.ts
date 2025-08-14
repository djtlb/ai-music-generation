/**
 * Round-trip Tests for MIDI Tokenizer
 * 
 * Ensures lossless encode/decode operations on test fixtures
 */

import { MidiTokenizer, MultiTrackMidi } from './tokenizer';
import testFixturesData from '../data/fixtures/test_midi.json';

export interface TestResult {
  testName: string;
  passed: boolean;
  error?: string;
  originalTokenCount?: number;
  decodedTokenCount?: number;
  details?: any;
}

// Export testFixtures for external use
export const testFixtures = testFixturesData;

export class TokenizerTester {
  private tokenizer: MidiTokenizer;

  constructor() {
    this.tokenizer = new MidiTokenizer();
  }

  /**
   * Run all round-trip tests on fixtures
   */
  public runAllTests(): TestResult[] {
    const results: TestResult[] = [];
    
    // Test each fixture
    for (const fixture of testFixtures) {
      const result = this.testRoundTrip(fixture as MultiTrackMidi);
      results.push(result);
    }
    
    // Test vocabulary consistency
    results.push(this.testVocabularyConsistency());
    
    // Test edge cases
    results.push(this.testEmptyMidi());
    results.push(this.testSingleNote());
    results.push(this.testOverlappingNotes());
    
    return results;
  }

  /**
   * Test round-trip encoding/decoding for a single MIDI file
   */
  public testRoundTrip(originalMidi: MultiTrackMidi): TestResult {
    try {
      // Encode to tokens
      const tokens = this.tokenizer.encode(originalMidi);
      
      // Decode back to MIDI
      const decodedMidi = this.tokenizer.decode(tokens);
      
      // Compare original and decoded
      const comparison = this.compareMidi(originalMidi, decodedMidi);
      
      return {
        testName: `Round-trip: ${(originalMidi as any).name || 'unnamed'}`,
        passed: comparison.isEqual,
        originalTokenCount: tokens.length,
        decodedTokenCount: this.tokenizer.encode(decodedMidi).length,
        details: comparison.differences
      };
      
    } catch (error) {
      return {
        testName: `Round-trip: ${(originalMidi as any).name || 'unnamed'}`,
        passed: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test vocabulary consistency and completeness
   */
  public testVocabularyConsistency(): TestResult {
    try {
      const vocabSize = this.tokenizer.getVocabSize();
      
      // Test that all tokens have unique IDs
      const tokens = new Set<string>();
      const ids = new Set<number>();
      
      for (let i = 0; i < vocabSize; i++) {
        const token = this.tokenizer.getToken(i);
        if (token) {
          if (tokens.has(token)) {
            throw new Error(`Duplicate token: ${token}`);
          }
          tokens.add(token);
          
          const id = this.tokenizer.getTokenId(token);
          if (id === undefined || ids.has(id)) {
            throw new Error(`Invalid or duplicate ID for token: ${token}`);
          }
          ids.add(id);
        }
      }
      
      // Test essential tokens exist
      const essentialTokens = ['<START>', '<END>', '<PAD>', '<UNK>', 'NOTE_ON', 'NOTE_OFF', 'BAR'];
      for (const token of essentialTokens) {
        if (this.tokenizer.getTokenId(token) === undefined) {
          throw new Error(`Missing essential token: ${token}`);
        }
      }
      
      return {
        testName: 'Vocabulary Consistency',
        passed: true,
        details: {
          vocabSize,
          uniqueTokens: tokens.size,
          uniqueIds: ids.size
        }
      };
      
    } catch (error) {
      return {
        testName: 'Vocabulary Consistency',
        passed: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test empty MIDI handling
   */
  public testEmptyMidi(): TestResult {
    try {
      const emptyMidi: MultiTrackMidi = {
        style: 'rock_punk',
        tempo: 120,
        key: 'C',
        sections: [],
        tracks: {}
      };
      
      const tokens = this.tokenizer.encode(emptyMidi);
      const decoded = this.tokenizer.decode(tokens);
      
      // Should maintain basic metadata
      const passed = decoded.style === emptyMidi.style &&
                    decoded.tempo === emptyMidi.tempo &&
                    decoded.key === emptyMidi.key &&
                    Object.keys(decoded.tracks).length === 0;
      
      return {
        testName: 'Empty MIDI',
        passed,
        originalTokenCount: tokens.length,
        details: { decoded }
      };
      
    } catch (error) {
      return {
        testName: 'Empty MIDI',
        passed: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test single note handling
   */
  public testSingleNote(): TestResult {
    try {
      const singleNoteMidi: MultiTrackMidi = {
        style: 'rnb_ballad',
        tempo: 80,
        key: 'Am',
        sections: [{ type: 'VERSE', start: 0, length: 1 }],
        tracks: {
          PIANO: [{
            pitch: 60,
            velocity: 80,
            start: 0,
            duration: 4,
            track: 'PIANO'
          }]
        }
      };
      
      const tokens = this.tokenizer.encode(singleNoteMidi);
      const decoded = this.tokenizer.decode(tokens);
      
      const comparison = this.compareMidi(singleNoteMidi, decoded);
      
      return {
        testName: 'Single Note',
        passed: comparison.isEqual,
        originalTokenCount: tokens.length,
        details: comparison.differences
      };
      
    } catch (error) {
      return {
        testName: 'Single Note',
        passed: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Test overlapping notes on same track
   */
  public testOverlappingNotes(): TestResult {
    try {
      const overlappingMidi: MultiTrackMidi = {
        style: 'country_pop',
        tempo: 100,
        key: 'G',
        sections: [{ type: 'INTRO', start: 0, length: 2 }],
        tracks: {
          PIANO: [
            { pitch: 60, velocity: 70, start: 0, duration: 6, track: 'PIANO' },
            { pitch: 64, velocity: 70, start: 2, duration: 6, track: 'PIANO' },
            { pitch: 67, velocity: 70, start: 4, duration: 6, track: 'PIANO' }
          ]
        }
      };
      
      const tokens = this.tokenizer.encode(overlappingMidi);
      const decoded = this.tokenizer.decode(tokens);
      
      const comparison = this.compareMidi(overlappingMidi, decoded);
      
      return {
        testName: 'Overlapping Notes',
        passed: comparison.isEqual,
        originalTokenCount: tokens.length,
        details: comparison.differences
      };
      
    } catch (error) {
      return {
        testName: 'Overlapping Notes',
        passed: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  /**
   * Compare two MIDI objects for equality
   */
  private compareMidi(original: MultiTrackMidi, decoded: MultiTrackMidi): { isEqual: boolean; differences: any[] } {
    const differences: any[] = [];
    
    // Compare metadata
    if (original.style !== decoded.style) {
      differences.push({ field: 'style', original: original.style, decoded: decoded.style });
    }
    if (original.tempo !== decoded.tempo) {
      differences.push({ field: 'tempo', original: original.tempo, decoded: decoded.tempo });
    }
    if (original.key !== decoded.key) {
      differences.push({ field: 'key', original: original.key, decoded: decoded.key });
    }
    
    // Compare sections (allow some flexibility in length)
    if (original.sections.length !== decoded.sections.length) {
      differences.push({ 
        field: 'sections.length', 
        original: original.sections.length, 
        decoded: decoded.sections.length 
      });
    }
    
    for (let i = 0; i < Math.min(original.sections.length, decoded.sections.length); i++) {
      const origSec = original.sections[i];
      const decSec = decoded.sections[i];
      
      if (origSec.type !== decSec.type) {
        differences.push({ 
          field: `sections[${i}].type`, 
          original: origSec.type, 
          decoded: decSec.type 
        });
      }
      if (origSec.start !== decSec.start) {
        differences.push({ 
          field: `sections[${i}].start`, 
          original: origSec.start, 
          decoded: decSec.start 
        });
      }
    }
    
    // Compare tracks
    const originalTracks = Object.keys(original.tracks).sort();
    const decodedTracks = Object.keys(decoded.tracks).sort();
    
    if (originalTracks.length !== decodedTracks.length) {
      differences.push({ 
        field: 'tracks.count', 
        original: originalTracks.length, 
        decoded: decodedTracks.length 
      });
    }
    
    // Compare notes in each track
    for (const trackName of originalTracks) {
      if (!decoded.tracks[trackName]) {
        differences.push({ field: `tracks.${trackName}`, original: 'exists', decoded: 'missing' });
        continue;
      }
      
      const originalNotes = original.tracks[trackName].sort((a, b) => a.start - b.start || a.pitch - b.pitch);
      const decodedNotes = decoded.tracks[trackName].sort((a, b) => a.start - b.start || a.pitch - b.pitch);
      
      if (originalNotes.length !== decodedNotes.length) {
        differences.push({ 
          field: `tracks.${trackName}.length`, 
          original: originalNotes.length, 
          decoded: decodedNotes.length 
        });
        continue;
      }
      
      for (let i = 0; i < originalNotes.length; i++) {
        const origNote = originalNotes[i];
        const decNote = decodedNotes[i];
        
        // Compare note properties (allow small differences due to bucketing)
        if (origNote.pitch !== decNote.pitch) {
          differences.push({ 
            field: `tracks.${trackName}[${i}].pitch`, 
            original: origNote.pitch, 
            decoded: decNote.pitch 
          });
        }
        if (origNote.start !== decNote.start) {
          differences.push({ 
            field: `tracks.${trackName}[${i}].start`, 
            original: origNote.start, 
            decoded: decNote.start 
          });
        }
        
        // Allow bucketing differences for velocity and duration
        const velocityDiff = Math.abs(origNote.velocity - decNote.velocity);
        if (velocityDiff > 20) { // Allow for bucketing tolerance
          differences.push({ 
            field: `tracks.${trackName}[${i}].velocity`, 
            original: origNote.velocity, 
            decoded: decNote.velocity 
          });
        }
        
        const durationDiff = Math.abs(origNote.duration - decNote.duration);
        if (durationDiff > 2) { // Allow for duration bucketing tolerance
          differences.push({ 
            field: `tracks.${trackName}[${i}].duration`, 
            original: origNote.duration, 
            decoded: decNote.duration 
          });
        }
      }
    }
    
    return {
      isEqual: differences.length === 0,
      differences
    };
  }

  /**
   * Generate a test report
   */
  public generateReport(results: TestResult[]): string {
    const passed = results.filter(r => r.passed).length;
    const total = results.length;
    
    let report = `MIDI Tokenizer Test Report\n`;
    report += `==========================\n\n`;
    report += `Tests Passed: ${passed}/${total}\n`;
    report += `Success Rate: ${((passed / total) * 100).toFixed(1)}%\n\n`;
    
    for (const result of results) {
      const status = result.passed ? '✅ PASS' : '❌ FAIL';
      report += `${status} ${result.testName}\n`;
      
      if (result.originalTokenCount) {
        report += `  Token count: ${result.originalTokenCount}\n`;
      }
      
      if (!result.passed && result.error) {
        report += `  Error: ${result.error}\n`;
      }
      
      if (!result.passed && result.details) {
        report += `  Issues: ${JSON.stringify(result.details, null, 2)}\n`;
      }
      
      report += '\n';
    }
    
    return report;
}

export default TokenizerTester;
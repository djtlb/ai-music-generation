/**
 * Multi-track MIDI Tokenizer for AI Music Generation
 * 
 * Converts MIDI data to/from tokens for neural network training.
 * Supports style-aware encoding with proper temporal and harmonic context.
 */

export interface MidiNote {
  pitch: number;
  velocity: number;
  start: number;  // in 1/16th note positions
  duration: number;  // in 1/16th note positions
  track: string;  // instrument role
}

export interface MidiEvent {
  type: 'NOTE_ON' | 'NOTE_OFF' | 'CHORD' | 'SECTION' | 'STYLE' | 'TEMPO' | 'KEY' | 'BAR' | 'POS';
  time: number;  // in 1/16th note positions
  data: any;
}

export interface MultiTrackMidi {
  style: 'rock_punk' | 'rnb_ballad' | 'country_pop';
  tempo: number;
  key: string;
  sections: Section[];
  tracks: { [role: string]: MidiNote[] };
}

export interface Section {
  type: 'INTRO' | 'VERSE' | 'CHORUS' | 'BRIDGE' | 'OUTRO';
  start: number;  // bar number
  length: number;  // in bars
}

export interface TokenizerVocab {
  // Style tokens
  styles: string[];
  
  // Musical structure tokens
  sections: string[];
  instruments: string[];
  
  // Time tokens
  tempos: number[];
  keys: string[];
  positions: string[];  // 1/16 grid positions
  
  // Note tokens
  pitches: number[];
  velocities: number[];  // bucketed
  durations: string[];   // bucketed
  
  // Chord tokens
  chords: string[];
  
  // Special tokens
  special: string[];
}

/**
 * MIDI Tokenizer with vocabulary management and round-trip encoding/decoding
 */
export class MidiTokenizer {
  private vocab: TokenizerVocab;
  private tokenToId: Map<string, number>;
  private idToToken: Map<number, string>;

  constructor() {
    this.vocab = this.createVocabulary();
    this.buildTokenMaps();
  }

  private createVocabulary(): TokenizerVocab {
    return {
      styles: ['rock_punk', 'rnb_ballad', 'country_pop'],
      
      sections: ['INTRO', 'VERSE', 'CHORUS', 'BRIDGE', 'OUTRO'],
      
      instruments: [
        'KICK', 'SNARE', 'HIHAT', 'CRASH', 'RIDE',
        'BASS_PICK', 'BASS_SLAP', 
        'ACOUSTIC_STRUM', 'ELECTRIC_POWER', 'ELECTRIC_CLEAN',
        'PIANO', 'SYNTH_PAD', 'SYNTH_LEAD',
        'LEAD', 'HARMONY'
      ],
      
      tempos: Array.from({length: 101}, (_, i) => 60 + i * 2), // 60-260 BPM in steps of 2
      
      keys: [
        'C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B',
        'Cm', 'C#m', 'Dbm', 'Dm', 'D#m', 'Ebm', 'Em', 'Fm', 'F#m', 'Gbm', 'Gm', 'G#m', 'Abm', 'Am', 'A#m', 'Bbm', 'Bm'
      ],
      
      positions: Array.from({length: 64}, (_, i) => `POS_${i}`), // 0-63 sixteenth note positions (4 bars)
      
      pitches: Array.from({length: 128}, (_, i) => i), // MIDI pitch range 0-127
      
      // Velocity buckets: pp(20-39), p(40-59), mp(60-79), mf(80-99), f(100-119), ff(120-127)
      velocities: [30, 50, 70, 90, 110, 125],
      
      // Duration buckets: 16th, 8th, dotted 8th, quarter, dotted quarter, half, whole
      durations: ['DUR_1', 'DUR_2', 'DUR_3', 'DUR_4', 'DUR_6', 'DUR_8', 'DUR_16'],
      
      chords: [
        // Major triads
        'C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B',
        // Minor triads
        'Cm', 'C#m', 'Dbm', 'Dm', 'D#m', 'Ebm', 'Em', 'Fm', 'F#m', 'Gbm', 'Gm', 'G#m', 'Abm', 'Am', 'A#m', 'Bbm', 'Bm',
        // Seventh chords
        'C7', 'Cm7', 'Cmaj7', 'D7', 'Dm7', 'Dmaj7', 'E7', 'Em7', 'Emaj7', 'F7', 'Fm7', 'Fmaj7',
        'G7', 'Gm7', 'Gmaj7', 'A7', 'Am7', 'Amaj7', 'B7', 'Bm7', 'Bmaj7',
        // Extended and altered chords
        'C9', 'Cm9', 'C11', 'C13', 'Cadd9', 'Csus2', 'Csus4', 'Cdim', 'Caug'
      ],
      
      special: ['<PAD>', '<START>', '<END>', '<UNK>', 'BAR', 'NOTE_ON', 'NOTE_OFF']
    };
  }

  private buildTokenMaps(): void {
    this.tokenToId = new Map();
    this.idToToken = new Map();
    
    let tokenId = 0;
    
    // Add all vocabulary tokens
    for (const [category, tokens] of Object.entries(this.vocab)) {
      for (const token of tokens) {
        const tokenStr = Array.isArray(token) ? String(token) : token;
        const prefixedToken = category === 'special' ? tokenStr : `${category.toUpperCase()}_${tokenStr}`;
        
        this.tokenToId.set(prefixedToken, tokenId);
        this.idToToken.set(tokenId, prefixedToken);
        tokenId++;
      }
    }
  }

  /**
   * Bucket velocity into 6 dynamic levels
   */
  private bucketVelocity(velocity: number): number {
    if (velocity < 40) return 30;
    if (velocity < 60) return 50;
    if (velocity < 80) return 70;
    if (velocity < 100) return 90;
    if (velocity < 120) return 110;
    return 125;
  }

  /**
   * Bucket duration into musical note values
   */
  private bucketDuration(duration: number): string {
    if (duration <= 1) return 'DUR_1';  // 16th note
    if (duration <= 2) return 'DUR_2';  // 8th note
    if (duration <= 3) return 'DUR_3';  // dotted 8th
    if (duration <= 4) return 'DUR_4';  // quarter note
    if (duration <= 6) return 'DUR_6';  // dotted quarter
    if (duration <= 8) return 'DUR_8';  // half note
    return 'DUR_16';  // whole note or longer
  }

  /**
   * Convert bucketed duration back to numeric value
   */
  private unbucketDuration(durationToken: string): number {
    const durationMap: { [key: string]: number } = {
      'DUR_1': 1,
      'DUR_2': 2,
      'DUR_3': 3,
      'DUR_4': 4,
      'DUR_6': 6,
      'DUR_8': 8,
      'DUR_16': 16
    };
    return durationMap[durationToken] || 4;
  }

  /**
   * Encode multi-track MIDI to token sequence
   */
  public encode(midi: MultiTrackMidi): number[] {
    const tokens: string[] = [];
    
    // Start token
    tokens.push('<START>');
    
    // Global metadata
    tokens.push(`STYLES_${midi.style}`);
    tokens.push(`TEMPOS_${midi.tempo}`);
    tokens.push(`KEYS_${midi.key}`);
    
    // Calculate total length in 16th notes
    const maxTime = Math.max(
      ...Object.values(midi.tracks).flat().map(note => note.start + note.duration)
    );
    const totalBars = Math.ceil(maxTime / 16);
    
    // Create timeline of events
    const events: MidiEvent[] = [];
    
    // Add section markers
    for (const section of midi.sections) {
      events.push({
        type: 'SECTION',
        time: section.start * 16, // Convert bars to 16th notes
        data: section.type
      });
    }
    
    // Add bar markers
    for (let bar = 0; bar < totalBars; bar++) {
      events.push({
        type: 'BAR',
        time: bar * 16,
        data: bar
      });
    }
    
    // Add note events
    for (const [trackName, notes] of Object.entries(midi.tracks)) {
      for (const note of notes) {
        // Note on event
        events.push({
          type: 'NOTE_ON',
          time: note.start,
          data: {
            track: note.track,
            pitch: note.pitch,
            velocity: this.bucketVelocity(note.velocity)
          }
        });
        
        // Note off event
        events.push({
          type: 'NOTE_OFF',
          time: note.start + note.duration,
          data: {
            track: note.track,
            pitch: note.pitch,
            duration: this.bucketDuration(note.duration)
          }
        });
      }
    }
    
    // Sort events by time
    events.sort((a, b) => a.time - b.time);
    
    // Convert events to tokens
    let currentBar = 0;
    let currentPosition = 0;
    
    for (const event of events) {
      const eventBar = Math.floor(event.time / 16);
      const eventPos = event.time % 16;
      
      // Add bar token if needed
      if (eventBar > currentBar) {
        tokens.push('BAR');
        currentBar = eventBar;
        currentPosition = 0;
      }
      
      // Add position token if needed
      if (eventPos > currentPosition) {
        tokens.push(`POSITIONS_POS_${eventPos}`);
        currentPosition = eventPos;
      }
      
      // Add event-specific tokens
      switch (event.type) {
        case 'SECTION':
          tokens.push(`SECTIONS_${event.data}`);
          break;
          
        case 'NOTE_ON':
          tokens.push('NOTE_ON');
          tokens.push(`INSTRUMENTS_${event.data.track}`);
          tokens.push(`PITCHES_${event.data.pitch}`);
          tokens.push(`VELOCITIES_${event.data.velocity}`);
          break;
          
        case 'NOTE_OFF':
          tokens.push('NOTE_OFF');
          tokens.push(`INSTRUMENTS_${event.data.track}`);
          tokens.push(`PITCHES_${event.data.pitch}`);
          tokens.push(`DURATIONS_${event.data.duration}`);
          break;
      }
    }
    
    // End token
    tokens.push('<END>');
    
    // Convert tokens to IDs
    return tokens.map(token => {
      const id = this.tokenToId.get(token);
      if (id === undefined) {
        console.warn(`Unknown token: ${token}`);
        return this.tokenToId.get('<UNK>') || 0;
      }
      return id;
    });
  }

  /**
   * Decode token sequence back to multi-track MIDI
   */
  public decode(tokenIds: number[]): MultiTrackMidi {
    const tokens = tokenIds.map(id => this.idToToken.get(id) || '<UNK>');
    
    const result: MultiTrackMidi = {
      style: 'rock_punk',  // default
      tempo: 120,  // default
      key: 'C',  // default
      sections: [],
      tracks: {}
    };
    
    let currentTime = 0;
    let currentBar = 0;
    let activeNotes: Map<string, { track: string; pitch: number; start: number; velocity: number }> = new Map();
    
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      
      if (token.startsWith('STYLES_')) {
        result.style = token.replace('STYLES_', '') as any;
      } else if (token.startsWith('TEMPOS_')) {
        result.tempo = parseInt(token.replace('TEMPOS_', ''));
      } else if (token.startsWith('KEYS_')) {
        result.key = token.replace('KEYS_', '');
      } else if (token.startsWith('SECTIONS_')) {
        const sectionType = token.replace('SECTIONS_', '') as any;
        result.sections.push({
          type: sectionType,
          start: currentBar,
          length: 4  // default section length
        });
      } else if (token === 'BAR') {
        currentBar++;
        currentTime = currentBar * 16;
      } else if (token.startsWith('POSITIONS_POS_')) {
        const pos = parseInt(token.replace('POSITIONS_POS_', ''));
        currentTime = currentBar * 16 + pos;
      } else if (token === 'NOTE_ON' && i + 3 < tokens.length) {
        const trackToken = tokens[i + 1];
        const pitchToken = tokens[i + 2];
        const velocityToken = tokens[i + 3];
        
        if (trackToken.startsWith('INSTRUMENTS_') && 
            pitchToken.startsWith('PITCHES_') && 
            velocityToken.startsWith('VELOCITIES_')) {
          
          const track = trackToken.replace('INSTRUMENTS_', '');
          const pitch = parseInt(pitchToken.replace('PITCHES_', ''));
          const velocity = parseInt(velocityToken.replace('VELOCITIES_', ''));
          
          const noteKey = `${track}_${pitch}`;
          activeNotes.set(noteKey, {
            track,
            pitch,
            start: currentTime,
            velocity
          });
          
          i += 3; // Skip the consumed tokens
        }
      } else if (token === 'NOTE_OFF' && i + 3 < tokens.length) {
        const trackToken = tokens[i + 1];
        const pitchToken = tokens[i + 2];
        const durationToken = tokens[i + 3];
        
        if (trackToken.startsWith('INSTRUMENTS_') && 
            pitchToken.startsWith('PITCHES_') && 
            durationToken.startsWith('DURATIONS_')) {
          
          const track = trackToken.replace('INSTRUMENTS_', '');
          const pitch = parseInt(pitchToken.replace('PITCHES_', ''));
          const duration = this.unbucketDuration(durationToken.replace('DURATIONS_', ''));
          
          const noteKey = `${track}_${pitch}`;
          const activeNote = activeNotes.get(noteKey);
          
          if (activeNote) {
            // Initialize track if needed
            if (!result.tracks[track]) {
              result.tracks[track] = [];
            }
            
            // Add completed note
            result.tracks[track].push({
              pitch: activeNote.pitch,
              velocity: activeNote.velocity,
              start: activeNote.start,
              duration: duration,
              track: activeNote.track
            });
            
            activeNotes.delete(noteKey);
          }
          
          i += 3; // Skip the consumed tokens
        }
      }
    }
    
    return result;
  }

  /**
   * Get the vocabulary as JSON for external use
   */
  public getVocabularyJson(): string {
    return JSON.stringify(this.vocab, null, 2);
  }

  /**
   * Get vocabulary size
   */
  public getVocabSize(): number {
    return this.tokenToId.size;
  }

  /**
   * Get token from ID
   */
  public getToken(id: number): string | undefined {
    return this.idToToken.get(id);
  }

  /**
   * Get ID from token
   */
  public getTokenId(token: string): number | undefined {
    return this.tokenToId.get(token);
  }
}
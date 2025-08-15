// Audio synthesis utilities for MIDI playback
export interface MIDINote {
  pitch: number;
  start: number;
  duration: number;
  velocity: number;
  chord?: string;
}

export interface MIDITrack {
  name: string;
  channel: number;
  notes: MIDINote[];
  instrument: string;
}

export interface ChordInfo {
  root: string;
  quality: string;
  notes: number[];
}

// Convert MIDI note number to frequency in Hz
export function midiToFreq(midiNote: number): number {
  return 440 * Math.pow(2, (midiNote - 69) / 12);
}

// Convert note name to MIDI number
export function noteToMidi(noteName: string): number {
  const noteMap: { [key: string]: number } = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
    'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
  };
  
  const match = noteName.match(/^([A-G][#b]?)(\d+)$/);
  if (!match) return 60; // Default to C4
  
  const [, note, octave] = match;
  return (parseInt(octave) + 1) * 12 + noteMap[note];
}

// Get chord notes from chord symbol
export function getChordNotes(chordSymbol: string): number[] {
  const chordMap: { [key: string]: number[] } = {
    'C': [60, 64, 67],      // C major
    'Cm': [60, 63, 67],     // C minor
    'C7': [60, 64, 67, 70], // C dominant 7
    'Cmaj7': [60, 64, 67, 71], // C major 7
    'Cm7': [60, 63, 67, 70],   // C minor 7
    'D': [62, 66, 69],
    'Dm': [62, 65, 69],
    'D7': [62, 66, 69, 72],
    'E': [64, 68, 71],
    'Em': [64, 67, 71],
    'E7': [64, 68, 71, 74],
    'F': [65, 69, 72],
    'Fm': [65, 68, 72],
    'F7': [65, 69, 72, 75],
    'G': [67, 71, 74],
    'Gm': [67, 70, 74],
    'G7': [67, 71, 74, 77],
    'A': [69, 73, 76],
    'Am': [69, 72, 76],
    'A7': [69, 73, 76, 79],
    'B': [71, 75, 78],
    'Bm': [71, 74, 78],
    'B7': [71, 75, 78, 81]
  };
  
  return chordMap[chordSymbol] || [60, 64, 67]; // Default to C major
}

export class AudioSynthesizer {
  private audioContext: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private activeNotes: Map<string, OscillatorNode> = new Map();
  private scheduledEvents: number[] = [];

  constructor() {
    this.init();
  }

  private async init() {
    try {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.masterGain = this.audioContext.createGain();
      this.masterGain.connect(this.audioContext.destination);
      this.masterGain.gain.value = 0.3; // Master volume
    } catch (error) {
      console.error('Failed to initialize audio context:', error);
    }
  }

  async resume() {
    if (this.audioContext?.state === 'suspended') {
      await this.audioContext.resume();
    }
  }

  createOscillator(frequency: number, type: OscillatorType = 'sine'): OscillatorNode {
    if (!this.audioContext) throw new Error('Audio context not initialized');
    
    const oscillator = this.audioContext.createOscillator();
    const gainNode = this.audioContext.createGain();
    
    oscillator.frequency.value = frequency;
    oscillator.type = type;
    
    oscillator.connect(gainNode);
    gainNode.connect(this.masterGain!);
    
    return oscillator;
  }

  playNote(pitch: number, duration: number, velocity: number = 80, instrument: string = 'piano'): void {
    if (!this.audioContext || !this.masterGain) return;

    const frequency = midiToFreq(pitch);
    const noteKey = `${pitch}-${Date.now()}`;
    
    // Choose oscillator type based on instrument
    let oscType: OscillatorType = 'sine';
    switch (instrument.toLowerCase()) {
      case 'guitar':
      case 'acoustic':
        oscType = 'sawtooth';
        break;
      case 'bass':
        oscType = 'triangle';
        break;
      case 'piano':
      case 'melody':
        oscType = 'sine';
        break;
      default:
        oscType = 'sine';
    }

    const oscillator = this.createOscillator(frequency, oscType);
    const gainNode = this.audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(this.masterGain);
    
    // Set volume based on velocity
    const volume = (velocity / 127) * 0.2;
    gainNode.gain.value = 0;
    gainNode.gain.setValueAtTime(0, this.audioContext.currentTime);
    gainNode.gain.linearRampToValueAtTime(volume, this.audioContext.currentTime + 0.01);
    gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + duration);
    
    oscillator.start(this.audioContext.currentTime);
    oscillator.stop(this.audioContext.currentTime + duration);
    
    this.activeNotes.set(noteKey, oscillator);
    
    // Clean up after note ends
    setTimeout(() => {
      this.activeNotes.delete(noteKey);
    }, duration * 1000 + 100);
  }

  playChord(chordSymbol: string, duration: number = 2.0, velocity: number = 70): void {
    const notes = getChordNotes(chordSymbol);
    notes.forEach((pitch, index) => {
      // Slight stagger for more natural chord sound
      setTimeout(() => {
        this.playNote(pitch, duration, velocity - index * 5, 'piano');
      }, index * 20);
    });
  }

  playMIDITrack(track: MIDITrack, startTime: number = 0): Promise<void> {
    return new Promise((resolve) => {
      if (!track.notes.length) {
        resolve();
        return;
      }

      const maxEndTime = Math.max(...track.notes.map(note => note.start + note.duration));
      
      track.notes.forEach(note => {
        const noteStartTime = (startTime + note.start) * 1000;
        const timeoutId = setTimeout(() => {
          this.playNote(note.pitch, note.duration, note.velocity, track.instrument);
        }, noteStartTime);
        
        this.scheduledEvents.push(timeoutId);
      });

      // Resolve when the track finishes
      const trackEndTime = (startTime + maxEndTime) * 1000;
      const endTimeoutId = setTimeout(() => {
        resolve();
      }, trackEndTime);
      
      this.scheduledEvents.push(endTimeoutId);
    });
  }

  async playMIDIComposition(tracks: MIDITrack[]): Promise<void> {
    await this.resume();
    
    // Play all tracks simultaneously
    const trackPromises = tracks.map(track => this.playMIDITrack(track));
    
    try {
      await Promise.all(trackPromises);
    } catch (error) {
      console.error('Error playing composition:', error);
    }
  }

  playChordProgression(chords: string[], chordDuration: number = 2.0): Promise<void> {
    return new Promise((resolve) => {
      chords.forEach((chord, index) => {
        const startTime = index * chordDuration * 1000;
        const timeoutId = setTimeout(() => {
          this.playChord(chord, chordDuration);
          if (index === chords.length - 1) {
            // Resolve after the last chord finishes
            setTimeout(resolve, chordDuration * 1000);
          }
        }, startTime);
        
        this.scheduledEvents.push(timeoutId);
      });
    });
  }

  stopAll(): void {
    // Clear all scheduled events
    this.scheduledEvents.forEach(id => clearTimeout(id));
    this.scheduledEvents = [];
    
    // Stop all active oscillators
    this.activeNotes.forEach(oscillator => {
      try {
        oscillator.stop();
      } catch (error) {
        // Oscillator might already be stopped
      }
    });
    this.activeNotes.clear();
  }

  setMasterVolume(volume: number): void {
    if (this.masterGain) {
      this.masterGain.gain.value = Math.max(0, Math.min(1, volume));
    }
  }
}

// Global audio synthesizer instance
export const audioSynth = new AudioSynthesizer();
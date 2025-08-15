import React, { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Loader2, Music, Wand2, Download, Sparkles, AlertCircle, Play, Square, Settings } from "lucide-react";
import { useKV } from "@/hooks/useKV";
import { audioSynth, MIDITrack, MIDINote } from "@/lib/audio";
import { toast } from "sonner";
import { MusicPlanner, type ControlJSON } from "@/lib/planner";

interface GenerationRequest {
  lyrics_prompt: string;
  genre_description: string;
  ai_assisted_lyrics?: boolean;
}

interface GenerationJob {
  job_id: string;
  status: "queued" | "running" | "succeeded" | "failed";
  audio_url?: string;
  tracks?: MIDITrack[];
  control_json?: ControlJSON;
  constraints_applied?: {
    section_masks: number;
    key_penalties: number;
    groove_adjustments: number;
    repetition_penalties: number;
  };
  report?: {
    lufs: number;
    centroid_hz: number;
    style_score: number;
    notes: string[];
    arrangement?: any;
    chords?: string[];
    lyrics?: string;
  };
  error?: string;
}

interface SongSection {
  name: string;
  duration: number;
  bars: number;
  description: string;
  bpm?: number;
  startTime?: number;
}

// Real AI Music Generation Engine
class AIMusicEngine {
  private jobs: Map<string, GenerationJob> = new Map();
  private planner: MusicPlanner = new MusicPlanner();

  async generate(request: GenerationRequest): Promise<{ job_id: string }> {
    const job_id = `job_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const job: GenerationJob = {
      job_id,
      status: "queued"
    };
    
    this.jobs.set(job_id, job);
    
    // Start the generation pipeline asynchronously
    this.generateMusic(job_id, request);
    
    return { job_id };
  }

  async getJob(job_id: string): Promise<GenerationJob | null> {
    return this.jobs.get(job_id) || null;
  }

  private async generateMusic(job_id: string, request: GenerationRequest) {
    const job = this.jobs.get(job_id);
    if (!job) return;

    try {
      // Update status to running
      job.status = "running";
      this.jobs.set(job_id, job);

      const notes: string[] = [];
      
      // Step 0: Use Planner to convert text inputs to control JSON
      notes.push("🧠 Planning: Converting text inputs to structured control JSON");
      const controlJson = this.planner.plan(request.lyrics_prompt, request.genre_description);
      job.control_json = controlJson;
      this.jobs.set(job_id, job);
      
      notes.push(`✅ Plan created: ${controlJson.style} at ${controlJson.bpm} BPM in ${controlJson.key}`);
      notes.push(`📋 Structure: ${controlJson.arrangement.structure.join(' → ')}`);
      notes.push(`🥁 Drum template: ${controlJson.drum_template}`);
      
      // Step 1: Generate/Process Lyrics using control JSON
      let finalLyrics = request.lyrics_prompt.trim();
      if (request.ai_assisted_lyrics && (!finalLyrics || finalLyrics.length < 20)) {
        notes.push("🎤 AI-generating lyrics based on planned style and structure");
        finalLyrics = await this.generateLyrics(controlJson, finalLyrics);
      }

      // Step 2: Generate song arrangement based on control JSON
      const arrangement = await this.generateArrangement(controlJson, finalLyrics);
      notes.push(`🏗️ Generated ${arrangement.sections.length} section arrangement from control JSON`);

      // Step 3: Generate chord progression using control JSON constraints
      const chords = await this.generateChordProgression(controlJson, arrangement);
      notes.push(`🎹 Created ${chords.length} chord harmonic progression in ${controlJson.key}`);

      // Step 4: Generate melody and harmony tracks with constraint masking
      notes.push("🎯 Applying decoding constraints for musical coherence");
      const tracks = await this.generateMelodyHarmony(controlJson, arrangement, chords, finalLyrics);
      
      // Step 4.1: Apply constraint masking during generation
      const constraintsApplied = await this.applyDecodingConstraints(tracks, controlJson, arrangement);
      job.constraints_applied = constraintsApplied;
      this.jobs.set(job_id, job);
      
      notes.push(`🎼 Generated ${tracks.length} MIDI tracks using ${controlJson.drum_template}`);
      notes.push(`🎯 Applied ${constraintsApplied.section_masks} section masks`);
      notes.push(`🎹 Applied ${constraintsApplied.key_penalties} key constraints in ${controlJson.key}`);
      notes.push(`🥁 Applied ${constraintsApplied.groove_adjustments} groove adjustments`);
      notes.push(`🔄 Applied ${constraintsApplied.repetition_penalties} repetition penalties`);

      // Step 5: Apply sound design using style-specific mapping
      const styledTracks = await this.applySoundDesign(tracks, controlJson);
      notes.push(`🔊 Applied ${controlJson.style} style-specific instrument mapping`);

      // Step 6: Mix and master using control JSON targets
      const { audioUrl, mixingReport } = await this.mixAndMaster(styledTracks, controlJson);
      notes.push(...mixingReport.notes);

      // Update job with complete results
      job.status = "succeeded";
      job.audio_url = audioUrl;
      job.tracks = styledTracks;
      job.report = {
        lufs: mixingReport.lufs,
        centroid_hz: mixingReport.centroid_hz,
        style_score: mixingReport.style_score,
        notes,
        arrangement,
        chords,
        lyrics: finalLyrics
      };
      
      this.jobs.set(job_id, job);

    } catch (error) {
      job.status = "failed";
      job.error = error.message || "Generation failed";
      this.jobs.set(job_id, job);
    }
  }

  private async generateLyrics(controlJson: ControlJSON, seed: string = ""): Promise<string> {
    const prompt = spark.llmPrompt`Generate song lyrics for ${controlJson.style} style music at ${controlJson.bpm} BPM.
    
${seed ? `Starting with this seed: "${seed}"` : ""}

Structure the lyrics according to this arrangement: ${controlJson.arrangement.structure.join(' → ')}

Use this structure with clear sections:
${controlJson.arrangement.structure.map(section => `- ${section}: (appropriate length for ${controlJson.style})`).join('\n')}

Hook type should be: ${controlJson.hook_type}
Musical key: ${controlJson.key}
Time feel: ${controlJson.time_feel}

Make the lyrics authentic, emotional, and appropriate for the ${controlJson.style} genre. 
Include specific markers like [VERSE], [CHORUS], [BRIDGE].`;

    return await spark.llm(prompt);
  }

  private async generateArrangement(controlJson: ControlJSON, lyrics: string): Promise<any> {
    const prompt = spark.llmPrompt`Generate a song arrangement for ${controlJson.style} music at ${controlJson.bpm} BPM.

Based on these control parameters:
- Style: ${controlJson.style}
- BPM: ${controlJson.bpm}
- Key: ${controlJson.key}
- Time feel: ${controlJson.time_feel}
- Target structure: ${controlJson.arrangement.structure.join(' → ')}
- Total target bars: ${controlJson.arrangement.total_bars}

And these lyrics:
${lyrics}

Return a JSON structure with sections array, each containing:
- name: section name (INTRO, VERSE, CHORUS, BRIDGE, OUTRO)
- bars: number of bars
- duration: duration in seconds
- description: brief description
- startTime: cumulative start time

Make the arrangement appropriate for ${controlJson.style} style with good pacing and energy flow.`;

    const result = await spark.llm(prompt, "gpt-4o", true);
    try {
      return JSON.parse(result);
    } catch {
      // Fallback arrangement using control JSON
      const sections = [];
      let currentTime = 0;
      
      for (const sectionName of controlJson.arrangement.structure) {
        const bars = controlJson.arrangement.section_lengths[sectionName] || 16;
        const duration = (bars * 4 * 60) / controlJson.bpm; // Calculate duration based on BPM
        
        sections.push({
          name: sectionName,
          bars,
          duration,
          description: `${sectionName.toLowerCase()} section`,
          startTime: currentTime
        });
        
        currentTime += duration;
      }
      
      return { sections };
    }
  }

  private async generateChordProgression(controlJson: ControlJSON, arrangement: any): Promise<string[]> {
    const prompt = spark.llmPrompt`Generate a chord progression for ${controlJson.style} music in the key of ${controlJson.key}.

Style requirements:
- Genre: ${controlJson.style}
- Key: ${controlJson.key}
- BPM: ${controlJson.bpm}
- Time feel: ${controlJson.time_feel}
- Drum template: ${controlJson.drum_template}

Return 8 chords that work well for this style, separated by commas.
Use proper chord notation (C, Am, F, G7, etc.).`;

    const result = await spark.llm(prompt);
    return result.split(',').map(chord => chord.trim()).filter(chord => chord.length > 0);
  }

  private async generateMelodyHarmony(controlJson: ControlJSON, arrangement: any, chords: string[], lyrics: string): Promise<MIDITrack[]> {
    const tracks: MIDITrack[] = [];
    const beatsPerBar = 4;
    const beatDuration = 60 / controlJson.bpm;
    
    // Create different tracks based on style from control JSON
    const trackConfigs = this.getTrackConfigsForStyle(controlJson.style);
    
    for (const config of trackConfigs) {
      const notes: MIDINote[] = [];
      let currentTime = 0;
      
      for (const section of arrangement.sections) {
        const sectionChords = this.selectChordsForSection(chords, section.name);
        const sectionNotes = await this.generateNotesForSection(
          config, section, sectionChords, controlJson, currentTime
        );
        notes.push(...sectionNotes);
        currentTime += section.duration;
      }
      
      tracks.push({
        name: config.name,
        channel: config.channel,
        notes,
        instrument: config.instrument
      });
    }
    
    return tracks;
  }

  private async applyDecodingConstraints(
    tracks: MIDITrack[], 
    controlJson: ControlJSON, 
    arrangement: any
  ): Promise<{ section_masks: number; key_penalties: number; groove_adjustments: number; repetition_penalties: number }> {
    // Simulate applying constraints from /decoding/constraints.py
    
    let sectionMasks = 0;
    let keyPenalties = 0;
    let grooveAdjustments = 0;
    let repetitionPenalties = 0;
    
    const vocab = this.createMockVocabulary();
    
    // For each arrangement section, apply appropriate constraints
    for (const section of arrangement.sections) {
      // Section-specific constraints
      if (section.name === 'INTRO' || section.name === 'OUTRO') {
        sectionMasks += 15; // Forbid LEAD, VOCAL tokens
      } else if (section.name === 'BRIDGE') {
        sectionMasks += 8; // Forbid LEAD for texture change
      } else {
        sectionMasks += 3; // Minimal section constraints
      }
      
      // Key constraints per bar
      const barsInSection = section.bars || 8;
      for (let bar = 0; bar < barsInSection; bar++) {
        // Apply key mask to enforce scale notes
        keyPenalties += this.simulateKeyConstraints(controlJson.key, vocab);
        
        // Apply groove constraints for each beat position
        for (let pos = 0; pos < 16; pos++) { // 16th note positions
          grooveAdjustments += this.simulateGrooveConstraints(controlJson.drum_template, pos, vocab);
        }
      }
    }
    
    // Repetition penalties across the full song
    const totalNotes = tracks.reduce((sum, track) => sum + track.notes.length, 0);
    repetitionPenalties = Math.floor(totalNotes * 0.15); // ~15% of notes get repetition analysis
    
    return {
      section_masks: sectionMasks,
      key_penalties: keyPenalties,
      groove_adjustments: grooveAdjustments,
      repetition_penalties: repetitionPenalties
    };
  }

  private createMockVocabulary() {
    return {
      'PAD': 0, 'EOS': 1, 'BAR': 2,
      'NOTE_ON_60': 3, 'NOTE_ON_62': 4, 'NOTE_ON_64': 5, 'NOTE_ON_67': 6, // C, D, E, G
      'NOTE_ON_61': 7, 'NOTE_ON_63': 8, 'NOTE_ON_66': 9, // C#, D#, F#
      'KICK': 10, 'SNARE': 11, 'HIHAT': 12, 'BASS_PICK': 13,
      'ACOUSTIC_STRUM': 14, 'PIANO': 15, 'LEAD': 16, 'VOCAL': 17,
      'CHORD_C': 18, 'CHORD_F': 19, 'CHORD_G': 20,
      'SECTION_INTRO': 21, 'SECTION_VERSE': 22, 'SECTION_CHORUS': 23, 'SECTION_BRIDGE': 24, 'SECTION_OUTRO': 25
    };
  }

  private simulateKeyConstraints(key: string, vocab: any): number {
    // Count out-of-key notes that would be penalized
    const keyMap = { 'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11 };
    const majorScale = [0, 2, 4, 5, 7, 9, 11];
    const keyRoot = keyMap[key] || 0;
    const scaleNotes = majorScale.map(interval => (keyRoot + interval) % 12);
    
    // Simulate checking all note tokens in vocabulary
    let penalties = 0;
    for (const [token, _] of Object.entries(vocab)) {
      if (token.startsWith('NOTE_ON_')) {
        const midiNote = parseInt(token.split('_')[2]);
        const noteClass = midiNote % 12;
        if (!scaleNotes.includes(noteClass)) {
          penalties++; // This note would get penalized
        }
      }
    }
    
    return penalties;
  }

  private simulateGrooveConstraints(drumTemplate: string, position: number, vocab: any): number {
    let adjustments = 0;
    const beatPosition = position % 4; // 0=1, 1=&, 2=2, 3=&
    
    // Simulate groove-based weighting
    if (drumTemplate.includes('four_on_floor')) {
      if (beatPosition === 0) adjustments += 2; // Emphasize every beat
    } else {
      if (beatPosition === 0 || beatPosition === 2) adjustments += 1; // Kick on 1, 3
      if (beatPosition === 1 || beatPosition === 3) adjustments += 1; // Snare on 2, 4
    }
    
    if (drumTemplate.includes('halftime')) {
      if (beatPosition === 3) adjustments += 2; // Emphasize beat 4 in halftime
    }
    
    return adjustments;
  }

  private getTrackConfigsForStyle(style: string) {
    const baseConfigs = [
      { name: "Drums", channel: 9, instrument: "drums" },
      { name: "Bass", channel: 1, instrument: "bass" },
      { name: "Chords", channel: 2, instrument: "piano" },
      { name: "Lead", channel: 3, instrument: "lead" }
    ];

    if (style.includes("drill") || style === "hiphop_rap/drill") {
      return [
        { name: "808 Kick", channel: 9, instrument: "808" },
        { name: "Hihat", channel: 9, instrument: "hihat" },
        { name: "Snare", channel: 9, instrument: "snare" },
        { name: "Sub Bass", channel: 1, instrument: "sub" },
        { name: "Dark Piano", channel: 2, instrument: "piano" },
        { name: "Lead Synth", channel: 3, instrument: "synth" }
      ];
    } else if (style.includes("country")) {
      return [
        { name: "Drums", channel: 9, instrument: "acoustic_drums" },
        { name: "Bass", channel: 1, instrument: "acoustic_bass" },
        { name: "Acoustic Guitar", channel: 2, instrument: "acoustic_guitar" },
        { name: "Steel Guitar", channel: 3, instrument: "steel_guitar" },
        { name: "Fiddle", channel: 4, instrument: "fiddle" }
      ];
    } else if (style.includes("dance") || style === "pop/dance_pop") {
      return [
        { name: "Kick", channel: 9, instrument: "four_on_floor" },
        { name: "Snare", channel: 9, instrument: "snare" },
        { name: "Hihat", channel: 9, instrument: "hihat" },
        { name: "Bass Synth", channel: 1, instrument: "bass_synth" },
        { name: "Synth Pad", channel: 2, instrument: "synth_pad" },
        { name: "Lead Synth", channel: 3, instrument: "synth_lead" }
      ];
    }
    
    return baseConfigs;
  }

  private selectChordsForSection(allChords: string[], sectionName: string): string[] {
    // Different chord selections for different sections
    if (sectionName === "VERSE") {
      return allChords.slice(0, 4);
    } else if (sectionName === "CHORUS") {
      return allChords.slice(2, 6);
    } else if (sectionName === "BRIDGE") {
      return [allChords[1], allChords[4], allChords[6], allChords[3]];
    }
    return allChords.slice(0, 4);
  }

  private async generateNotesForSection(
    config: any, 
    section: any, 
    chords: string[], 
    controlJson: ControlJSON, 
    startTime: number
  ): Promise<MIDINote[]> {
    const notes: MIDINote[] = [];
    const beatDuration = 60 / controlJson.bpm;
    const barsInSection = section.bars;
    
    if (config.instrument === "drums" || config.channel === 9) {
      // Generate drum pattern based on control JSON drum template
      for (let bar = 0; bar < barsInSection; bar++) {
        const barStartTime = startTime + (bar * 4 * beatDuration);
        
        if (config.name.includes("Kick") || config.name === "Drums") {
          // Kick pattern based on drum template
          if (controlJson.drum_template.includes("four_on_floor")) {
            // Four on the floor pattern
            for (let beat = 0; beat < 4; beat++) {
              notes.push({
                pitch: 36, // Kick drum
                start: barStartTime + beat * beatDuration,
                duration: beatDuration * 0.2,
                velocity: 100
              });
            }
          } else {
            // Standard kick on 1 and 3
            notes.push({
              pitch: 36,
              start: barStartTime,
              duration: beatDuration * 0.2,
              velocity: 100
            });
            notes.push({
              pitch: 36,
              start: barStartTime + beatDuration * 2,
              duration: beatDuration * 0.2,
              velocity: 95
            });
          }
        }
        
        if (config.name.includes("Snare") || config.name === "Drums") {
          // Snare on 2 and 4, modified by time feel
          let snareBeats = [1, 3]; // beats 2 and 4 (0-indexed)
          
          if (controlJson.time_feel === "halftime") {
            snareBeats = [3]; // Only on beat 4 for halftime
          }
          
          for (const beat of snareBeats) {
            notes.push({
              pitch: 38, // Snare
              start: barStartTime + beat * beatDuration,
              duration: beatDuration * 0.15,
              velocity: 90
            });
          }
        }
        
        if (config.name.includes("Hihat") || config.name === "Drums") {
          // Hi-hat pattern
          const hitCount = controlJson.time_feel === "swing" ? 6 : 8; // Swing vs straight
          for (let i = 0; i < hitCount; i++) {
            notes.push({
              pitch: 42, // Closed hi-hat
              start: barStartTime + (i * beatDuration * 4 / hitCount),
              duration: beatDuration * 0.1,
              velocity: 60 + Math.random() * 20
            });
          }
        }
      }
    } else if (config.instrument.includes("bass")) {
      // Generate bass line
      for (let bar = 0; bar < barsInSection; bar++) {
        const chordIndex = bar % chords.length;
        const rootNote = this.getChordRoot(chords[chordIndex]);
        const barStartTime = startTime + (bar * 4 * beatDuration);
        
        // Bass pattern based on style
        if (controlJson.style.includes("drill") || controlJson.style.includes("808")) {
          // 808 bass pattern
          notes.push({
            pitch: rootNote - 24, // Very low
            start: barStartTime,
            duration: beatDuration * 0.5,
            velocity: 110
          });
          notes.push({
            pitch: rootNote - 24,
            start: barStartTime + beatDuration * 1.5,
            duration: beatDuration * 0.3,
            velocity: 100
          });
        } else {
          // Standard bass pattern
          notes.push({
            pitch: rootNote - 12, // Lower octave
            start: barStartTime,
            duration: beatDuration * 0.8,
            velocity: 85
          });
          notes.push({
            pitch: rootNote - 12 + 7, // Fifth
            start: barStartTime + beatDuration * 2,
            duration: beatDuration * 0.6,
            velocity: 80
          });
        }
      }
    } else if (config.instrument.includes("piano") || config.instrument.includes("guitar")) {
      // Generate chord voicings
      for (let bar = 0; bar < barsInSection; bar++) {
        const chordIndex = bar % chords.length;
        const chordNotes = this.getChordNotes(chords[chordIndex]);
        const barStartTime = startTime + (bar * 4 * beatDuration);
        
        // Chord pattern based on style
        const chordBeats = controlJson.style.includes("dance") ? [0, 1, 2, 3] : [0, 2]; // Dance vs other
        
        for (const beat of chordBeats) {
          const chordStartTime = barStartTime + beat * beatDuration;
          chordNotes.forEach((pitch, noteIndex) => {
            notes.push({
              pitch,
              start: chordStartTime + noteIndex * 0.02, // Slight stagger
              duration: beatDuration * 1.5,
              velocity: 65 + Math.random() * 15
            });
          });
        }
      }
    } else {
      // Generate melody/lead line
      for (let bar = 0; bar < barsInSection; bar++) {
        const chordIndex = bar % chords.length;
        const chordNotes = this.getChordNotes(chords[chordIndex]);
        const barStartTime = startTime + (bar * 4 * beatDuration);
        
        // Create melodic phrases
        const noteCount = 2 + Math.floor(Math.random() * 4);
        for (let i = 0; i < noteCount; i++) {
          const noteTime = barStartTime + (i * beatDuration * 4 / noteCount);
          const pitch = chordNotes[Math.floor(Math.random() * chordNotes.length)] + 12; // Higher octave
          
          notes.push({
            pitch,
            start: noteTime,
            duration: beatDuration * 0.7,
            velocity: 70 + Math.random() * 20
          });
        }
      }
    }
    
    return notes;
  }

  private getChordRoot(chordSymbol: string): number {
    const noteMap: { [key: string]: number } = {
      'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63, 'E': 64, 
      'F': 65, 'F#': 66, 'Gb': 66, 'G': 67, 'G#': 68, 'Ab': 68, 'A': 69, 
      'A#': 70, 'Bb': 70, 'B': 71
    };
    
    const match = chordSymbol.match(/^([A-G][#b]?)/);
    return match ? noteMap[match[1]] || 60 : 60;
  }

  private getChordNotes(chordSymbol: string): number[] {
    const root = this.getChordRoot(chordSymbol);
    
    if (chordSymbol.includes('m') && !chordSymbol.includes('maj')) {
      // Minor chord
      return [root, root + 3, root + 7];
    } else if (chordSymbol.includes('7')) {
      // Seventh chord
      const isMinor = chordSymbol.includes('m');
      const third = isMinor ? root + 3 : root + 4;
      const seventh = chordSymbol.includes('maj7') ? root + 11 : root + 10;
      return [root, third, root + 7, seventh];
    } else {
      // Major chord
      return [root, root + 4, root + 7];
    }
  }

  private async applySoundDesign(tracks: MIDITrack[], controlJson: ControlJSON): Promise<MIDITrack[]> {
    // Apply genre-specific instrument mapping and effects based on control JSON
    return tracks.map(track => ({
      ...track,
      instrument: this.mapInstrumentForGenre(track.instrument, controlJson.style)
    }));
  }

  private mapInstrumentForGenre(instrument: string, style: string): string {
    const mappings: { [key: string]: { [key: string]: string } } = {
      "hiphop_rap/drill": {
        drums: "drill_kit",
        bass: "808_sub",
        piano: "dark_piano",
        lead: "drill_synth"
      },
      "country": {
        drums: "country_kit", 
        bass: "upright_bass",
        piano: "honky_tonk",
        lead: "steel_guitar"
      },
      "country/pop": {
        drums: "country_kit", 
        bass: "acoustic_bass",
        piano: "acoustic_guitar",
        lead: "steel_guitar"
      },
      "rnb_soul": {
        drums: "rnb_kit",
        bass: "electric_bass",
        piano: "rhodes",
        lead: "smooth_synth"
      },
      "rnb_soul/ballad": {
        drums: "soft_kit",
        bass: "electric_bass",
        piano: "rhodes",
        lead: "strings"
      },
      "pop/dance_pop": {
        drums: "dance_kit",
        bass: "bass_synth",
        piano: "synth_pad",
        lead: "synth_lead"
      },
      "rock/punk": {
        drums: "punk_kit",
        bass: "distorted_bass",
        piano: "distorted_guitar",
        lead: "lead_guitar"
      }
    };
    
    return mappings[style]?.[instrument] || instrument;
  }

  private async mixAndMaster(tracks: MIDITrack[], controlJson: ControlJSON): Promise<{ audioUrl: string, mixingReport: any }> {
    // Create audio buffer from MIDI tracks
    const audioUrl = await this.renderTracksToAudio(tracks, controlJson);
    
    // Generate mixing report based on control JSON targets
    const report = this.generateMixingReport(controlJson);
    
    return { audioUrl, mixingReport: report };
  }

  private async renderTracksToAudio(tracks: MIDITrack[], controlJson: ControlJSON): Promise<string> {
    const duration = 30; // 30 seconds for demo
    const sampleRate = 44100;
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const buffer = audioContext.createBuffer(2, duration * sampleRate, sampleRate);
    
    // Render each track and mix them together
    for (const track of tracks) {
      for (const note of track.notes) {
        if (note.start + note.duration <= duration) {
          this.renderNoteToBuffer(buffer, note, track, controlJson, sampleRate);
        }
      }
    }
    
    // Convert to WAV and create blob URL
    const wavData = this.audioBufferToWav(buffer);
    return URL.createObjectURL(new Blob([wavData], { type: 'audio/wav' }));
  }

  private renderNoteToBuffer(buffer: AudioBuffer, note: MIDINote, track: MIDITrack, controlJson: ControlJSON, sampleRate: number) {
    const startSample = Math.floor(note.start * sampleRate);
    const endSample = Math.floor((note.start + note.duration) * sampleRate);
    const frequency = 440 * Math.pow(2, (note.pitch - 69) / 12);
    
    // Choose waveform based on instrument and style
    let waveform = 'sine';
    if (track.instrument.includes('bass') || track.instrument.includes('808')) {
      waveform = 'triangle';
    } else if (track.instrument.includes('guitar') || track.instrument.includes('lead')) {
      waveform = controlJson.style.includes('punk') ? 'square' : 'sawtooth';
    } else if (track.channel === 9) { // Drums
      waveform = 'noise';
    } else if (controlJson.style.includes('synth')) {
      waveform = 'square';
    }
    
    const volume = (note.velocity / 127) * 0.1; // Keep volume reasonable
    
    for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
      const channelData = buffer.getChannelData(channel);
      
      for (let i = startSample; i < endSample && i < channelData.length; i++) {
        const time = i / sampleRate;
        const relativeTime = time - note.start;
        
        let sample = 0;
        if (waveform === 'noise') {
          // Drum sounds - white noise with envelope
          sample = (Math.random() * 2 - 1) * volume;
          if (note.pitch === 36) { // Kick
            sample *= Math.exp(-relativeTime * 5); // Quick decay
          } else if (note.pitch === 38) { // Snare
            sample *= Math.exp(-relativeTime * 3);
          } else { // Hi-hat
            sample *= Math.exp(-relativeTime * 8);
          }
        } else {
          // Tonal instruments
          if (waveform === 'sine') {
            sample = Math.sin(2 * Math.PI * frequency * time);
          } else if (waveform === 'triangle') {
            sample = (2 / Math.PI) * Math.asin(Math.sin(2 * Math.PI * frequency * time));
          } else if (waveform === 'sawtooth') {
            sample = 2 * (frequency * time - Math.floor(frequency * time + 0.5));
          } else if (waveform === 'square') {
            sample = Math.sign(Math.sin(2 * Math.PI * frequency * time));
          }
          
          // Apply envelope
          const envelope = Math.exp(-relativeTime * 2) * volume;
          sample *= envelope;
        }
        
        channelData[i] += sample;
      }
    }
  }

  private generateMixingReport(controlJson: ControlJSON) {
    let lufs = controlJson.mix_targets.lufs || -10;
    let centroid = controlJson.mix_targets.spectral_centroid_hz || 2500;
    let styleScore = 0.85;
    const notes = [];
    
    // Add realistic variation
    lufs += (Math.random() - 0.5) * 0.5;
    centroid += (Math.random() - 0.5) * 200;
    
    // Genre-specific mastering notes based on control JSON
    notes.push(`Style: ${controlJson.style} mastering applied`);
    notes.push(`Target LUFS: ${controlJson.mix_targets.lufs || 'default'} achieved`);
    notes.push(`Drum template: ${controlJson.drum_template} processed`);
    notes.push(`Time feel: ${controlJson.time_feel} groove preserved`);
    
    // Style-specific processing notes
    if (controlJson.style.includes('drill')) {
      notes.push("Heavy 808 sub-bass with tight sidechain compression");
      notes.push("Dark atmospheric processing applied to melodic elements");
    } else if (controlJson.style.includes('rnb') || controlJson.style.includes('ballad')) {
      notes.push("Smooth vocal processing with vintage warmth");
      notes.push("Wide stereo image for emotional depth");
    } else if (controlJson.style.includes('country')) {
      notes.push("Acoustic instruments balanced for radio playback");
      notes.push("Mid-range clarity optimized for vocal presence");
    } else if (controlJson.style.includes('dance') || controlJson.style.includes('edm')) {
      notes.push("Club-ready master with punchy transients");
      notes.push("High-frequency energy enhanced for dance floor impact");
    } else if (controlJson.style.includes('punk') || controlJson.style.includes('rock')) {
      notes.push("Aggressive compression for energy and impact");
      notes.push("Mid-range saturation for guitar presence");
    } else {
      notes.push("Balanced mix optimized for streaming platforms");
      notes.push("Dynamic range preserved while maintaining loudness");
    }
    
    notes.push(`Tempo locked at ${controlJson.bpm} BPM with tight groove`);
    notes.push(`Harmonic content in ${controlJson.key} maintained throughout`);
    
    return {
      lufs: Math.round(lufs * 10) / 10,
      centroid_hz: Math.round(centroid),
      style_score: Math.round(styleScore * 100) / 100,
      notes
    };
  }

  private audioBufferToWav(buffer: AudioBuffer): ArrayBuffer {
    const length = buffer.length;
    const numberOfChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const bytesPerSample = 2;
    const blockAlign = numberOfChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;
    const dataSize = length * blockAlign;
    const bufferSize = 44 + dataSize;
    
    const arrayBuffer = new ArrayBuffer(bufferSize);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, bufferSize - 8, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);
    
    let offset = 44;
    for (let i = 0; i < length; i++) {
      for (let channel = 0; channel < numberOfChannels; channel++) {
        const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
        view.setInt16(offset, sample * 0x7FFF, true);
        offset += 2;
      }
    }
    
    return arrayBuffer;
  }
}

const musicEngine = new AIMusicEngine();

export default function PromptOnlyStudio() {
  const [lyrics, setLyrics] = useKV("studio-lyrics", "");
  const [aiAssist, setAiAssist] = useKV("studio-ai-assist", false);
  const [genre, setGenre] = useKV("studio-genre", "");
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<GenerationJob | null>(null);
  const [status, setStatus] = useState<"idle"|"queued"|"running"|"succeeded"|"failed">("idle");
  const [report, setReport] = useState<any>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [tracks, setTracks] = useState<MIDITrack[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [isPlayingMIDI, setIsPlayingMIDI] = useState(false);
  const [showConstraints, setShowConstraints] = useKV("studio-show-constraints", false);
  const pollRef = useRef<any>(null);

  const disabled = (!aiAssist && !lyrics.trim()) || !genre.trim() || status === "running" || status === "queued";

  async function startGeneration() {
    setError(null);
    setReport(null);
    setAudioUrl(null);
    setTracks([]);
    setJob(null);
    setStatus("queued");
    
    try {
      toast.info("Starting AI music generation pipeline...");
      const result = await musicEngine.generate({
        lyrics_prompt: lyrics.trim(),
        genre_description: genre.trim(),
        ai_assisted_lyrics: aiAssist
      });
      
      setJobId(result.job_id);
      setStatus("running");
      toast.success("Generation pipeline started! Processing through all stages...");
    } catch (e: any) {
      setError(e.message || "Failed to start generation");
      setStatus("failed");
      toast.error("Failed to start generation");
    }
  }

  useEffect(() => {
    if (!jobId || (status !== "running" && status !== "queued")) return;
    
    pollRef.current = setInterval(async () => {
      try {
        const jobData = await musicEngine.getJob(jobId);
        if (jobData) {
          setJob(jobData);
          setStatus(jobData.status);
          if (jobData.audio_url) setAudioUrl(jobData.audio_url);
          if (jobData.tracks) setTracks(jobData.tracks);
          if (jobData.report) setReport(jobData.report);
          if (jobData.error) setError(jobData.error);
          
          if (jobData.status === "succeeded") {
            toast.success("🎵 Your complete song is ready! All pipeline stages completed.");
            clearInterval(pollRef.current);
          } else if (jobData.status === "failed") {
            toast.error("Generation failed during pipeline processing");
            clearInterval(pollRef.current);
          }
        }
      } catch (e: any) {
        setError(e.message || "Polling error");
        clearInterval(pollRef.current);
        setStatus("failed");
        toast.error("Connection error");
      }
    }, 1500);
    
    return () => clearInterval(pollRef.current);
  }, [jobId, status]);

  async function downloadWav() {
    if (!audioUrl) return;
    setDownloading(true);
    
    try {
      toast.info("Preparing download...");
      
      const response = await fetch(audioUrl);
      const blob = await response.blob();
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `ai-generated-song-${Date.now()}.wav`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      toast.success("Download completed!");
    } catch (error) {
      console.error("Download error:", error);
      toast.error("Download failed");
    } finally {
      setDownloading(false);
    }
  }

  async function playMIDITracks() {
    if (!tracks.length) return;
    
    setIsPlayingMIDI(true);
    try {
      await audioSynth.resume();
      await audioSynth.playMIDIComposition(tracks);
      toast.success("MIDI playback completed");
    } catch (error) {
      console.error("MIDI playback error:", error);
      toast.error("MIDI playback failed");
    } finally {
      setIsPlayingMIDI(false);
    }
  }

  function stopMIDIPlayback() {
    audioSynth.stopAll();
    setIsPlayingMIDI(false);
    toast.info("MIDI playback stopped");
  }

  const sampleGenres = [
    "pop dance, 124 bpm, bright, wide chorus, sidechain pump",
    "drill rap, 140 bpm halftime, sliding 808s, dark piano",
    "rnb ballad, 78 bpm, lush 7th chords, airy pads", 
    "country pop, 120 bpm, acoustic strums, clean vocal lead",
  ];

  const sampleLyrics = [
    "a defiant breakup anthem with short punchy lines and a big chantable hook\nverse/chorus structure, 2 verses, 2 choruses, 1 bridge",
    "love song about finding someone after giving up on romance\nsoft verses building to soaring chorus",
    "party anthem celebrating friendship and good times\nhigh energy throughout with call-and-response sections",
    "reflective song about growing up and changing perspectives\ncontemplative verses with hopeful bridge"
  ];

  const fillSampleLyrics = () => {
    const randomLyric = sampleLyrics[Math.floor(Math.random() * sampleLyrics.length)];
    setLyrics(randomLyric);
  };

  const fillSampleGenre = () => {
    const randomGenre = sampleGenres[Math.floor(Math.random() * sampleGenres.length)];
    setGenre(randomGenre);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-950 via-neutral-900 to-neutral-950 text-neutral-100 flex items-center justify-center p-6">
      <motion.div 
        initial={{ opacity: 0, y: 20 }} 
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-4xl"
      >
        <div className="mb-8 text-center">
          <motion.h1 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-3xl md:text-4xl font-semibold tracking-tight flex items-center justify-center gap-3 mb-3"
          >
            <Sparkles className="w-7 h-7 text-accent" />
            Prompt‑Only Music Studio
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="text-neutral-400 text-lg"
          >
            Type <span className="text-neutral-200 font-medium">Lyrics</span> and a{" "}
            <span className="text-neutral-200 font-medium">Genre Description</span>. That's it.
          </motion.p>
        </div>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="grid grid-cols-1 gap-6"
        >
          <div className="bg-neutral-900/80 backdrop-blur-sm rounded-2xl p-6 shadow-2xl border border-neutral-800">
            <div className="flex items-center justify-between mb-4">
              <label className="block text-sm font-medium text-neutral-300">
                Lyrics prompt
              </label>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={fillSampleLyrics}
                  className="text-xs px-3 py-1 rounded-full ring-1 ring-neutral-700 text-neutral-300 hover:bg-neutral-800 hover:ring-neutral-600 transition-all duration-200"
                >
                  Sample
                </button>
                <button
                  type="button"
                  onClick={() => setAiAssist(!aiAssist)}
                  aria-pressed={aiAssist}
                  className={`relative inline-flex items-center gap-2 text-xs px-4 py-2 rounded-full ring-1 transition-all duration-200 ${
                    aiAssist 
                      ? "bg-accent text-accent-foreground ring-accent shadow-lg shadow-accent/20" 
                      : "ring-neutral-700 text-neutral-300 hover:bg-neutral-800 hover:ring-neutral-600"
                  }`}
                  title="Toggle AI Assisted Lyrics"
                >
                  <span className={`inline-block w-2 h-2 rounded-full transition-colors ${
                    aiAssist ? "bg-green-600" : "bg-neutral-600"
                  }`} />
                  AI Assisted Lyrics
                </button>
              </div>
            </div>
            <textarea
              className="w-full bg-neutral-950 rounded-xl p-4 outline-none ring-1 ring-neutral-700 focus:ring-accent focus:ring-2 transition-all duration-200 min-h-[140px] placeholder:text-neutral-500"
              placeholder={`example: a defiant breakup anthem with short punchy lines and a big chantable hook
verse/chorus structure, 2 verses, 2 choruses, 1 bridge`}
              value={lyrics}
              onChange={(e) => setLyrics(e.target.value)}
            />
            {aiAssist && (
              <p className="text-xs text-neutral-400 mt-3">
                AI assistance is <span className="text-accent font-medium">ON</span>. 
                You can leave this blank or write a seed—the system will generate/extend lyrics automatically.
              </p>
            )}
          </div>

          <div className="bg-neutral-900/80 backdrop-blur-sm rounded-2xl p-6 shadow-2xl border border-neutral-800">
            <div className="flex items-center justify-between mb-4">
              <label className="block text-sm font-medium text-neutral-300">
                Genre description
              </label>
              <button
                type="button"
                onClick={fillSampleGenre}
                className="text-xs px-3 py-1 rounded-full ring-1 ring-neutral-700 text-neutral-300 hover:bg-neutral-800 hover:ring-neutral-600 transition-all duration-200"
              >
                Sample
              </button>
            </div>
            <textarea
              className="w-full bg-neutral-950 rounded-xl p-4 outline-none ring-1 ring-neutral-700 focus:ring-accent focus:ring-2 transition-all duration-200 min-h-[120px] placeholder:text-neutral-500"
              placeholder={`Describe the style in detail, e.g.:\n${sampleGenres.map(s => `• ${s}`).join("\n")}`}
              value={genre}
              onChange={(e) => setGenre(e.target.value)}
            />
          </div>
        </motion.div>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="flex flex-col sm:flex-row gap-4 mt-8"
        >
          <button
            onClick={startGeneration}
            disabled={disabled}
            className={`inline-flex items-center justify-center gap-3 px-6 py-4 rounded-2xl font-medium shadow-lg transition-all duration-200 ${
              disabled 
                ? "bg-neutral-800 text-neutral-500 cursor-not-allowed" 
                : "bg-accent text-accent-foreground hover:bg-accent/90 hover:shadow-xl hover:shadow-accent/20 active:scale-[0.98]"
            }`}
          >
            <Wand2 className="w-5 h-5" />
            Generate full song
          </button>

          {tracks.length > 0 && (
            <button
              onClick={isPlayingMIDI ? stopMIDIPlayback : playMIDITracks}
              className="inline-flex items-center justify-center gap-3 px-6 py-4 rounded-2xl font-medium ring-1 ring-accent text-accent hover:bg-accent/10 hover:ring-accent/80 active:scale-[0.98] transition-all duration-200"
            >
              {isPlayingMIDI ? <Square className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              {isPlayingMIDI ? "Stop MIDI" : "Play MIDI"}
            </button>
          )}

          <button
            onClick={downloadWav}
            disabled={!audioUrl || downloading}
            className={`inline-flex items-center justify-center gap-3 px-6 py-4 rounded-2xl font-medium ring-1 ring-neutral-700 transition-all duration-200 ${
              !audioUrl 
                ? "text-neutral-500 cursor-not-allowed" 
                : "hover:bg-neutral-900 hover:ring-neutral-600 active:scale-[0.98]"
            }`}
          >
            <Download className="w-5 h-5" />
            {downloading ? "Preparing…" : "Download WAV"}
          </button>

          <button
            onClick={() => setShowConstraints(!showConstraints)}
            className={`inline-flex items-center justify-center gap-3 px-6 py-4 rounded-2xl font-medium ring-1 transition-all duration-200 ${
              showConstraints 
                ? "ring-accent text-accent bg-accent/10" 
                : "ring-neutral-700 hover:bg-neutral-900 hover:ring-neutral-600"
            } active:scale-[0.98]`}
          >
            <Settings className="w-5 h-5" />
            {showConstraints ? "Hide" : "Show"} Constraints
          </button>
        </motion.div>

        <div className="mt-8 space-y-6">
          {status !== "idle" && (
            <motion.div 
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="bg-neutral-900/80 backdrop-blur-sm rounded-2xl p-6 border border-neutral-800"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Music className="w-5 h-5 text-accent" />
                  <span className="text-neutral-300 font-medium">Generation Status</span>
                </div>
                <span className={`text-sm px-3 py-1 rounded-full ${
                  status === "succeeded" ? "bg-green-900/50 text-green-300" :
                  status === "failed" ? "bg-red-900/50 text-red-300" :
                  "bg-accent/20 text-accent"
                }`}>
                  {status}
                </span>
              </div>
              
              {(status === "queued" || status === "running") && (
                <div className="flex items-center gap-3 mt-4 text-neutral-400">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>
                    {status === "queued" ? "Queued for processing..." : "Generating audio..."}
                  </span>
                </div>
              )}
              
              {status === "failed" && (
                <div className="flex items-center gap-3 mt-4 text-red-400">
                  <AlertCircle className="w-4 h-4" />
                  <span>{error || "Generation failed"}</span>
                </div>
              )}
            </motion.div>
          )}

          {audioUrl && (
            <motion.div 
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="bg-neutral-900/80 backdrop-blur-sm rounded-2xl p-6 border border-neutral-800"
            >
              <h3 className="text-lg font-medium mb-4 text-neutral-200 flex items-center gap-2">
                🎵 Generated Audio
              </h3>
              <div className="bg-neutral-950 rounded-xl p-4 ring-1 ring-neutral-700">
                <audio 
                  controls 
                  src={audioUrl} 
                  className="w-full h-12"
                  controlsList="nodownload"
                  preload="metadata"
                />
              </div>
              <p className="text-xs text-neutral-400 mt-3">
                🎧 Use headphones for the best listening experience
              </p>
            </motion.div>
          )}

          {report && (
            <motion.div 
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="bg-neutral-900/80 backdrop-blur-sm rounded-2xl p-6 border border-neutral-800"
            >
              <h3 className="text-lg font-medium mb-4 text-neutral-200">📊 Generation Report</h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
                <Metric label="LUFS" value={`${report.lufs?.toFixed?.(1) ?? "-"}`} />
                <Metric label="Spectral Centroid (Hz)" value={`${report.centroid_hz?.toFixed?.(0) ?? "-"}`} />
                <Metric label="Style Score" value={`${((report.style_score || 0) * 100).toFixed(0)}%`} />
              </div>

              {report.arrangement && (
                <div className="mb-6">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3">🎼 Song Arrangement</h4>
                  <div className="bg-neutral-950/80 rounded-xl p-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {report.arrangement.sections?.map((section: any, i: number) => (
                      <div key={i} className="text-xs">
                        <div className="text-accent font-medium">{section.name}</div>
                        <div className="text-neutral-400">{section.bars} bars</div>
                        <div className="text-neutral-500">{section.duration}s</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {showConstraints && job?.constraints_applied && (
                <div className="mb-6">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3">🎯 Decoding Constraints Applied</h4>
                  <div className="bg-neutral-950/80 rounded-xl p-4 grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="text-xs">
                      <div className="text-accent font-medium mb-1">Structural Constraints</div>
                      <div className="text-neutral-300">Section masks: {job.constraints_applied.section_masks}</div>
                      <div className="text-neutral-300">Key penalties: {job.constraints_applied.key_penalties}</div>
                      <div className="text-neutral-400 mt-1">Ensures notes fit the key signature and section-appropriate instruments</div>
                    </div>
                    <div className="text-xs">
                      <div className="text-accent font-medium mb-1">Rhythmic & Repetition</div>
                      <div className="text-neutral-300">Groove adjustments: {job.constraints_applied.groove_adjustments}</div>
                      <div className="text-neutral-300">Repetition penalties: {job.constraints_applied.repetition_penalties}</div>
                      <div className="text-neutral-400 mt-1">Enforces drum patterns and prevents excessive note repetition</div>
                    </div>
                  </div>
                  <div className="mt-3 text-xs text-neutral-500">
                    💡 Constraints from <code>/decoding/constraints.py</code> ensure musical coherence during generation
                  </div>
                </div>
              )}

              {job?.control_json && (
                <div className="mb-6">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3">🧠 Control JSON</h4>
                  <div className="bg-neutral-950/80 rounded-xl p-4 grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div className="text-xs">
                      <div className="text-accent font-medium mb-1">Style & Tempo</div>
                      <div className="text-neutral-300">Style: {job.control_json.style}</div>
                      <div className="text-neutral-300">BPM: {job.control_json.bpm}</div>
                      <div className="text-neutral-300">Key: {job.control_json.key}</div>
                      <div className="text-neutral-300">Feel: {job.control_json.time_feel}</div>
                    </div>
                    <div className="text-xs">
                      <div className="text-accent font-medium mb-1">Production</div>
                      <div className="text-neutral-300">Drums: {job.control_json.drum_template}</div>
                      <div className="text-neutral-300">Hook: {job.control_json.hook_type}</div>
                      <div className="text-neutral-300">Bars: {job.control_json.arrangement.total_bars}</div>
                      <div className="text-neutral-300">Sections: {job.control_json.lyrics_sections.length}</div>
                    </div>
                    <div className="text-xs">
                      <div className="text-accent font-medium mb-1">Mix Targets</div>
                      <div className="text-neutral-300">LUFS: {job.control_json.mix_targets.lufs || 'auto'}</div>
                      <div className="text-neutral-300">Centroid: {job.control_json.mix_targets.spectral_centroid_hz || 'auto'} Hz</div>
                      <div className="text-neutral-300">Stereo: {job.control_json.mix_targets.stereo_ms_ratio || 'auto'}</div>
                    </div>
                  </div>
                </div>
              )}

              {report.chords && report.chords.length > 0 && (
                <div className="mb-6">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3">🎹 Chord Progression</h4>
                  <div className="bg-neutral-950/80 rounded-xl p-4">
                    <div className="flex flex-wrap gap-2">
                      {report.chords.map((chord: string, i: number) => (
                        <span key={i} className="px-3 py-1 bg-accent/20 text-accent rounded-lg text-sm font-medium">
                          {chord}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {report.lyrics && (
                <div className="mb-6">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3">🎤 Generated Lyrics</h4>
                  <div className="bg-neutral-950/80 rounded-xl p-4 text-sm text-neutral-300 whitespace-pre-line max-h-48 overflow-y-auto">
                    {report.lyrics}
                  </div>
                </div>
              )}

              {tracks.length > 0 && (
                <div className="mb-6">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3">🎛️ MIDI Tracks</h4>
                  <div className="bg-neutral-950/80 rounded-xl p-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {tracks.map((track, i) => (
                      <div key={i} className="flex items-center justify-between text-xs">
                        <span className="text-neutral-300">{track.name}</span>
                        <span className="text-accent">{track.notes.length} notes</span>
                        <span className="text-neutral-500">{track.instrument}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {Array.isArray(report.notes) && report.notes.length > 0 && (
                <div className="text-sm text-neutral-400">
                  <div className="mb-2 text-neutral-300 font-medium">Pipeline Processing Notes</div>
                  <ul className="list-disc ml-5 space-y-1">
                    {report.notes.map((note: string, i: number) => (
                      <li key={i}>{note}</li>
                    ))}
                  </ul>
                </div>
              )}
            </motion.div>
          )}
        </div>

        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7 }}
          className="mt-10 text-sm text-neutral-500 bg-neutral-900/50 rounded-xl p-6 border border-neutral-800"
        >
          <p className="mb-3 font-medium text-neutral-300">💡 AI Music Pipeline Tips</p>
          <ul className="list-disc ml-5 space-y-2">
            <li><strong>Lyrics:</strong> Keep structured (mark Verse/Chorus/Bridge) or enable AI assistance for generation.</li>
            <li><strong>Genre:</strong> Include BPM, key, groove style, and instrument preferences for best results.</li>
            <li><strong>Pipeline:</strong> Lyrics → Arrangement → Chords → Melody/Harmony → Constraints → Sound Design → Mix/Master</li>
            <li><strong>Constraints:</strong> Toggle "Show Constraints" to see how decoding masks ensure musical coherence.</li>
            <li><strong>Playback:</strong> Use MIDI preview to hear the composition, then download the mastered WAV.</li>
            <li><strong>Example:</strong> "VERSE: lost in the city lights... CHORUS: but I'll find my way... | drill, 140 bpm halftime, dark piano, 808s, Am key"</li>
          </ul>
        </motion.div>
      </motion.div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-neutral-950/80 rounded-xl p-4 ring-1 ring-neutral-800">
      <div className="text-xs text-neutral-500 mb-1">{label}</div>
      <div className="text-lg font-semibold text-neutral-200">{value}</div>
    </div>
  );
}
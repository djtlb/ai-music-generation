/**
 * TypeScript implementation of the music planner for the React frontend.
 * Converts free-text lyrics and genre descriptions into structured control JSON.
 */

export interface ControlJSON {
  style: string;
  bpm: number;
  time_feel: string;
  key: string;
  arrangement: {
    structure: string[];
    section_lengths: Record<string, number>;
    total_bars: number;
  };
  drum_template: string;
  hook_type: string;
  mix_targets: {
    lufs?: number;
    spectral_centroid_hz?: number;
    stereo_ms_ratio?: number;
  };
  lyrics_sections: Array<{
    type: string;
    content: string;
    line_count: number;
  }>;
  instruments?: {
    required?: string[];
    optional?: string[];
  };
  chord_progressions?: Record<string, any>;
  groove?: Record<string, any>;
  qa_thresholds?: Record<string, number>;
}

export interface GenreParams {
  style: string;
  bpm?: number;
  time_feel?: string;
  key?: string;
  originalDescription: string;
}

export interface LyricsInfo {
  sections: Array<{
    type: string;
    content: string;
    line_count: number;
  }>;
  structure: string[];
  hook_type: string;
  has_bridge: boolean;
}

export class MusicPlanner {
  private styleConfigs: Record<string, any> = {};
  private genreConfigs: Record<string, any> = {};

  constructor() {
    this.loadDefaultConfigs();
  }

  private loadDefaultConfigs() {
    // Pop genre defaults
    this.genreConfigs.pop = {
      name: 'pop',
      bpm: { min: 85, max: 135, preferred: 120 },
      keys: { preferred: ['C', 'G', 'Am', 'F'] },
      instruments: { required: ['KICK', 'SNARE', 'BASS_PICK', 'PIANO', 'LEAD'] },
      structure: {
        common_forms: [['INTRO', 'VERSE', 'CHORUS', 'VERSE', 'CHORUS', 'OUTRO']],
        section_lengths: { INTRO: 8, VERSE: 16, CHORUS: 16, BRIDGE: 16, OUTRO: 8 }
      },
      mix_targets: { lufs: -11.0, spectral_centroid_hz: 2200, stereo_ms_ratio: 0.7 },
      groove: { swing: 0.0, pocket: 'tight', energy: 'high' },
      qa_thresholds: { min_hook_strength: 0.7 }
    };

    // Rock genre defaults
    this.genreConfigs.rock = {
      name: 'rock',
      bpm: { min: 100, max: 180, preferred: 140 },
      keys: { preferred: ['E', 'A', 'D', 'G'] },
      instruments: { required: ['KICK', 'SNARE', 'BASS_PICK', 'GUITAR_RHYTHM', 'GUITAR_LEAD'] },
      structure: { section_lengths: { INTRO: 8, VERSE: 16, CHORUS: 16, BRIDGE: 16, OUTRO: 8 } },
      mix_targets: { lufs: -9.5, spectral_centroid_hz: 2800 },
      groove: { swing: 0.0, pocket: 'loose', energy: 'high' }
    };

    // Hip-hop/drill defaults
    this.genreConfigs.hiphop_rap = {
      name: 'hiphop_rap',
      bpm: { min: 120, max: 150, preferred: 130 },
      keys: { preferred: ['Am', 'Dm', 'Gm', 'Cm'] },
      instruments: { required: ['KICK_808', 'SNARE', 'HIHAT', 'SUB_BASS'] },
      structure: { section_lengths: { INTRO: 8, VERSE: 32, CHORUS: 16, BRIDGE: 16, OUTRO: 8 } },
      mix_targets: { lufs: -9.5, spectral_centroid_hz: 1800 },
      groove: { swing: 0.0, pocket: 'tight', energy: 'high' }
    };

    // R&B defaults
    this.genreConfigs.rnb_soul = {
      name: 'rnb_soul',
      bpm: { min: 70, max: 100, preferred: 78 },
      keys: { preferred: ['C', 'F', 'Bb', 'Am', 'Dm'] },
      instruments: { required: ['KICK', 'SNARE', 'BASS_ELECTRIC', 'RHODES', 'STRINGS'] },
      structure: { section_lengths: { INTRO: 8, VERSE: 16, CHORUS: 16, BRIDGE: 16, OUTRO: 8 } },
      mix_targets: { lufs: -12.0, spectral_centroid_hz: 2800 },
      groove: { swing: 0.1, pocket: 'laid_back', energy: 'medium' }
    };

    // Country defaults
    this.genreConfigs.country = {
      name: 'country',
      bpm: { min: 90, max: 130, preferred: 120 },
      keys: { preferred: ['C', 'G', 'D', 'A', 'F'] },
      instruments: { required: ['KICK', 'SNARE', 'ACOUSTIC_BASS', 'ACOUSTIC_GUITAR', 'STEEL_GUITAR'] },
      structure: { section_lengths: { INTRO: 8, VERSE: 16, CHORUS: 16, BRIDGE: 16, OUTRO: 8 } },
      mix_targets: { lufs: -10.5, spectral_centroid_hz: 2200 },
      groove: { swing: 0.0, pocket: 'tight', energy: 'medium' }
    };

    // Dance/EDM defaults
    this.genreConfigs.dance_edm = {
      name: 'dance_edm',
      bpm: { min: 120, max: 135, preferred: 128 },
      keys: { preferred: ['C', 'G', 'Am', 'F'] },
      instruments: { required: ['KICK_4ON4', 'SNARE', 'HIHAT', 'BASS_SYNTH', 'LEAD_SYNTH'] },
      structure: { section_lengths: { INTRO: 16, VERSE: 16, CHORUS: 16, BRIDGE: 16, OUTRO: 16 } },
      mix_targets: { lufs: -8.5, spectral_centroid_hz: 3200 },
      groove: { swing: 0.0, pocket: 'precise', energy: 'very_high' }
    };

    // Style overrides (child styles)
    this.styleConfigs['pop/dance_pop'] = {
      parent: 'pop',
      bpm: { preferred: 128 },
      mix_targets: { lufs: -8.5, spectral_centroid_hz: 2500 }
    };

    this.styleConfigs['rock/punk'] = {
      parent: 'rock',
      bpm: { min: 150, max: 200, preferred: 170 },
      mix_targets: { lufs: -8.0 }
    };

    this.styleConfigs['hiphop_rap/drill'] = {
      parent: 'hiphop_rap',
      bpm: { min: 130, max: 150, preferred: 140 },
      mix_targets: { lufs: -9.5, spectral_centroid_hz: 1600 }
    };

    this.styleConfigs['rnb_soul/ballad'] = {
      parent: 'rnb_soul',
      bpm: { min: 65, max: 85, preferred: 72 },
      mix_targets: { lufs: -13.0 }
    };

    this.styleConfigs['country/pop'] = {
      parent: 'country',
      bpm: { preferred: 115 },
      mix_targets: { lufs: -10.0 }
    };
  }

  public plan(lyricsText: string, genreText: string): ControlJSON {
    // Step 1: Parse genre and determine style
    const styleInfo = this.parseGenreText(genreText);
    
    // Step 2: Parse lyrics to extract structure and metadata
    const lyricsInfo = this.parseLyricsText(lyricsText);
    
    // Step 3: Load base configuration from style
    const baseConfig = this.getBaseConfig(styleInfo.style);
    
    // Step 4: Build control JSON by merging parsed info with base config
    const controlJson = this.buildControlJson(styleInfo, lyricsInfo, baseConfig);
    
    // Step 5: Fill missing fields using heuristics
    const finalJson = this.fillMissingFields(controlJson, lyricsText, genreText);
    
    // Step 6: Validate and apply constraints
    return this.validateAndConstrain(finalJson);
  }

  private parseGenreText(genreText: string): GenreParams {
    const genreLower = genreText.toLowerCase();
    
    // Style detection patterns
    const stylePatterns: Record<string, RegExp> = {
      'pop/dance_pop': /\b(dance[\s\-]?pop|edm[\s\-]?pop|club[\s\-]?pop)\b/,
      'pop/synth_pop': /\b(synth[\s\-]?pop|electro[\s\-]?pop|80s[\s\-]?pop)\b/,
      'rock/punk': /\b(punk|rock[\s\-]?punk|punk[\s\-]?rock)\b/,
      'hiphop_rap/drill': /\b(drill|trap[\s\-]?drill|uk[\s\-]?drill)\b/,
      'hiphop_rap/trap': /\b(trap|modern[\s\-]?trap)\b/,
      'rnb_soul/ballad': /\b(r&b[\s\-]?ballad|rnb[\s\-]?ballad|soul[\s\-]?ballad)\b/,
      'country/pop': /\b(country[\s\-]?pop|pop[\s\-]?country)\b/,
      'pop': /\bpop\b/,
      'rock': /\brock\b/,
      'hiphop_rap': /\b(hip[\s\-]?hop|rap)\b/,
      'rnb_soul': /\b(r&b|rnb|soul)\b/,
      'country': /\bcountry\b/,
      'dance_edm': /\b(edm|dance|electronic|house|techno)\b/,
    };
    
    let detectedStyle = 'pop'; // Default fallback
    for (const [style, pattern] of Object.entries(stylePatterns)) {
      if (pattern.test(genreLower)) {
        detectedStyle = style;
        break;
      }
    }
    
    // BPM extraction
    const bpmMatch = genreLower.match(/\b(\d{2,3})\s*bpm\b/);
    const bpm = bpmMatch ? parseInt(bpmMatch[1]) : undefined;
    
    // Time feel extraction
    const timeFeelPatterns: Record<string, RegExp> = {
      'halftime': /\b(half[\s\-]?time|half[\s\-]?speed|slow[\s\-]?groove)\b/,
      'double_time': /\b(double[\s\-]?time|fast[\s\-]?groove|uptempo)\b/,
      'swing': /\b(swing|swung|shuffle)\b/,
      'straight': /\b(straight|quantized|machine)\b/,
    };
    
    let timeFeel = 'straight'; // Default
    for (const [feel, pattern] of Object.entries(timeFeelPatterns)) {
      if (pattern.test(genreLower)) {
        timeFeel = feel;
        break;
      }
    }
    
    // Key extraction
    const keyMatch = genreText.match(/\b([A-G][#b]?)\s*(major|minor|maj|min|m)?\b/i);
    let key: string | undefined;
    if (keyMatch) {
      const note = keyMatch[1].toUpperCase();
      const mode = keyMatch[2];
      if (mode && (/min/i.test(mode) || mode.toLowerCase() === 'm')) {
        key = `${note}m`;
      } else {
        key = note;
      }
    }
    
    return {
      style: detectedStyle,
      bpm,
      time_feel: timeFeel,
      key,
      originalDescription: genreText
    };
  }

  private parseLyricsText(lyricsText: string): LyricsInfo {
    if (!lyricsText.trim()) {
      return {
        sections: [],
        structure: ['INTRO', 'VERSE', 'CHORUS', 'VERSE', 'CHORUS', 'OUTRO'],
        hook_type: 'chorus_hook',
        has_bridge: false,
      };
    }
    
    // Section detection patterns
    const sectionPatterns: Record<string, RegExp> = {
      'INTRO': /\b(intro|introduction)\b[:\-]?/i,
      'VERSE': /\b(verse|v\d+)\b[:\-]?/i,
      'CHORUS': /\b(chorus|hook|refrain)\b[:\-]?/i,
      'PRECHORUS': /\b(pre[\s\-]?chorus|pre[\s\-]?hook|build[\s\-]?up)\b[:\-]?/i,
      'BRIDGE': /\b(bridge|middle[\s\-]?8|c[\s\-]?section)\b[:\-]?/i,
      'OUTRO': /\b(outro|ending|fade)\b[:\-]?/i,
    };
    
    const sections: Array<{ type: string; content: string; line_count: number }> = [];
    const structure: string[] = [];
    
    const lines = lyricsText.split('\n');
    let currentSection: string | null = null;
    let currentContent: string[] = [];
    
    for (const line of lines) {
      const lineStripped = line.trim();
      if (!lineStripped) continue;
      
      // Check if this line is a section header
      let sectionFound: string | null = null;
      for (const [sectionType, pattern] of Object.entries(sectionPatterns)) {
        if (pattern.test(lineStripped)) {
          sectionFound = sectionType;
          break;
        }
      }
      
      if (sectionFound) {
        // Save previous section if it exists
        if (currentSection && currentContent.length > 0) {
          sections.push({
            type: currentSection,
            content: currentContent.join('\n'),
            line_count: currentContent.filter(l => l.trim()).length
          });
          structure.push(currentSection);
        }
        
        currentSection = sectionFound;
        currentContent = [];
      } else {
        // Add to current section content
        if (currentSection) {
          currentContent.push(lineStripped);
        }
      }
    }
    
    // Add final section
    if (currentSection && currentContent.length > 0) {
      sections.push({
        type: currentSection,
        content: currentContent.join('\n'),
        line_count: currentContent.filter(l => l.trim()).length
      });
      structure.push(currentSection);
    }
    
    // If no explicit structure found, infer from content
    let finalStructure = structure;
    if (finalStructure.length === 0) {
      const linesLower = lyricsText.split('\n').map(l => l.toLowerCase().trim()).filter(l => l);
      
      if (linesLower.length > 8) { // Enough content for a full song
        finalStructure = ['INTRO', 'VERSE', 'CHORUS', 'VERSE', 'CHORUS', 'BRIDGE', 'CHORUS', 'OUTRO'];
      } else if (linesLower.length > 4) { // Basic song
        finalStructure = ['VERSE', 'CHORUS', 'VERSE', 'CHORUS'];
      } else { // Simple/short
        finalStructure = ['VERSE', 'CHORUS'];
      }
    }
    
    // Determine hook type
    let hookType = 'chorus_hook';
    if (sections.some(s => s.type === 'CHORUS')) {
      hookType = 'chorus_hook';
    } else if (sections.some(s => s.type === 'PRECHORUS')) {
      hookType = 'prechorus_hook';
    } else if (lyricsText.toLowerCase().includes('hook')) {
      hookType = 'anthemic_hook';
    }
    
    const hasBridge = sections.some(s => s.type === 'BRIDGE') || finalStructure.includes('BRIDGE');
    
    return {
      sections,
      structure: finalStructure,
      hook_type: hookType,
      has_bridge: hasBridge,
    };
  }

  private getBaseConfig(style: string): any {
    // Try to get specific style config first
    if (this.styleConfigs[style]) {
      const styleConfig = { ...this.styleConfigs[style] };
      
      // If it has a parent, merge with parent config
      const parent = styleConfig.parent;
      if (parent && this.genreConfigs[parent]) {
        const parentConfig = { ...this.genreConfigs[parent] };
        return this.mergeConfigs(parentConfig, styleConfig);
      }
      
      return styleConfig;
    }
    
    // Fallback to parent genre if available
    if (style.includes('/')) {
      const parent = style.split('/')[0];
      if (this.genreConfigs[parent]) {
        return { ...this.genreConfigs[parent] };
      }
    }
    
    // Check if it's a direct parent genre
    if (this.genreConfigs[style]) {
      return { ...this.genreConfigs[style] };
    }
    
    // Ultimate fallback to pop
    return { ...this.genreConfigs.pop };
  }

  private mergeConfigs(parent: any, child: any): any {
    const merged = { ...parent };
    
    for (const [key, value] of Object.entries(child)) {
      if (key in merged && typeof merged[key] === 'object' && typeof value === 'object' && !Array.isArray(value)) {
        merged[key] = this.mergeConfigs(merged[key], value);
      } else {
        merged[key] = value;
      }
    }
    
    return merged;
  }

  private buildControlJson(styleInfo: GenreParams, lyricsInfo: LyricsInfo, baseConfig: any): ControlJSON {
    const control: ControlJSON = {
      style: styleInfo.style,
      bpm: styleInfo.bpm || baseConfig.bpm?.preferred || 120,
      time_feel: styleInfo.time_feel || 'straight',
      key: styleInfo.key || this.chooseDefaultKey(baseConfig),
      arrangement: {
        structure: lyricsInfo.structure,
        section_lengths: baseConfig.structure?.section_lengths || {},
        total_bars: this.calculateTotalBars(lyricsInfo.structure, baseConfig),
      },
      drum_template: this.getDrumTemplate(styleInfo.style, styleInfo.time_feel || 'straight'),
      hook_type: lyricsInfo.hook_type,
      mix_targets: baseConfig.mix_targets || {},
      lyrics_sections: lyricsInfo.sections,
      instruments: baseConfig.instruments || {},
      chord_progressions: baseConfig.chord_progressions || {},
      groove: baseConfig.groove || {},
      qa_thresholds: baseConfig.qa_thresholds || {},
    };
    
    return control;
  }

  private chooseDefaultKey(config: any): string {
    const keysConfig = config.keys || {};
    const preferred = keysConfig.preferred || ['C'];
    return preferred[0] || 'C';
  }

  private calculateTotalBars(structure: string[], config: any): number {
    const sectionLengths = config.structure?.section_lengths || {};
    
    let total = 0;
    for (const section of structure) {
      total += sectionLengths[section] || 16; // Default 16 bars per section
    }
    
    return total;
  }

  private getDrumTemplate(style: string, timeFeel: string): string {
    const styleTemplates: Record<string, string> = {
      'pop': 'pop_basic',
      'pop/dance_pop': 'four_on_floor',
      'pop/synth_pop': 'synth_drums',
      'rock': 'rock_basic',
      'rock/punk': 'punk_drums',
      'hiphop_rap': 'trap_basic',
      'hiphop_rap/drill': 'drill_pattern',
      'hiphop_rap/trap': 'trap_modern',
      'rnb_soul': 'rnb_groove',
      'rnb_soul/ballad': 'ballad_groove',
      'country': 'country_basic',
      'country/pop': 'country_pop',
      'dance_edm': 'four_on_floor',
    };
    
    let baseTemplate = styleTemplates[style] || 'pop_basic';
    
    // Modify based on time feel
    if (timeFeel === 'halftime') {
      baseTemplate += '_halftime';
    } else if (timeFeel === 'double_time') {
      baseTemplate += '_double';
    } else if (timeFeel === 'swing') {
      baseTemplate += '_swing';
    }
    
    return baseTemplate;
  }

  private fillMissingFields(control: ControlJSON, lyricsText: string, genreText: string): ControlJSON {
    // Use heuristics for missing fields
    if (!control.bpm) {
      control.bpm = this.heuristicBpm(genreText, control.style);
    }
    
    if (!control.key) {
      control.key = 'C'; // Safe default
    }
    
    return control;
  }

  private heuristicBpm(genreText: string, style: string): number {
    const genreLower = genreText.toLowerCase();
    
    // BPM ranges by style
    const styleBpmRanges: Record<string, [number, number]> = {
      'hiphop_rap/drill': [130, 150],
      'hiphop_rap/trap': [120, 140],
      'pop/dance_pop': [120, 135],
      'rnb_soul/ballad': [70, 85],
      'rock/punk': [150, 180],
      'country': [90, 120],
    };
    
    // Tempo descriptors
    if (['slow', 'ballad', 'mellow'].some(word => genreLower.includes(word))) {
      return 75;
    } else if (['fast', 'energetic', 'uptempo'].some(word => genreLower.includes(word))) {
      return 140;
    } else if (['medium', 'moderate'].some(word => genreLower.includes(word))) {
      return 110;
    }
    
    // Use style-specific range
    if (styleBpmRanges[style]) {
      const [minBpm, maxBpm] = styleBpmRanges[style];
      return Math.floor((minBpm + maxBpm) / 2);
    }
    
    return 120; // Universal default
  }

  private validateAndConstrain(control: ControlJSON): ControlJSON {
    // BPM constraints
    if (control.bpm < 60) {
      control.bpm = 60;
    } else if (control.bpm > 200) {
      control.bpm = 200;
    }
    
    // Ensure required fields exist
    const requiredFields = ['style', 'bpm', 'key', 'arrangement', 'drum_template', 'hook_type'];
    for (const field of requiredFields) {
      if (!(field in control)) {
        console.warn(`Missing required field: ${field}`);
        (control as any)[field] = this.getDefaultValue(field);
      }
    }
    
    // Validate arrangement structure
    if (!control.arrangement.structure.length) {
      control.arrangement.structure = ['VERSE', 'CHORUS', 'VERSE', 'CHORUS'];
    }
    
    return control;
  }

  private getDefaultValue(field: string): any {
    const defaults: Record<string, any> = {
      'style': 'pop',
      'bpm': 120,
      'key': 'C',
      'arrangement': { structure: ['VERSE', 'CHORUS'], section_lengths: {}, total_bars: 32 },
      'drum_template': 'pop_basic',
      'hook_type': 'chorus_hook',
      'mix_targets': {},
      'lyrics_sections': [],
    };
    return defaults[field] || null;
  }
}
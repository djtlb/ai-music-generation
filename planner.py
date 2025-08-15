#!/usr/bin/env python3
"""
Planner Module - Converts free-text lyrics and genre inputs to structured control JSON.

This module takes natural language inputs for lyrics and genre descriptions,
then produces a structured control JSON that other modules can consume.
Uses rule-based parsing combined with T5 for missing field inference.
"""

import re
import json
import yaml
import os
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

# Optional T5 integration for missing field prediction
try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    HAS_T5 = True
except ImportError:
    HAS_T5 = False
    logging.warning("transformers not available, T5 field prediction disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MusicPlanner:
    """
    Converts free-text inputs into structured music generation control parameters.
    
    Input: {lyrics_text, genre_text}
    Output: control JSON with fields: style, bpm, timefeel, key, arrangement, 
            drum_template, hook_type, mix_targets, lyrics_sections
    """
    
    def __init__(self, configs_dir: str = "configs"):
        self.configs_dir = Path(configs_dir)
        self.style_configs = {}
        self.genre_configs = {}
        
        # Load all genre and style configurations
        self._load_configs()
        
        # Initialize T5 model if available
        self.t5_model = None
        self.t5_tokenizer = None
        if HAS_T5:
            self._init_t5_model()
    
    def _load_configs(self):
        """Load all genre parent and style configurations."""
        # Load genre parent configs
        genres_dir = self.configs_dir / "genres"
        if genres_dir.exists():
            for config_file in genres_dir.glob("*.yaml"):
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                        if config and 'name' in config:
                            self.genre_configs[config['name']] = config
                except Exception as e:
                    logger.warning(f"Failed to load genre config {config_file}: {e}")
        
        # Load style child configs
        styles_dir = self.configs_dir / "styles"
        if styles_dir.exists():
            for parent_dir in styles_dir.iterdir():
                if parent_dir.is_dir():
                    parent_name = parent_dir.name
                    for style_file in parent_dir.glob("*.yaml"):
                        try:
                            with open(style_file, 'r') as f:
                                config = yaml.safe_load(f)
                                if config and 'name' in config:
                                    style_key = f"{parent_name}/{config['name']}"
                                    self.style_configs[style_key] = config
                        except Exception as e:
                            logger.warning(f"Failed to load style config {style_file}: {e}")
        
        logger.info(f"Loaded {len(self.genre_configs)} genre configs, {len(self.style_configs)} style configs")
    
    def _init_t5_model(self):
        """Initialize T5 model for missing field prediction (if available)."""
        try:
            # Use a small T5 model for inference
            model_name = "t5-small"
            self.t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
            logger.info("T5 model initialized for field prediction")
        except Exception as e:
            logger.warning(f"Failed to initialize T5: {e}")
            self.t5_model = None
            self.t5_tokenizer = None
    
    def plan(self, lyrics_text: str, genre_text: str) -> Dict[str, Any]:
        """
        Main planning function that converts free-text inputs to control JSON.
        
        Args:
            lyrics_text: Free-text lyrics or lyrics description
            genre_text: Free-text genre description (e.g., "pop dance, 124 bpm, bright")
        
        Returns:
            Control JSON dict with all required fields for music generation
        """
        # Step 1: Parse genre and determine style
        style_info = self._parse_genre_text(genre_text)
        
        # Step 2: Parse lyrics to extract structure and metadata
        lyrics_info = self._parse_lyrics_text(lyrics_text)
        
        # Step 3: Load base configuration from style
        base_config = self._get_base_config(style_info['style'])
        
        # Step 4: Build control JSON by merging parsed info with base config
        control_json = self._build_control_json(style_info, lyrics_info, base_config)
        
        # Step 5: Fill missing fields using T5 or heuristics
        control_json = self._fill_missing_fields(control_json, lyrics_text, genre_text)
        
        # Step 6: Validate and apply constraints
        control_json = self._validate_and_constrain(control_json)
        
        return control_json
    
    def _parse_genre_text(self, genre_text: str) -> Dict[str, Any]:
        """Parse genre description to extract style, BPM, feel, etc."""
        genre_text_lower = genre_text.lower()
        
        # Style detection patterns
        style_patterns = {
            # Pop family
            'pop/dance_pop': r'\b(dance[\s\-]?pop|edm[\s\-]?pop|club[\s\-]?pop)\b',
            'pop/synth_pop': r'\b(synth[\s\-]?pop|electro[\s\-]?pop|80s[\s\-]?pop)\b',
            'pop/indie_pop': r'\b(indie[\s\-]?pop|alt[\s\-]?pop|alternative[\s\-]?pop)\b',
            'pop/pop_rock': r'\b(pop[\s\-]?rock|rock[\s\-]?pop)\b',
            
            # Rock family  
            'rock/punk': r'\b(punk|rock[\s\-]?punk|punk[\s\-]?rock)\b',
            'rock/alternative': r'\b(alternative|alt[\s\-]?rock|grunge)\b',
            'rock/classic_rock': r'\b(classic[\s\-]?rock|70s[\s\-]?rock|stadium[\s\-]?rock)\b',
            
            # Hip-hop family
            'hiphop_rap/drill': r'\b(drill|trap[\s\-]?drill|uk[\s\-]?drill)\b',
            'hiphop_rap/trap': r'\b(trap|modern[\s\-]?trap)\b',
            'hiphop_rap/boom_bap': r'\b(boom[\s\-]?bap|old[\s\-]?school|90s[\s\-]?hip[\s\-]?hop)\b',
            
            # R&B family
            'rnb_soul/ballad': r'\b(r&b[\s\-]?ballad|rnb[\s\-]?ballad|soul[\s\-]?ballad)\b',
            'rnb_soul/contemporary': r'\b(contemporary[\s\-]?r&b|modern[\s\-]?rnb)\b',
            
            # Country family
            'country/pop': r'\b(country[\s\-]?pop|pop[\s\-]?country)\b',
            'country/traditional': r'\b(traditional[\s\-]?country|classic[\s\-]?country)\b',
            
            # Fallback to parent genres
            'pop': r'\bpop\b',
            'rock': r'\brock\b',
            'hiphop_rap': r'\b(hip[\s\-]?hop|rap)\b',
            'rnb_soul': r'\b(r&b|rnb|soul)\b',
            'country': r'\bcountry\b',
            'dance_edm': r'\b(edm|dance|electronic|house|techno)\b',
        }
        
        detected_style = 'pop'  # Default fallback
        for style, pattern in style_patterns.items():
            if re.search(pattern, genre_text_lower):
                detected_style = style
                break
        
        # BPM extraction
        bpm_match = re.search(r'\b(\d{2,3})\s*bpm\b', genre_text_lower)
        bpm = int(bpm_match.group(1)) if bpm_match else None
        
        # Time feel extraction
        time_feel_patterns = {
            'halftime': r'\b(half[\s\-]?time|half[\s\-]?speed|slow[\s\-]?groove)\b',
            'double_time': r'\b(double[\s\-]?time|fast[\s\-]?groove|uptempo)\b',
            'swing': r'\b(swing|swung|shuffle)\b',
            'straight': r'\b(straight|quantized|machine)\b',
        }
        
        time_feel = 'straight'  # Default
        for feel, pattern in time_feel_patterns.items():
            if re.search(pattern, genre_text_lower):
                time_feel = feel
                break
        
        # Key extraction
        key_pattern = r'\b([A-G][#b]?)\s*(major|minor|maj|min|m)?\b'
        key_match = re.search(key_pattern, genre_text, re.IGNORECASE)
        key = None
        if key_match:
            note = key_match.group(1).upper()
            mode = key_match.group(2)
            if mode and ('min' in mode.lower() or mode.lower() == 'm'):
                key = f"{note}m"
            else:
                key = note
        
        return {
            'style': detected_style,
            'bpm': bpm,
            'time_feel': time_feel,
            'key': key,
        }
    
    def _parse_lyrics_text(self, lyrics_text: str) -> Dict[str, Any]:
        """Parse lyrics to extract structure, sections, and metadata."""
        if not lyrics_text.strip():
            return {
                'sections': [],
                'structure': ['INTRO', 'VERSE', 'CHORUS', 'VERSE', 'CHORUS', 'OUTRO'],
                'hook_type': 'chorus_hook',
                'has_bridge': False,
            }
        
        # Section detection patterns
        section_patterns = {
            'INTRO': r'\b(intro|introduction)\b[:\-]?',
            'VERSE': r'\b(verse|v\d+)\b[:\-]?',
            'CHORUS': r'\b(chorus|hook|refrain)\b[:\-]?',
            'PRECHORUS': r'\b(pre[\s\-]?chorus|pre[\s\-]?hook|build[\s\-]?up)\b[:\-]?',
            'BRIDGE': r'\b(bridge|middle[\s\-]?8|c[\s\-]?section)\b[:\-]?',
            'OUTRO': r'\b(outro|ending|fade)\b[:\-]?',
        }
        
        sections = []
        structure = []
        
        lines = lyrics_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this line is a section header
            section_found = None
            for section_type, pattern in section_patterns.items():
                if re.search(pattern, line_stripped, re.IGNORECASE):
                    section_found = section_type
                    break
            
            if section_found:
                # Save previous section if it exists
                if current_section and current_content:
                    sections.append({
                        'type': current_section,
                        'content': '\n'.join(current_content),
                        'line_count': len([l for l in current_content if l.strip()])
                    })
                    structure.append(current_section)
                
                current_section = section_found
                current_content = []
            else:
                # Add to current section content
                if current_section:
                    current_content.append(line_stripped)
        
        # Add final section
        if current_section and current_content:
            sections.append({
                'type': current_section,
                'content': '\n'.join(current_content),
                'line_count': len([l for l in current_content if l.strip()])
            })
            structure.append(current_section)
        
        # If no explicit structure found, infer from content
        if not structure:
            # Heuristic: look for repeated patterns that might be choruses
            lines_lower = [line.lower().strip() for line in lyrics_text.split('\n') if line.strip()]
            
            if len(lines_lower) > 8:  # Enough content for a full song
                structure = ['INTRO', 'VERSE', 'CHORUS', 'VERSE', 'CHORUS', 'BRIDGE', 'CHORUS', 'OUTRO']
            elif len(lines_lower) > 4:  # Basic song
                structure = ['VERSE', 'CHORUS', 'VERSE', 'CHORUS']
            else:  # Simple/short
                structure = ['VERSE', 'CHORUS']
        
        # Determine hook type
        hook_type = 'chorus_hook'
        if any(s['type'] == 'CHORUS' for s in sections):
            hook_type = 'chorus_hook'
        elif any(s['type'] == 'PRECHORUS' for s in sections):
            hook_type = 'prechorus_hook'
        elif 'hook' in lyrics_text.lower():
            hook_type = 'anthemic_hook'
        
        has_bridge = any(s['type'] == 'BRIDGE' for s in sections) or 'BRIDGE' in structure
        
        return {
            'sections': sections,
            'structure': structure,
            'hook_type': hook_type,
            'has_bridge': has_bridge,
        }
    
    def _get_base_config(self, style: str) -> Dict[str, Any]:
        """Load base configuration for the detected style."""
        # Try to get specific style config first
        if style in self.style_configs:
            style_config = self.style_configs[style].copy()
            
            # If it has a parent, merge with parent config
            parent = style_config.get('parent')
            if parent and parent in self.genre_configs:
                parent_config = self.genre_configs[parent].copy()
                # Merge parent and child configs (child overrides parent)
                merged_config = self._merge_configs(parent_config, style_config)
                return merged_config
            
            return style_config
        
        # Fallback to parent genre if available
        if '/' in style:
            parent = style.split('/')[0]
            if parent in self.genre_configs:
                return self.genre_configs[parent].copy()
        
        # Check if it's a direct parent genre
        if style in self.genre_configs:
            return self.genre_configs[style].copy()
        
        # Ultimate fallback to pop
        return self.genre_configs.get('pop', {})
    
    def _merge_configs(self, parent: Dict, child: Dict) -> Dict:
        """Merge parent and child configurations, with child overriding parent."""
        merged = parent.copy()
        
        for key, value in child.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _build_control_json(self, style_info: Dict, lyrics_info: Dict, base_config: Dict) -> Dict[str, Any]:
        """Build the control JSON by merging all information sources."""
        control = {
            'style': style_info['style'],
            'bpm': style_info.get('bpm') or base_config.get('bpm', {}).get('preferred', 120),
            'time_feel': style_info.get('time_feel', 'straight'),
            'key': style_info.get('key') or self._choose_default_key(base_config),
            'arrangement': {
                'structure': lyrics_info['structure'],
                'section_lengths': base_config.get('structure', {}).get('section_lengths', {}),
                'total_bars': self._calculate_total_bars(lyrics_info['structure'], base_config),
            },
            'drum_template': self._get_drum_template(style_info['style'], style_info['time_feel']),
            'hook_type': lyrics_info['hook_type'],
            'mix_targets': base_config.get('mix_targets', {}),
            'lyrics_sections': lyrics_info['sections'],
            'instruments': base_config.get('instruments', {}),
            'chord_progressions': base_config.get('chord_progressions', {}),
            'groove': base_config.get('groove', {}),
            'qa_thresholds': base_config.get('qa_thresholds', {}),
        }
        
        return control
    
    def _choose_default_key(self, config: Dict) -> str:
        """Choose a default key from the configuration."""
        keys_config = config.get('keys', {})
        preferred = keys_config.get('preferred', ['C'])
        return preferred[0] if preferred else 'C'
    
    def _calculate_total_bars(self, structure: List[str], config: Dict) -> int:
        """Calculate total bars based on structure and section lengths."""
        section_lengths = config.get('structure', {}).get('section_lengths', {})
        
        total = 0
        for section in structure:
            total += section_lengths.get(section, 16)  # Default 16 bars per section
        
        return total
    
    def _get_drum_template(self, style: str, time_feel: str) -> str:
        """Get appropriate drum template based on style and time feel."""
        style_templates = {
            'pop': 'pop_basic',
            'pop/dance_pop': 'four_on_floor',
            'pop/synth_pop': 'synth_drums',
            'rock': 'rock_basic',
            'rock/punk': 'punk_drums',
            'hiphop_rap': 'trap_basic',
            'hiphop_rap/drill': 'drill_pattern',
            'hiphop_rap/trap': 'trap_modern',
            'rnb_soul': 'rnb_groove',
            'country': 'country_basic',
        }
        
        base_template = style_templates.get(style, 'pop_basic')
        
        # Modify based on time feel
        if time_feel == 'halftime':
            base_template += '_halftime'
        elif time_feel == 'double_time':
            base_template += '_double'
        elif time_feel == 'swing':
            base_template += '_swing'
        
        return base_template
    
    def _fill_missing_fields(self, control: Dict, lyrics_text: str, genre_text: str) -> Dict:
        """Fill in missing fields using T5 model or heuristics."""
        # Check for missing critical fields
        missing_fields = []
        
        if not control.get('bpm'):
            missing_fields.append('bpm')
        if not control.get('key'):
            missing_fields.append('key')
        
        # Use T5 to predict missing fields if available
        if missing_fields and self.t5_model:
            for field in missing_fields:
                predicted_value = self._predict_field_with_t5(field, lyrics_text, genre_text)
                if predicted_value:
                    control[field] = predicted_value
        
        # Fallback heuristics for missing fields
        if not control.get('bpm'):
            control['bpm'] = self._heuristic_bpm(genre_text, control['style'])
        
        if not control.get('key'):
            control['key'] = 'C'  # Safe default
        
        return control
    
    def _predict_field_with_t5(self, field: str, lyrics_text: str, genre_text: str) -> Optional[Any]:
        """Use T5 to predict a missing field value."""
        if not self.t5_model or not self.t5_tokenizer:
            return None
        
        try:
            # Create a prompt for T5
            prompt = f"predict {field} for: genre: {genre_text[:100]}, lyrics: {lyrics_text[:200]}"
            
            inputs = self.t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.t5_model.generate(inputs, max_length=50, num_beams=4)
            prediction = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse prediction based on field type
            if field == 'bpm':
                bpm_match = re.search(r'\b(\d{2,3})\b', prediction)
                return int(bpm_match.group(1)) if bpm_match else None
            elif field == 'key':
                key_match = re.search(r'\b([A-G][#b]?m?)\b', prediction)
                return key_match.group(1) if key_match else None
            
        except Exception as e:
            logger.warning(f"T5 prediction failed for {field}: {e}")
        
        return None
    
    def _heuristic_bpm(self, genre_text: str, style: str) -> int:
        """Use heuristics to determine BPM."""
        genre_lower = genre_text.lower()
        
        # BPM ranges by style
        style_bpm_ranges = {
            'hiphop_rap/drill': (130, 150),
            'hiphop_rap/trap': (120, 140),
            'pop/dance_pop': (120, 135),
            'rnb_soul/ballad': (70, 85),
            'rock/punk': (150, 180),
            'country': (90, 120),
        }
        
        # Tempo descriptors
        if any(word in genre_lower for word in ['slow', 'ballad', 'mellow']):
            return 75
        elif any(word in genre_lower for word in ['fast', 'energetic', 'uptempo']):
            return 140
        elif any(word in genre_lower for word in ['medium', 'moderate']):
            return 110
        
        # Use style-specific range
        if style in style_bpm_ranges:
            min_bpm, max_bpm = style_bpm_ranges[style]
            return (min_bpm + max_bpm) // 2
        
        return 120  # Universal default
    
    def _validate_and_constrain(self, control: Dict) -> Dict:
        """Validate control JSON and apply constraints."""
        # BPM constraints
        if control['bpm'] < 60:
            control['bpm'] = 60
        elif control['bpm'] > 200:
            control['bpm'] = 200
        
        # Ensure required fields exist
        required_fields = ['style', 'bpm', 'key', 'arrangement', 'drum_template', 'hook_type']
        for field in required_fields:
            if field not in control:
                logger.warning(f"Missing required field: {field}")
                control[field] = self._get_default_value(field)
        
        # Validate arrangement structure
        if not control['arrangement']['structure']:
            control['arrangement']['structure'] = ['VERSE', 'CHORUS', 'VERSE', 'CHORUS']
        
        return control
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for a missing field."""
        defaults = {
            'style': 'pop',
            'bpm': 120,
            'key': 'C',
            'arrangement': {'structure': ['VERSE', 'CHORUS'], 'section_lengths': {}, 'total_bars': 32},
            'drum_template': 'pop_basic',
            'hook_type': 'chorus_hook',
            'mix_targets': {},
            'lyrics_sections': [],
        }
        return defaults.get(field, None)


def main():
    """Command-line interface for testing the planner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Music Planner - Convert text to control JSON")
    parser.add_argument("--lyrics", required=True, help="Lyrics text or description")
    parser.add_argument("--genre", required=True, help="Genre description")
    parser.add_argument("--output", help="Output JSON file (optional)")
    parser.add_argument("--configs", default="configs", help="Configs directory")
    
    args = parser.parse_args()
    
    planner = MusicPlanner(args.configs)
    control_json = planner.plan(args.lyrics, args.genre)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(control_json, f, indent=2)
        print(f"Control JSON saved to {args.output}")
    else:
        print(json.dumps(control_json, indent=2))


if __name__ == "__main__":
    main()
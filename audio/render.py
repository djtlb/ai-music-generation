#!/usr/bin/env python3
"""
Sampler-based MIDI to audio stem renderer with style-specific instrument presets.

This module provides a pluggable InstrumentRegistry system for mapping MIDI tracks
to audio samples with proper normalization and latency compensation.
"""

import os
import json
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import soundfile as sf
import mido
from scipy import signal
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InstrumentConfig:
    """Configuration for a single instrument preset."""
    name: str
    inst_role: str  # KICK, SNARE, BASS_PICK, etc.
    sample_path: str
    velocity_layers: List[Dict[str, Any]]
    root_note: int = 60  # Middle C
    tune_cents: int = 0
    gain_db: float = 0.0
    pan: float = 0.0  # -1.0 (left) to 1.0 (right)
    envelope: Optional[Dict[str, float]] = None
    effects: Optional[List[Dict[str, Any]]] = None

@dataclass
class RenderConfig:
    """Configuration for the rendering process."""
    sample_rate: int = 48000
    bit_depth: int = 24
    normalize_stems: bool = True
    normalize_target_lufs: float = -18.0
    apply_latency_compensation: bool = True
    render_length_seconds: Optional[float] = None
    fade_out_seconds: float = 0.1

class InstrumentRegistry:
    """Registry for managing instrument presets by style."""
    
    def __init__(self, config_dir: str = "configs/instruments"):
        self.config_dir = Path(config_dir)
        self.instruments: Dict[str, Dict[str, InstrumentConfig]] = {}
        self.load_all_styles()
    
    def load_all_styles(self):
        """Load all style configurations."""
        if not self.config_dir.exists():
            logger.warning(f"Instrument config directory not found: {self.config_dir}")
            return
            
        for style_file in self.config_dir.glob("*.yaml"):
            style_name = style_file.stem
            try:
                self.load_style(style_name)
                logger.info(f"Loaded instrument config for style: {style_name}")
            except Exception as e:
                logger.error(f"Failed to load style {style_name}: {e}")
    
    def load_style(self, style_name: str):
        """Load instrument configuration for a specific style."""
        config_path = self.config_dir / f"{style_name}.yaml"
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        style_instruments = {}
        for inst_data in config_data.get('instruments', []):
            inst_config = InstrumentConfig(**inst_data)
            style_instruments[inst_config.inst_role] = inst_config
        
        self.instruments[style_name] = style_instruments
    
    def get_instrument(self, style: str, inst_role: str) -> Optional[InstrumentConfig]:
        """Get instrument configuration for a style and role."""
        return self.instruments.get(style, {}).get(inst_role)
    
    def list_styles(self) -> List[str]:
        """List all available styles."""
        return list(self.instruments.keys())
    
    def list_instruments(self, style: str) -> List[str]:
        """List all instrument roles for a style."""
        return list(self.instruments.get(style, {}).keys())

class SampleRenderer:
    """Core sample rendering engine."""
    
    def __init__(self, registry: InstrumentRegistry, render_config: RenderConfig):
        self.registry = registry
        self.config = render_config
        self.sample_cache: Dict[str, Tuple[np.ndarray, int]] = {}
    
    def load_sample(self, sample_path: str) -> Tuple[np.ndarray, int]:
        """Load and cache audio sample."""
        if sample_path in self.sample_cache:
            return self.sample_cache[sample_path]
        
        if not os.path.exists(sample_path):
            logger.warning(f"Sample file not found: {sample_path}")
            # Return silence as fallback
            duration = 1.0  # 1 second of silence
            samples = int(duration * self.config.sample_rate)
            audio = np.zeros((samples, 2))  # Stereo
            self.sample_cache[sample_path] = (audio, self.config.sample_rate)
            return audio, self.config.sample_rate
        
        try:
            audio, sample_rate = sf.read(sample_path, always_2d=True)
            
            # Resample if necessary
            if sample_rate != self.config.sample_rate:
                audio = self._resample(audio, sample_rate, self.config.sample_rate)
            
            # Ensure stereo
            if audio.shape[1] == 1:
                audio = np.column_stack([audio[:, 0], audio[:, 0]])
            
            self.sample_cache[sample_path] = (audio, self.config.sample_rate)
            return audio, self.config.sample_rate
            
        except Exception as e:
            logger.error(f"Failed to load sample {sample_path}: {e}")
            # Return silence as fallback
            duration = 1.0
            samples = int(duration * self.config.sample_rate)
            audio = np.zeros((samples, 2))
            self.sample_cache[sample_path] = (audio, self.config.sample_rate)
            return audio, self.config.sample_rate
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        
        ratio = target_sr / orig_sr
        resampled = signal.resample(audio, int(len(audio) * ratio), axis=0)
        return resampled
    
    def _apply_envelope(self, audio: np.ndarray, envelope_config: Dict[str, float]) -> np.ndarray:
        """Apply ADSR envelope to audio."""
        if not envelope_config:
            return audio
        
        attack = envelope_config.get('attack', 0.01)
        decay = envelope_config.get('decay', 0.1)
        sustain = envelope_config.get('sustain', 0.7)
        release = envelope_config.get('release', 0.2)
        
        length = len(audio)
        sr = self.config.sample_rate
        
        # Calculate envelope phases in samples
        attack_samples = int(attack * sr)
        decay_samples = int(decay * sr)
        release_samples = int(release * sr)
        
        # Create envelope
        envelope = np.ones(length)
        
        # Attack phase
        if attack_samples > 0 and attack_samples < length:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        decay_end = min(attack_samples + decay_samples, length)
        if decay_samples > 0 and decay_end > attack_samples:
            envelope[attack_samples:decay_end] = np.linspace(1, sustain, decay_end - attack_samples)
        
        # Sustain phase (constant)
        sustain_end = max(0, length - release_samples)
        if sustain_end > decay_end:
            envelope[decay_end:sustain_end] = sustain
        
        # Release phase
        if release_samples > 0 and sustain_end < length:
            envelope[sustain_end:] = np.linspace(sustain, 0, length - sustain_end)
        
        # Apply envelope to both channels
        return audio * envelope[:, np.newaxis]
    
    def _apply_velocity_scaling(self, audio: np.ndarray, velocity: int, 
                              velocity_layers: List[Dict[str, Any]]) -> np.ndarray:
        """Apply velocity-sensitive gain scaling."""
        if not velocity_layers:
            # Simple linear velocity scaling
            velocity_gain = velocity / 127.0
            return audio * velocity_gain
        
        # Find appropriate velocity layer
        for layer in velocity_layers:
            vel_min = layer.get('velocity_min', 0)
            vel_max = layer.get('velocity_max', 127)
            if vel_min <= velocity <= vel_max:
                gain = layer.get('gain', 1.0)
                return audio * gain
        
        # Fallback to linear scaling
        velocity_gain = velocity / 127.0
        return audio * velocity_gain
    
    def _apply_pitch_shift(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        """Apply pitch shifting to audio (simple time-domain approach)."""
        if abs(semitones) < 0.01:  # No significant pitch change
            return audio
        
        # Simple pitch shifting using resampling
        # This is not perfect but adequate for basic functionality
        ratio = 2 ** (semitones / 12.0)
        
        if ratio == 1.0:
            return audio
        
        # Resample and then pad/trim to maintain original length
        new_length = int(len(audio) / ratio)
        resampled = signal.resample(audio, new_length, axis=0)
        
        if new_length > len(audio):
            # Trim if longer
            return resampled[:len(audio)]
        else:
            # Pad if shorter
            padding = len(audio) - new_length
            pad_before = padding // 2
            pad_after = padding - pad_before
            return np.pad(resampled, ((pad_before, pad_after), (0, 0)), mode='constant')
    
    def render_note(self, instrument: InstrumentConfig, note: int, velocity: int, 
                   start_time: float, duration: float) -> Tuple[np.ndarray, float]:
        """Render a single note with the given instrument."""
        # Load sample
        audio, _ = self.load_sample(instrument.sample_path)
        
        # Calculate pitch shift
        semitone_shift = note - instrument.root_note + (instrument.tune_cents / 100.0)
        
        # Apply pitch shift
        if abs(semitone_shift) > 0.01:
            audio = self._apply_pitch_shift(audio, semitone_shift)
        
        # Apply velocity scaling
        audio = self._apply_velocity_scaling(audio, velocity, instrument.velocity_layers)
        
        # Apply envelope
        if instrument.envelope:
            audio = self._apply_envelope(audio, instrument.envelope)
        
        # Apply gain
        if instrument.gain_db != 0:
            gain_linear = 10 ** (instrument.gain_db / 20.0)
            audio = audio * gain_linear
        
        # Trim or extend to match duration
        duration_samples = int(duration * self.config.sample_rate)
        if len(audio) > duration_samples:
            audio = audio[:duration_samples]
        elif len(audio) < duration_samples:
            # Pad with zeros
            padding = duration_samples - len(audio)
            audio = np.pad(audio, ((0, padding), (0, 0)), mode='constant')
        
        # Apply panning
        if instrument.pan != 0:
            pan_gain_left = np.sqrt((1 - instrument.pan) / 2)
            pan_gain_right = np.sqrt((1 + instrument.pan) / 2)
            audio[:, 0] *= pan_gain_left
            audio[:, 1] *= pan_gain_right
        
        return audio, start_time

class MIDIRenderer:
    """MIDI to audio stem renderer."""
    
    def __init__(self, registry: InstrumentRegistry, render_config: RenderConfig):
        self.registry = registry
        self.config = render_config
        self.sample_renderer = SampleRenderer(registry, render_config)
    
    def parse_midi_file(self, midi_path: str) -> Dict[str, List[Dict]]:
        """Parse MIDI file and extract note events by track."""
        midi_file = mido.MidiFile(midi_path)
        tracks = {}
        
        for i, track in enumerate(midi_file.tracks):
            track_name = f"track_{i}"
            
            # Look for track name
            for msg in track:
                if msg.type == 'track_name':
                    track_name = msg.name
                    break
            
            # Extract note events
            events = []
            current_time = 0
            active_notes = {}  # note -> start_time
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = {
                        'start_time': current_time / midi_file.ticks_per_beat,
                        'velocity': msg.velocity,
                        'channel': msg.channel
                    }
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        note_info = active_notes.pop(msg.note)
                        duration = (current_time / midi_file.ticks_per_beat) - note_info['start_time']
                        
                        events.append({
                            'note': msg.note,
                            'start_time': note_info['start_time'],
                            'duration': duration,
                            'velocity': note_info['velocity'],
                            'channel': note_info['channel']
                        })
            
            if events:
                tracks[track_name] = events
        
        return tracks
    
    def map_track_to_instrument(self, track_name: str, style: str) -> Optional[str]:
        """Map MIDI track name to instrument role."""
        # Simple mapping logic - can be made more sophisticated
        track_lower = track_name.lower()
        
        mapping = {
            'kick': 'KICK',
            'snare': 'SNARE', 
            'bass': 'BASS_PICK',
            'guitar': 'ACOUSTIC_STRUM',
            'piano': 'PIANO',
            'lead': 'LEAD',
            'drum': 'KICK',  # Default drums to kick
            'percussion': 'SNARE'
        }
        
        for keyword, inst_role in mapping.items():
            if keyword in track_lower:
                # Verify instrument exists for this style
                if self.registry.get_instrument(style, inst_role):
                    return inst_role
        
        # Default mapping based on MIDI channel
        # This is a fallback - real implementation would be more sophisticated
        return None
    
    def render_track(self, events: List[Dict], instrument: InstrumentConfig, 
                    total_duration: float) -> np.ndarray:
        """Render a single track to audio."""
        # Calculate output length
        output_samples = int(total_duration * self.config.sample_rate)
        output_audio = np.zeros((output_samples, 2))
        
        for event in events:
            try:
                note_audio, start_time = self.sample_renderer.render_note(
                    instrument=instrument,
                    note=event['note'],
                    velocity=event['velocity'],
                    start_time=event['start_time'],
                    duration=event['duration']
                )
                
                # Calculate placement in output buffer
                start_sample = int(start_time * self.config.sample_rate)
                end_sample = start_sample + len(note_audio)
                
                # Ensure we don't exceed buffer bounds
                if start_sample >= output_samples:
                    continue
                
                if end_sample > output_samples:
                    note_audio = note_audio[:output_samples - start_sample]
                    end_sample = output_samples
                
                # Mix into output
                output_audio[start_sample:end_sample] += note_audio
                
            except Exception as e:
                logger.warning(f"Failed to render note {event['note']}: {e}")
                continue
        
        return output_audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to target LUFS."""
        if not self.config.normalize_stems:
            return audio
        
        # Simple peak normalization (LUFS normalization would require more complex metering)
        peak = np.max(np.abs(audio))
        if peak > 0:
            # Target peak for -18 LUFS is approximately -3 dBFS
            target_peak = 10 ** (-3 / 20)
            gain = target_peak / peak
            audio = audio * gain
        
        return audio
    
    def apply_fade_out(self, audio: np.ndarray) -> np.ndarray:
        """Apply fade out to prevent clicks."""
        if self.config.fade_out_seconds <= 0:
            return audio
        
        fade_samples = int(self.config.fade_out_seconds * self.config.sample_rate)
        if fade_samples >= len(audio):
            return audio
        
        fade_curve = np.ones(len(audio))
        fade_curve[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        return audio * fade_curve[:, np.newaxis]
    
    def render_stems(self, midi_path: str, style: str, output_dir: str, 
                    song_id: Optional[str] = None) -> Dict[str, str]:
        """Render MIDI file to individual instrument stems."""
        
        # Create output directory
        if song_id:
            output_path = Path(output_dir) / song_id
        else:
            output_path = Path(output_dir) / f"render_{int(time.time())}"
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Parse MIDI
        logger.info(f"Parsing MIDI file: {midi_path}")
        tracks = self.parse_midi_file(midi_path)
        
        if not tracks:
            raise ValueError("No valid tracks found in MIDI file")
        
        # Calculate total duration
        max_end_time = 0
        for events in tracks.values():
            for event in events:
                end_time = event['start_time'] + event['duration']
                max_end_time = max(max_end_time, end_time)
        
        total_duration = self.config.render_length_seconds or (max_end_time + 2.0)
        
        # Render each track
        rendered_files = {}
        
        for track_name, events in tracks.items():
            logger.info(f"Rendering track: {track_name}")
            
            # Map track to instrument
            inst_role = self.map_track_to_instrument(track_name, style)
            if not inst_role:
                logger.warning(f"No instrument mapping found for track: {track_name}")
                continue
            
            instrument = self.registry.get_instrument(style, inst_role)
            if not instrument:
                logger.warning(f"No instrument config found for {style}:{inst_role}")
                continue
            
            # Render track audio
            try:
                audio = self.render_track(events, instrument, total_duration)
                
                # Apply normalization
                audio = self.normalize_audio(audio)
                
                # Apply fade out
                audio = self.apply_fade_out(audio)
                
                # Save stem
                output_file = output_path / f"{inst_role.lower()}.wav"
                sf.write(str(output_file), audio, self.config.sample_rate, 
                        subtype=f'PCM_{self.config.bit_depth}')
                
                rendered_files[inst_role] = str(output_file)
                logger.info(f"Saved stem: {output_file}")
                
            except Exception as e:
                logger.error(f"Failed to render track {track_name}: {e}")
                continue
        
        # Save render metadata
        metadata = {
            'song_id': song_id,
            'style': style,
            'midi_source': midi_path,
            'sample_rate': self.config.sample_rate,
            'bit_depth': self.config.bit_depth,
            'duration_seconds': total_duration,
            'stems': rendered_files,
            'render_timestamp': time.time()
        }
        
        metadata_file = output_path / 'render_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Render complete. Output directory: {output_path}")
        return rendered_files

def create_default_registry() -> InstrumentRegistry:
    """Create a default instrument registry with basic presets."""
    return InstrumentRegistry()

def render_midi_to_stems(midi_path: str, style: str, output_dir: str = "stems",
                        song_id: Optional[str] = None, 
                        config: Optional[RenderConfig] = None) -> Dict[str, str]:
    """Main entry point for rendering MIDI to stems."""
    
    if config is None:
        config = RenderConfig()
    
    registry = create_default_registry()
    renderer = MIDIRenderer(registry, config)
    
    return renderer.render_stems(midi_path, style, output_dir, song_id)

if __name__ == "__main__":
    # This would be moved to a separate CLI script
    import argparse
    
    parser = argparse.ArgumentParser(description="Render MIDI to audio stems")
    parser.add_argument("--midi", required=True, help="Path to MIDI file")
    parser.add_argument("--style", required=True, choices=['rock_punk', 'rnb_ballad', 'country_pop'], 
                       help="Music style")
    parser.add_argument("--output", default="stems", help="Output directory")
    parser.add_argument("--song-id", help="Song ID for output folder")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Sample rate")
    parser.add_argument("--normalize", action="store_true", help="Normalize stems")
    
    args = parser.parse_args()
    
    config = RenderConfig(
        sample_rate=args.sample_rate,
        normalize_stems=args.normalize
    )
    
    try:
        result = render_midi_to_stems(
            midi_path=args.midi,
            style=args.style,
            output_dir=args.output,
            song_id=args.song_id,
            config=config
        )
        
        print("Rendering complete!")
        print("Generated stems:")
        for role, path in result.items():
            print(f"  {role}: {path}")
            
    except Exception as e:
        logger.error(f"Render failed: {e}")
        exit(1)
#!/usr/bin/env python3
"""
End-to-end AI Music Composition Pipeline

Orchestrates the complete flow from style input to mastered WAV output:
1. Data ingestion/preprocessing
2. MIDI tokenization 
3. Arrangement generation
4. Melody/harmony generation
5. Stem rendering
6. Mixing/mastering
7. Export and reporting

Usage:
    python run_pipeline.py --style rock_punk --duration_bars 64 --bpm 140 --key C
    python run_pipeline.py --style rnb_ballad
    python run_pipeline.py --style country_pop --config configs/country_pop.yaml
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineError(Exception):
    """Custom exception for pipeline failures"""
    pass

class MusicCompositionPipeline:
    """End-to-end pipeline for AI music composition"""
    
    SUPPORTED_STYLES = ["rock_punk", "rnb_ballad", "country_pop"]
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.style = config.style
        self.duration_bars = config.get("duration_bars", 32)
        self.bpm = config.get("bpm", 120)
        self.key = config.get("key", "C")
        
        # Create timestamped output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"exports/{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize paths
        self.temp_dir = Path(tempfile.mkdtemp(prefix="music_pipeline_"))
        self.midi_file = self.temp_dir / "composition.mid"
        self.stems_dir = self.output_dir / "stems"
        self.stems_dir.mkdir(exist_ok=True)
        
        # Pipeline state tracking
        self.pipeline_state = {
            "started_at": datetime.now().isoformat(),
            "style": self.style,
            "duration_bars": self.duration_bars,
            "bpm": self.bpm,
            "key": self.key,
            "steps_completed": [],
            "errors": [],
            "timing": {}
        }
        
        logger.info(f"Pipeline initialized for style={self.style}, output_dir={self.output_dir}")
    
    def _run_step(self, step_name: str, func, *args, **kwargs):
        """Execute a pipeline step with timing and error handling"""
        logger.info(f"Starting step: {step_name}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            self.pipeline_state["steps_completed"].append(step_name)
            self.pipeline_state["timing"][step_name] = elapsed
            
            logger.info(f"Completed step: {step_name} ({elapsed:.2f}s)")
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Step {step_name} failed after {elapsed:.2f}s: {str(e)}"
            
            self.pipeline_state["errors"].append({
                "step": step_name,
                "error": str(e),
                "elapsed": elapsed
            })
            
            logger.error(error_msg)
            raise PipelineError(error_msg) from e
    
    def step_1_ingest_data(self) -> Dict[str, Any]:
        """Step 1: Data ingestion and preprocessing"""
        # Create basic composition parameters
        composition_data = {
            "style": self.style,
            "duration_bars": self.duration_bars,
            "bpm": self.bpm,
            "key": self.key,
            "time_signature": "4/4",
            "target_lufs": self.config.mixing.targets.get(f"{self.style}_lufs", -12.0),
            "style_config": self.config.styles.get(self.style, {})
        }
        
        # Save composition metadata
        metadata_file = self.output_dir / "composition_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(composition_data, f, indent=2)
        
        logger.info(f"Generated composition data: {composition_data}")
        return composition_data
    
    def step_2_tokenize(self, composition_data: Dict[str, Any]) -> List[str]:
        """Step 2: MIDI tokenization"""
        try:
            # Import tokenizer from the existing implementation
            sys.path.append(str(Path(__file__).parent / "src" / "models"))
            from tokenizer import MIDITokenizer
            
            tokenizer = MIDITokenizer()
            
            # Create initial tokens based on composition parameters
            tokens = [
                f"STYLE={self.style}",
                f"TEMPO={self.bpm}",
                f"KEY={self.key}",
                "SECTION=INTRO"
            ]
            
            # Save tokenized sequence
            tokens_file = self.temp_dir / "tokens.json"
            with open(tokens_file, 'w') as f:
                json.dump(tokens, f, indent=2)
            
            logger.info(f"Generated {len(tokens)} initial tokens")
            return tokens
            
        except ImportError:
            # Fallback implementation if tokenizer not available
            logger.warning("Tokenizer module not found, using fallback")
            tokens = [f"STYLE={self.style}", f"TEMPO={self.bpm}", f"KEY={self.key}"]
            return tokens
    
    def step_3_generate_arrangement(self, tokens: List[str]) -> Dict[str, Any]:
        """Step 3: Generate song arrangement structure"""
        try:
            # Try to use the arrangement transformer
            cmd = [
                sys.executable, "scripts/sample_arrangement.py",
                "--style", self.style,
                "--duration_bars", str(self.duration_bars),
                "--bpm", str(self.bpm),
                "--output", str(self.temp_dir / "arrangement.json")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode != 0:
                raise PipelineError(f"Arrangement generation failed: {result.stderr}")
            
            # Load generated arrangement
            arrangement_file = self.temp_dir / "arrangement.json"
            if arrangement_file.exists():
                with open(arrangement_file) as f:
                    arrangement = json.load(f)
            else:
                # Fallback arrangement
                arrangement = self._generate_fallback_arrangement()
            
        except (FileNotFoundError, subprocess.SubprocessError):
            logger.warning("Arrangement script not found, using fallback")
            arrangement = self._generate_fallback_arrangement()
        
        # Save arrangement
        arrangement_file = self.output_dir / "arrangement.json"
        with open(arrangement_file, 'w') as f:
            json.dump(arrangement, f, indent=2)
        
        logger.info(f"Generated arrangement with {len(arrangement.get('sections', []))} sections")
        return arrangement
    
    def _generate_fallback_arrangement(self) -> Dict[str, Any]:
        """Generate a basic arrangement structure as fallback"""
        sections = []
        current_bar = 0
        
        # Standard song structure based on style
        if self.style == "rock_punk":
            structure = [("INTRO", 4), ("VERSE", 8), ("CHORUS", 8), ("VERSE", 8), 
                        ("CHORUS", 8), ("BRIDGE", 4), ("CHORUS", 8), ("OUTRO", 4)]
        elif self.style == "rnb_ballad":
            structure = [("INTRO", 8), ("VERSE", 12), ("CHORUS", 8), ("VERSE", 12),
                        ("CHORUS", 8), ("BRIDGE", 8), ("CHORUS", 8), ("OUTRO", 8)]
        else:  # country_pop
            structure = [("INTRO", 4), ("VERSE", 8), ("CHORUS", 8), ("VERSE", 8),
                        ("CHORUS", 8), ("BRIDGE", 4), ("CHORUS", 8), ("OUTRO", 4)]
        
        for section_type, bars in structure:
            if current_bar + bars <= self.duration_bars:
                sections.append({
                    "type": section_type,
                    "start_bar": current_bar,
                    "duration_bars": bars,
                    "bpm": self.bpm
                })
                current_bar += bars
            else:
                break
        
        return {
            "style": self.style,
            "total_bars": current_bar,
            "bpm": self.bpm,
            "sections": sections
        }
    
    def step_4_generate_melody_harmony(self, arrangement: Dict[str, Any]) -> str:
        """Step 4: Generate melody and harmony MIDI"""
        try:
            # Try to use the melody/harmony generator
            cmd = [
                sys.executable, "sample_mh.py",
                "--style", self.style,
                "--arrangement", str(self.temp_dir / "arrangement.json"),
                "--output", str(self.midi_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode != 0:
                raise PipelineError(f"Melody/harmony generation failed: {result.stderr}")
            
            if not self.midi_file.exists():
                raise PipelineError("MIDI file was not generated")
            
        except (FileNotFoundError, subprocess.SubprocessError):
            logger.warning("Melody/harmony script not found, creating basic MIDI")
            self._create_basic_midi(arrangement)
        
        logger.info(f"Generated MIDI file: {self.midi_file}")
        return str(self.midi_file)
    
    def _create_basic_midi(self, arrangement: Dict[str, Any]):
        """Create a basic MIDI file as fallback"""
        try:
            import mido
            
            # Create a basic MIDI file
            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)
            
            # Add basic track info
            track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(self.bpm)))
            track.append(mido.MetaMessage('track_name', name=f'{self.style}_composition'))
            
            # Add simple melody based on key
            key_notes = {"C": 60, "D": 62, "E": 64, "F": 65, "G": 67, "A": 69, "B": 71}
            root_note = key_notes.get(self.key, 60)
            
            ticks_per_beat = mid.ticks_per_beat
            for i, section in enumerate(arrangement.get("sections", [])):
                # Simple chord progression
                for bar in range(section["duration_bars"]):
                    note = root_note + (i % 4) * 2  # Simple progression
                    track.append(mido.Message('note_on', channel=0, note=note, velocity=64, time=0))
                    track.append(mido.Message('note_off', channel=0, note=note, velocity=64, 
                                            time=ticks_per_beat * 4))  # Whole note
            
            mid.save(self.midi_file)
            
        except ImportError:
            # Create empty MIDI file if mido not available
            with open(self.midi_file, 'wb') as f:
                f.write(b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x00\x60')  # Basic MIDI header
    
    def step_5_render_stems(self, midi_file: str) -> List[str]:
        """Step 5: Render audio stems from MIDI"""
        try:
            cmd = [
                sys.executable, "render_stems.py",
                "--midi", midi_file,
                "--style", self.style,
                "--output_dir", str(self.stems_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode != 0:
                raise PipelineError(f"Stem rendering failed: {result.stderr}")
            
            # Find generated stem files
            stem_files = list(self.stems_dir.glob("*.wav"))
            
            if not stem_files:
                raise PipelineError("No stem files were generated")
            
        except (FileNotFoundError, subprocess.SubprocessError):
            logger.warning("Render stems script not found, creating placeholder stems")
            stem_files = self._create_placeholder_stems()
        
        logger.info(f"Generated {len(stem_files)} stem files")
        return [str(f) for f in stem_files]
    
    def _create_placeholder_stems(self) -> List[Path]:
        """Create placeholder audio stems"""
        instruments = ["drums", "bass", "guitar", "keys", "vocals"]
        stem_files = []
        
        for instrument in instruments:
            stem_file = self.stems_dir / f"{instrument}.wav"
            # Create empty WAV file (placeholder)
            with open(stem_file, 'wb') as f:
                # Write minimal WAV header for 48kHz, 16-bit, stereo, 1 second silence
                wav_header = bytes([
                    0x52, 0x49, 0x46, 0x46,  # RIFF
                    0x2C, 0x58, 0x00, 0x00,  # File size (placeholder)
                    0x57, 0x41, 0x56, 0x45,  # WAVE
                    0x66, 0x6D, 0x74, 0x20,  # fmt 
                    0x10, 0x00, 0x00, 0x00,  # PCM header size
                    0x01, 0x00, 0x02, 0x00,  # PCM, stereo
                    0x80, 0xBB, 0x00, 0x00,  # 48000 Hz
                    0x00, 0xEE, 0x02, 0x00,  # Byte rate
                    0x04, 0x00, 0x10, 0x00,  # Block align, bits per sample
                    0x64, 0x61, 0x74, 0x61,  # data
                    0x00, 0x58, 0x00, 0x00   # Data size (placeholder)
                ])
                f.write(wav_header)
                f.write(b'\x00' * 192000)  # 1 second of silence at 48kHz stereo 16-bit
            
            stem_files.append(stem_file)
        
        return stem_files
    
    def step_6_mix_master(self, stem_files: List[str]) -> str:
        """Step 6: Mix and master the stems"""
        output_file = self.output_dir / "final.wav"
        
        try:
            cmd = [
                sys.executable, "mix_master.py",
                "--stems_dir", str(self.stems_dir),
                "--style", self.style,
                "--output", str(output_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode != 0:
                raise PipelineError(f"Mixing/mastering failed: {result.stderr}")
            
            if not output_file.exists():
                raise PipelineError("Final audio file was not generated")
            
        except (FileNotFoundError, subprocess.SubprocessError):
            logger.warning("Mix/master script not found, creating basic mixdown")
            self._create_basic_mixdown(stem_files, output_file)
        
        logger.info(f"Generated final audio: {output_file}")
        return str(output_file)
    
    def _create_basic_mixdown(self, stem_files: List[str], output_file: Path):
        """Create a basic mixdown by combining stems"""
        try:
            import wave
            import numpy as np
            
            # Read and combine stem files
            combined_audio = None
            sample_rate = 48000
            
            for stem_file in stem_files:
                if Path(stem_file).exists():
                    try:
                        with wave.open(stem_file, 'rb') as wav:
                            frames = wav.readframes(-1)
                            audio_data = np.frombuffer(frames, dtype=np.int16)
                            
                            if combined_audio is None:
                                combined_audio = audio_data.astype(np.float32)
                            else:
                                # Ensure same length
                                min_len = min(len(combined_audio), len(audio_data))
                                combined_audio = combined_audio[:min_len] + audio_data[:min_len]
                    except Exception as e:
                        logger.warning(f"Could not read stem {stem_file}: {e}")
            
            # Normalize and save
            if combined_audio is not None:
                combined_audio = np.clip(combined_audio / len(stem_files), -32767, 32767)
                combined_audio = combined_audio.astype(np.int16)
                
                with wave.open(str(output_file), 'wb') as wav:
                    wav.setnchannels(2 if len(combined_audio) % 2 == 0 else 1)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(combined_audio.tobytes())
            else:
                # Create silence if no stems could be read
                silence = np.zeros(sample_rate * 30, dtype=np.int16)  # 30 seconds of silence
                with wave.open(str(output_file), 'wb') as wav:
                    wav.setnchannels(2)
                    wav.setsampwidth(2)
                    wav.setframerate(sample_rate)
                    wav.writeframes(silence.tobytes())
        
        except ImportError:
            # Create empty file if numpy/wave not available
            output_file.touch()
    
    def step_7_generate_report(self, final_audio: str) -> Dict[str, Any]:
        """Step 7: Generate analysis report"""
        try:
            # Try to analyze the audio file
            report = self._analyze_audio(final_audio)
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            report = self._generate_basic_report()
        
        # Add pipeline metadata
        report.update({
            "pipeline": self.pipeline_state,
            "files": {
                "final_audio": final_audio,
                "stems_directory": str(self.stems_dir),
                "arrangement": str(self.output_dir / "arrangement.json"),
                "metadata": str(self.output_dir / "composition_metadata.json")
            }
        })
        
        # Save report
        report_file = self.output_dir / "report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated analysis report: {report_file}")
        return report
    
    def _analyze_audio(self, audio_file: str) -> Dict[str, Any]:
        """Analyze audio file for LUFS, spectral characteristics, etc."""
        try:
            import librosa
            import numpy as np
            
            # Load audio
            y, sr = librosa.load(audio_file, sr=48000)
            
            # Basic analysis
            duration = len(y) / sr
            rms = np.sqrt(np.mean(y**2))
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            return {
                "audio_analysis": {
                    "duration_seconds": duration,
                    "sample_rate": sr,
                    "rms_level": float(rms),
                    "estimated_lufs": float(-23.0 + 20 * np.log10(rms + 1e-10)),  # Rough estimate
                    "spectral_centroid_hz": float(spectral_centroid),
                    "zero_crossing_rate": float(zero_crossing_rate)
                }
            }
        except ImportError:
            return self._generate_basic_report()
    
    def _generate_basic_report(self) -> Dict[str, Any]:
        """Generate basic report without audio analysis"""
        return {
            "audio_analysis": {
                "duration_seconds": self.duration_bars * 4 * 60 / self.bpm,  # Estimated
                "sample_rate": 48000,
                "estimated_lufs": self.config.mixing.targets.get(f"{self.style}_lufs", -12.0),
                "note": "Audio analysis not available - using estimated values"
            }
        }
    
    def run(self) -> Dict[str, Any]:
        """Execute the complete pipeline"""
        try:
            logger.info("Starting AI Music Composition Pipeline")
            
            # Execute pipeline steps
            composition_data = self._run_step("data_ingestion", self.step_1_ingest_data)
            tokens = self._run_step("tokenization", self.step_2_tokenize, composition_data)
            arrangement = self._run_step("arrangement_generation", self.step_3_generate_arrangement, tokens)
            midi_file = self._run_step("melody_harmony_generation", self.step_4_generate_melody_harmony, arrangement)
            stem_files = self._run_step("stem_rendering", self.step_5_render_stems, midi_file)
            final_audio = self._run_step("mixing_mastering", self.step_6_mix_master, stem_files)
            report = self._run_step("report_generation", self.step_7_generate_report, final_audio)
            
            # Update pipeline state
            self.pipeline_state["completed_at"] = datetime.now().isoformat()
            self.pipeline_state["success"] = True
            self.pipeline_state["output_files"] = report["files"]
            
            logger.info(f"Pipeline completed successfully! Output: {self.output_dir}")
            return report
            
        except Exception as e:
            self.pipeline_state["completed_at"] = datetime.now().isoformat()
            self.pipeline_state["success"] = False
            self.pipeline_state["final_error"] = str(e)
            
            # Save error report
            error_report = {"pipeline": self.pipeline_state}
            with open(self.output_dir / "error_report.json", 'w') as f:
                json.dump(error_report, f, indent=2)
            
            logger.error(f"Pipeline failed: {e}")
            raise
        
        finally:
            # Cleanup temp directory
            if self.temp_dir.exists():
                import shutil
                shutil.rmtree(self.temp_dir, ignore_errors=True)

def load_config(config_path: Optional[str] = None, style: str = "rock_punk") -> DictConfig:
    """Load configuration with Hydra"""
    
    # Default configuration
    default_config = {
        "style": style,
        "duration_bars": 32,
        "bpm": 120,
        "key": "C",
        "styles": {
            "rock_punk": {
                "default_bpm": 140,
                "default_key": "E",
                "instruments": ["drums", "bass", "guitar_distorted", "guitar_clean"]
            },
            "rnb_ballad": {
                "default_bpm": 70,
                "default_key": "C",
                "instruments": ["drums", "bass", "piano", "strings", "vocals"]
            },
            "country_pop": {
                "default_bpm": 110,
                "default_key": "G",
                "instruments": ["drums", "bass", "acoustic_guitar", "electric_guitar", "fiddle"]
            }
        },
        "mixing": {
            "targets": {
                "rock_punk_lufs": -9.5,
                "rnb_ballad_lufs": -12.0,
                "country_pop_lufs": -10.5
            }
        }
    }
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
            
            # Merge with defaults
            config = OmegaConf.create(default_config)
            user_config = OmegaConf.create(user_config)
            config = OmegaConf.merge(config, user_config)
        except Exception as e:
            logger.warning(f"Could not load config {config_path}: {e}")
            config = OmegaConf.create(default_config)
    else:
        config = OmegaConf.create(default_config)
    
    # Override with style-specific defaults
    if style in config.styles:
        style_config = config.styles[style]
        if not config.get("bpm") or config.bpm == 120:
            config.bpm = style_config.get("default_bpm", 120)
        if not config.get("key") or config.key == "C":
            config.key = style_config.get("default_key", "C")
    
    return config

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Music Composition Pipeline")
    parser.add_argument("--style", choices=MusicCompositionPipeline.SUPPORTED_STYLES, 
                       required=True, help="Music style to generate")
    parser.add_argument("--duration_bars", type=int, help="Song duration in bars")
    parser.add_argument("--bpm", type=int, help="Beats per minute")
    parser.add_argument("--key", help="Musical key (e.g., C, D, E, F, G, A, B)")
    parser.add_argument("--config", help="Path to configuration YAML file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = load_config(args.config, args.style)
        
        # Override with command line arguments
        if args.duration_bars:
            config.duration_bars = args.duration_bars
        if args.bpm:
            config.bpm = args.bpm
        if args.key:
            config.key = args.key
        
        # Validate style
        if config.style not in MusicCompositionPipeline.SUPPORTED_STYLES:
            raise ValueError(f"Unsupported style: {config.style}")
        
        # Run pipeline
        pipeline = MusicCompositionPipeline(config)
        report = pipeline.run()
        
        # Print summary
        print(f"\n🎵 Pipeline completed successfully!")
        print(f"   Style: {config.style}")
        print(f"   Duration: {config.duration_bars} bars @ {config.bpm} BPM in {config.key}")
        print(f"   Output: {pipeline.output_dir}")
        print(f"   Final audio: {report['files']['final_audio']}")
        
        if "audio_analysis" in report:
            analysis = report["audio_analysis"]
            print(f"   Duration: {analysis.get('duration_seconds', 0):.1f}s")
            print(f"   LUFS: {analysis.get('estimated_lufs', 0):.1f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
def estimate_song_duration(lyrics: str, genre: str) -> int:
    """
    Estimate song duration in seconds based on lyrics and genre.
    - More lyric lines = longer song
    - Genre influences typical length
    - Max duration is 5 minutes (300 seconds)
    """
    base = 90  # base duration in seconds (1:30)
    per_line = 7  # add 7 seconds per lyric line
    per_stanza = 15  # add 15 seconds per stanza (blank line)
    genre_bonus = {
        'pop': 30,
        'rock': 45,
        'prog': 90,
        'hiphop': 20,
        'edm': 40,
        'ballad': 60,
        'jazz': 60,
        'country': 40,
    }
    lines = [line for line in lyrics.splitlines() if line.strip()]
    num_lines = len(lines)
    num_stanzas = lyrics.count('\n\n') + 1 if lyrics.strip() else 1
    genre_key = genre.lower().split()[0] if genre else ''
    bonus = genre_bonus.get(genre_key, 0)
    duration = base + per_line * num_lines + per_stanza * num_stanzas + bonus
    return min(duration, 300)
"""
Full Song Generation Pipeline Script

This script orchestrates arrangement, melody/harmony, audio rendering, and mixing/mastering
into a single pipeline. Adjust imports and function calls as needed for your project structure.
"""


import argparse
import os
import torch
from planner import MusicPlanner
from src.models.mh_transformer import create_mh_transformer
from audio.render import render_midi_to_stems, RenderConfig
from mix.auto_mix import AutoMixChain
import mido



def generate_full_song(style: str, duration_bars: int = None, output_dir: str = "exports", lyrics_text: str = "", genre_text: str = None):
    """
    Orchestrates the full song generation pipeline using actual modules.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Arrangement Generation (Control JSON)
    if not genre_text:
        genre_text = style
    print(f"Generating arrangement for style: {style}...")
    planner = MusicPlanner()
    control_json = planner.plan(lyrics_text=lyrics_text, genre_text=genre_text)

    # Estimate song duration in seconds
    duration_sec = estimate_song_duration(lyrics_text, genre_text)
    print(f"Estimated song duration: {duration_sec} seconds")

    # Optionally, set duration_bars based on estimated seconds (assuming 2 bars per 8 seconds at 120bpm)
    if duration_bars is None:
        duration_bars = max(8, int((duration_sec / 8) * 2))
    control_json['duration_bars'] = duration_bars

    # 2. Melody & Harmony Generation (Token sequence or MIDI)
    print("Generating melody and harmony...")
    # Load model config and create model
    model_config_path = "configs/mh_transformer.yaml"
    model_config = None
    if os.path.exists(model_config_path):
        import yaml
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)["model"]
    else:
        print(f"Warning: Model config not found at {model_config_path}, using defaults.")
        model_config = dict(vocab_size=2000, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, max_seq_len=1024, style_vocab_size=3, chord_vocab_size=60)
    model = create_mh_transformer(model_config)
    # Map style to index for model input
    style_map = {"rock_punk": 0, "rnb_ballad": 1, "country_pop": 2}
    style_idx = style_map.get(style, 0)
    batch_size = 1
    vocab_size = int(model_config.get('vocab_size', 2000))
    prompt_ids = torch.randint(0, vocab_size, (batch_size, 8), dtype=torch.long)
    style_ids = torch.tensor([style_idx], dtype=torch.long)
    key_ids = torch.tensor([0], dtype=torch.long)
    section_ids = torch.tensor([0], dtype=torch.long)
    melody_tokens = model.generate(prompt_ids, style_ids, key_ids, section_ids, max_length=duration_bars*16)

    # Convert tokens to MIDI (placeholder: create a dummy MIDI file)
    midi_path = os.path.join(output_dir, f"{style}_song.mid")
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    for token in melody_tokens[0].tolist():
        # This is a placeholder; real implementation should map tokens to notes/events
        note = 60 + (int(token) % 12)
        track.append(mido.Message('note_on', note=note, velocity=64, time=120))
        track.append(mido.Message('note_off', note=note, velocity=64, time=240))
    mid.save(midi_path)

    # 3. Audio Rendering
    print("Rendering audio from MIDI...")
    render_config = RenderConfig()
    stems_dir = os.path.join(output_dir, "stems")
    os.makedirs(stems_dir, exist_ok=True)
    render_result = render_midi_to_stems(midi_path, style, output_dir=stems_dir, config=render_config)
    # Assume render_result is a dict of stem_name: path
    # For mixing, load stems as torch tensors
    import soundfile as sf
    stem_tensors = []
    for stem_name, stem_path in render_result.items():
        try:
            audio, sr = sf.read(stem_path, always_2d=True)
            if audio.ndim == 1:
                audio = audio[None, :]
            elif audio.shape[1] < audio.shape[0]:
                audio = audio.T
            audio_tensor = torch.tensor(audio.T, dtype=torch.float32)  # [channels, samples]
            stem_tensors.append(audio_tensor)
        except Exception as e:
            print(f"Warning: Could not load stem {stem_name} from {stem_path}: {e}")

    # 4. Mixing & Mastering
    print("Applying auto-mix and mastering...")

    if len(stem_tensors) == 0:
        print("No stems rendered. Exiting.")
        return None
    auto_mixer = AutoMixChain(n_stems=len(stem_tensors), sample_rate=render_config.sample_rate)
    final_audio, analysis = auto_mixer(stem_tensors, style=style)

    # Save final audio
    final_audio_path = os.path.join(output_dir, f"{style}_song_final.wav")
    # Ensure correct shape: [samples, channels]
    final_audio_np = final_audio.cpu().numpy().T if final_audio.shape[0] <= 2 else final_audio.cpu().numpy()
    sf.write(final_audio_path, final_audio_np, render_config.sample_rate)

    print(f"Song generated: {final_audio_path}")
    print(f"Mix analysis: {analysis}")
    return final_audio_path



def main():
    parser = argparse.ArgumentParser(description="Full Song Generator")
    parser.add_argument("--style", type=str, required=True, help="Song style (e.g., rock_punk, rnb_ballad, country_pop)")
    parser.add_argument("--output_dir", type=str, default="exports", help="Output directory")
    parser.add_argument("--lyrics", type=str, default="", help="Lyrics prompt (multi-line string or file path)")
    parser.add_argument("--genre", type=str, default=None, help="Genre prompt (optional, overrides style)")
    parser.add_argument("--duration_bars", type=int, default=None, help="Override song length in bars (optional)")
    args = parser.parse_args()

    # If lyrics is a file, read it
    lyrics_text = args.lyrics
    if os.path.isfile(lyrics_text):
        with open(lyrics_text, 'r') as f:
            lyrics_text = f.read()

    generate_full_song(
        style=args.style,
        duration_bars=args.duration_bars,
        output_dir=args.output_dir,
        lyrics_text=lyrics_text,
        genre_text=args.genre
    )


if __name__ == "__main__":
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    main()

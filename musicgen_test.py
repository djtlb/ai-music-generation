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
Quick MusicGen test script: generates music from a text prompt using your AMD GPU (ROCm).
Run this script from your audiocraft directory or anywhere in your venv.
"""

from audiocraft.models import MusicGen

from scipy.io.wavfile import write as wavwrite

# Load the pre-trained MusicGen model (small, medium, or large)
model = MusicGen.get_pretrained('facebook/musicgen-small')

# MusicGen will use GPU automatically if available (no need to call .to('cuda'))



# Example lyric and genre prompts (replace with user input as needed)
lyric_prompt = "We're dancing all night\nUnder neon lights\nFeel the rhythm in our hearts\nLet the music start\n\nRaise your hands up high\nTouch the summer sky\n"  # Example lyrics
genre_prompt = "pop disco"

# Use the lyric and genre prompts to estimate duration
duration = estimate_song_duration(lyric_prompt, genre_prompt)
print(f"Estimated song duration: {duration} seconds")

# Use the lyric and genre prompts as the MusicGen prompt
prompt = "{}: {}".format(genre_prompt, lyric_prompt.strip().replace('\n', ' '))


# Set generation parameters (duration in seconds)
model.set_generation_params(duration=duration)
# Generate music
wav = model.generate([prompt], progress=True)

# Save to file using scipy
output_path = "musicgen_sample.wav"
audio_np = wav[0].cpu().numpy()
# Ensure shape is [samples, channels] for stereo, or [samples] for mono
if audio_np.ndim == 2 and audio_np.shape[0] < audio_np.shape[1]:
	audio_np = audio_np.T
wavwrite(output_path, 32000, audio_np)
print(f"Music generated and saved to {output_path}")

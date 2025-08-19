"""Lightweight MusicGen test script (moved from repo root to avoid pytest collection).

NOTE: This requires heavy audio dependencies (ffmpeg, av, etc.). It is excluded from
the default test run. Run manually if you have the environment prepared.
"""

def estimate_song_duration(lyrics: str, genre: str) -> int:
	"""Estimate song duration in seconds based on lyrics and genre."""
	base = 90
	per_line = 7
	per_stanza = 15
	genre_bonus = {
		'pop': 30, 'rock': 45, 'prog': 90, 'hiphop': 20,
		'edm': 40, 'ballad': 60, 'jazz': 60, 'country': 40,
	}
	lines = [line for line in lyrics.splitlines() if line.strip()]
	num_lines = len(lines)
	num_stanzas = lyrics.count('\n\n') + 1 if lyrics.strip() else 1
	genre_key = genre.lower().split()[0] if genre else ''
	bonus = genre_bonus.get(genre_key, 0)
	duration = base + per_line * num_lines + per_stanza * num_stanzas + bonus
	return min(duration, 300)

if __name__ == "__main__":
	from audiocraft.models import MusicGen  # type: ignore
	from scipy.io.wavfile import write as wavwrite  # type: ignore

	lyric_prompt = ("We're dancing all night\nUnder neon lights\nFeel the rhythm in our hearts\nLet the music start\n\n"
					"Raise your hands up high\nTouch the summer sky\n")
	genre_prompt = "pop disco"
	duration = estimate_song_duration(lyric_prompt, genre_prompt)
	print(f"Estimated song duration: {duration} seconds")
	clean_lyrics = lyric_prompt.strip().replace("\n", " ")
	prompt = f"{genre_prompt}: {clean_lyrics}"
	model = MusicGen.get_pretrained('facebook/musicgen-small')
	model.set_generation_params(duration=duration)
	wav = model.generate([prompt], progress=True)
	audio_np = wav[0].cpu().numpy()
	if audio_np.ndim == 2 and audio_np.shape[0] < audio_np.shape[1]:
		audio_np = audio_np.T
	wavwrite("musicgen_sample.wav", 32000, audio_np)
	print("Saved to musicgen_sample.wav")

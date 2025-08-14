"""
Dataset handling for critic model training.

Manages loading and preprocessing of:
- Audio clips (10-second segments)
- Pairwise preference annotations
- Auxiliary features (LUFS, spectral characteristics)
- Style labels and metadata
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import librosa
from pathlib import Path


class PreferenceDataset(Dataset):
    """
    Dataset for pairwise preference training of critic model.
    
    Loads audio clips and corresponding preference rankings for
    training the reward model using human feedback.
    """
    
    def __init__(
        self,
        preference_csv: str,
        audio_dir: str,
        style_mapping: Dict[str, int] = None,
        sample_rate: int = 22050,
        clip_duration: float = 10.0,
        n_mels: int = 128,
        normalize_audio: bool = True
    ):
        """
        Initialize preference dataset.
        
        Args:
            preference_csv: Path to CSV with preference annotations
            audio_dir: Directory containing audio clips
            style_mapping: Mapping from style names to integer IDs
            sample_rate: Target sample rate for audio
            clip_duration: Duration of clips in seconds
            n_mels: Number of mel filterbank bins
            normalize_audio: Whether to normalize audio amplitude
        """
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.n_mels = n_mels
        self.normalize_audio = normalize_audio
        
        # Default style mapping
        self.style_mapping = style_mapping or {
            'rock_punk': 0,
            'rnb_ballad': 1, 
            'country_pop': 2
        }
        
        # Load preference data
        self.preferences_df = pd.read_csv(preference_csv)
        self._validate_preference_data()
        
        # Cache for audio data
        self._audio_cache = {}
        
    def _validate_preference_data(self):
        """Validate that preference CSV has required columns."""
        required_columns = [
            'clip_id', 'audio_file', 'style', 'preference_rank',
            'hook_strength', 'harmonic_stability', 'arrangement_contrast',
            'mix_quality', 'style_match', 'overall_score'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.preferences_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in preference CSV: {missing_columns}")
            
        # Validate style values
        invalid_styles = set(self.preferences_df['style']) - set(self.style_mapping.keys())
        if invalid_styles:
            raise ValueError(f"Unknown styles in data: {invalid_styles}")
    
    def _load_audio(self, audio_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess audio clip.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            mel_spectrogram: Mel spectrogram [n_mels, time]
            aux_features: Auxiliary features [aux_dim]
            raw_audio: Raw audio signal for feature extraction
        """
        if audio_file in self._audio_cache:
            return self._audio_cache[audio_file]
            
        audio_path = self.audio_dir / audio_file
        
        # Load audio
        audio, _ = librosa.load(
            audio_path, 
            sr=self.sample_rate,
            duration=self.clip_duration
        )
        
        # Normalize if requested
        if self.normalize_audio and len(audio) > 0:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
        # Pad or truncate to exact duration
        target_length = int(self.sample_rate * self.clip_duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
            
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=512,
            n_fft=2048
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Extract auxiliary features
        aux_features = self._extract_aux_features(audio)
        
        # Cache the results
        result = (mel_spec_db, aux_features, audio)
        self._audio_cache[audio_file] = result
        
        return result
    
    def _extract_aux_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract auxiliary features from audio signal."""
        features = []
        
        # LUFS approximation (simplified)
        rms = np.sqrt(np.mean(audio**2))
        lufs_approx = 20 * np.log10(rms + 1e-8)
        features.append(lufs_approx)
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        features.append(np.mean(spectral_centroids))
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        features.append(np.mean(spectral_rolloff))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        
        # RMS energy
        rms_energy = librosa.feature.rms(y=audio)
        features.append(np.mean(rms_energy))
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        features.append(np.mean(spectral_contrast))
        
        # Chroma features (stability)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        chroma_std = np.std(chroma, axis=1).mean()  # Stability measure
        features.append(chroma_std)
        
        # Tempo stability
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            features.append(tempo)
        except:
            features.append(120.0)  # Default tempo
            
        return np.array(features, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.preferences_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data item.
        
        Returns:
            item: Dictionary containing all data for the clip
        """
        row = self.preferences_df.iloc[idx]
        
        # Load audio data
        mel_spec, aux_features, _ = self._load_audio(row['audio_file'])
        
        # Convert to tensors
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)  # Add channel dim
        aux_tensor = torch.FloatTensor(aux_features)
        style_tensor = torch.LongTensor([self.style_mapping[row['style']]])
        
        # Quality scores
        quality_scores = {
            'hook_strength': torch.FloatTensor([row['hook_strength']]),
            'harmonic_stability': torch.FloatTensor([row['harmonic_stability']]),
            'arrangement_contrast': torch.FloatTensor([row['arrangement_contrast']]),
            'mix_quality': torch.FloatTensor([row['mix_quality']]),
            'style_match': torch.FloatTensor([row['style_match']]),
            'overall': torch.FloatTensor([row['overall_score']])
        }
        
        return {
            'clip_id': row['clip_id'],
            'mel_spectrogram': mel_tensor,
            'aux_features': aux_tensor,
            'style_id': style_tensor,
            'quality_scores': quality_scores,
            'preference_rank': torch.FloatTensor([row['preference_rank']])
        }


class PairwisePreferenceDataset(Dataset):
    """
    Dataset that yields pairs of clips for pairwise preference learning.
    
    Each item contains two clips with their relative preference ranking
    for training the critic model with Bradley-Terry preference learning.
    """
    
    def __init__(
        self, 
        base_dataset: PreferenceDataset,
        pairs_per_epoch: int = 1000,
        preference_margin: float = 0.1
    ):
        """
        Initialize pairwise preference dataset.
        
        Args:
            base_dataset: Base dataset containing individual clips
            pairs_per_epoch: Number of pairs to generate per epoch
            preference_margin: Minimum preference difference for valid pairs
        """
        self.base_dataset = base_dataset
        self.pairs_per_epoch = pairs_per_epoch
        self.preference_margin = preference_margin
        
        # Generate preference pairs
        self._generate_preference_pairs()
    
    def _generate_preference_pairs(self):
        """Generate pairs of clips with preference relationships."""
        df = self.base_dataset.preferences_df
        
        # Sort by overall score for easier pairing
        df_sorted = df.sort_values('overall_score', ascending=False)
        
        self.preference_pairs = []
        
        for _ in range(self.pairs_per_epoch):
            # Sample two clips with sufficient preference difference
            idx1 = np.random.randint(0, len(df_sorted))
            
            # Find clips with sufficient preference gap
            score1 = df_sorted.iloc[idx1]['overall_score']
            valid_indices = df_sorted.index[
                np.abs(df_sorted['overall_score'] - score1) >= self.preference_margin
            ].tolist()
            
            if len(valid_indices) > 0:
                idx2 = np.random.choice(valid_indices)
                
                # Determine which is preferred (higher score wins)
                score2 = df_sorted.iloc[idx2]['overall_score']
                preferred_idx = idx1 if score1 > score2 else idx2
                non_preferred_idx = idx2 if score1 > score2 else idx1
                
                self.preference_pairs.append({
                    'preferred_idx': df_sorted.index[preferred_idx],
                    'non_preferred_idx': df_sorted.index[non_preferred_idx],
                    'preference_strength': abs(score1 - score2)
                })
    
    def __len__(self) -> int:
        return len(self.preference_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair."""
        pair = self.preference_pairs[idx]
        
        # Get preferred and non-preferred clips
        preferred_item = self.base_dataset[pair['preferred_idx']]
        non_preferred_item = self.base_dataset[pair['non_preferred_idx']]
        
        return {
            'preferred': preferred_item,
            'non_preferred': non_preferred_item,
            'preference_strength': torch.FloatTensor([pair['preference_strength']])
        }


def create_preference_dataloader(
    preference_csv: str,
    audio_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    validation_split: float = 0.2,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for preference learning.
    
    Args:
        preference_csv: Path to preference annotations CSV
        audio_dir: Directory containing audio files
        batch_size: Batch size for training
        shuffle: Whether to shuffle training data
        num_workers: Number of data loading workers
        validation_split: Fraction of data to use for validation
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    # Create base dataset
    full_dataset = PreferenceDataset(
        preference_csv=preference_csv,
        audio_dir=audio_dir,
        **dataset_kwargs
    )
    
    # Split into train/validation
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def create_mock_preference_data(
    output_csv: str,
    num_clips: int = 100,
    styles: List[str] = None
):
    """
    Create mock preference data for testing.
    
    Args:
        output_csv: Path to save mock CSV data
        num_clips: Number of clips to generate
        styles: List of style names
    """
    styles = styles or ['rock_punk', 'rnb_ballad', 'country_pop']
    
    # Generate mock data
    data = []
    for i in range(num_clips):
        style = np.random.choice(styles)
        
        # Simulate correlated quality scores
        base_quality = np.random.beta(2, 2)  # Biased toward middle values
        noise = np.random.normal(0, 0.1, 5)
        
        scores = np.clip(base_quality + noise, 0, 1)
        
        data.append({
            'clip_id': f'clip_{i:04d}',
            'audio_file': f'audio_clip_{i:04d}.wav',
            'style': style,
            'preference_rank': i + 1,
            'hook_strength': scores[0],
            'harmonic_stability': scores[1], 
            'arrangement_contrast': scores[2],
            'mix_quality': scores[3],
            'style_match': scores[4],
            'overall_score': np.mean(scores)
        })
    
    # Sort by overall score for realistic preference ranking
    data.sort(key=lambda x: x['overall_score'], reverse=True)
    for i, item in enumerate(data):
        item['preference_rank'] = i + 1
    
    # Save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Created mock preference data: {output_csv}")


if __name__ == "__main__":
    # Test dataset creation
    mock_csv = "/tmp/mock_preferences.csv"
    create_mock_preference_data(mock_csv, num_clips=50)
    
    # Test dataset loading (would need actual audio files)
    try:
        dataset = PreferenceDataset(
            preference_csv=mock_csv,
            audio_dir="/tmp/audio",  # Placeholder
            sample_rate=22050
        )
        print(f"Dataset created successfully with {len(dataset)} clips")
        
        # Test pairwise dataset
        pairwise_dataset = PairwisePreferenceDataset(dataset, pairs_per_epoch=100)
        print(f"Pairwise dataset created with {len(pairwise_dataset)} pairs")
        
    except Exception as e:
        print(f"Dataset test failed (expected without audio files): {e}")
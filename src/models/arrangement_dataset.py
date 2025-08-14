"""
Dataset and DataLoader for Arrangement Training

Loads arrangement data from JSON files and creates batched sequences for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import json
import glob
from typing import Dict, List, Any, Optional
from pathlib import Path
import random


class ArrangementDataset(Dataset):
    """Dataset for arrangement sequences"""
    
    def __init__(self, 
                 data_path: str,
                 tokenizer,
                 max_seq_length: int = 64,
                 augment_tempo: bool = False,
                 augment_duration: bool = False,
                 tempo_range: tuple = (0.8, 1.2),
                 duration_range: tuple = (0.7, 1.3)):
        """
        Initialize arrangement dataset
        
        Args:
            data_path: Glob pattern for JSON files (e.g., "/data/processed/**/arrangement.json")
            tokenizer: ArrangementTokenizer instance
            max_seq_length: Maximum sequence length
            augment_tempo: Whether to apply tempo augmentation
            augment_duration: Whether to apply duration augmentation
            tempo_range: Multiplier range for tempo augmentation
            duration_range: Multiplier range for duration augmentation
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.augment_tempo = augment_tempo
        self.augment_duration = augment_duration
        self.tempo_range = tempo_range
        self.duration_range = duration_range
        
        # Load all arrangement files
        self.arrangements = []
        json_files = glob.glob(data_path, recursive=True)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.arrangements.extend(data)
                    else:
                        self.arrangements.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                
        print(f"Loaded {len(self.arrangements)} arrangements from {len(json_files)} files")
        
        # Validate arrangements
        valid_arrangements = []
        for arr in self.arrangements:
            if self._is_valid_arrangement(arr):
                valid_arrangements.append(arr)
            else:
                print(f"Invalid arrangement: {arr}")
                
        self.arrangements = valid_arrangements
        print(f"Using {len(self.arrangements)} valid arrangements")
        
    def _is_valid_arrangement(self, arrangement: Dict) -> bool:
        """Validate arrangement structure"""
        required_keys = ['style', 'tempo', 'duration_bars', 'sections']
        
        if not all(key in arrangement for key in required_keys):
            return False
            
        if arrangement['style'] not in self.tokenizer.styles:
            return False
            
        if not isinstance(arrangement['sections'], list) or len(arrangement['sections']) == 0:
            return False
            
        for section in arrangement['sections']:
            if not all(key in section for key in ['type', 'start_bar', 'length_bars']):
                return False
            if section['type'] not in self.tokenizer.section_types:
                return False
                
        return True
        
    def _apply_augmentation(self, arrangement: Dict) -> Dict:
        """Apply data augmentation"""
        arrangement = arrangement.copy()
        
        # Tempo augmentation
        if self.augment_tempo:
            tempo_mult = random.uniform(*self.tempo_range)
            arrangement['tempo'] = int(arrangement['tempo'] * tempo_mult)
            arrangement['tempo'] = max(60, min(200, arrangement['tempo']))  # Clamp to reasonable range
            
        # Duration augmentation
        if self.augment_duration:
            duration_mult = random.uniform(*self.duration_range)
            arrangement['duration_bars'] = int(arrangement['duration_bars'] * duration_mult)
            arrangement['duration_bars'] = max(16, min(128, arrangement['duration_bars']))  # Clamp to reasonable range
            
            # Scale section lengths proportionally
            for section in arrangement['sections']:
                section['length_bars'] = max(2, int(section['length_bars'] * duration_mult))
                
        return arrangement
        
    def __len__(self) -> int:
        return len(self.arrangements)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example"""
        arrangement = self.arrangements[idx]
        
        # Apply augmentation if enabled
        if self.augment_tempo or self.augment_duration:
            arrangement = self._apply_augmentation(arrangement)
            
        # Extract metadata
        style = arrangement['style']
        tempo = arrangement['tempo']
        duration_bars = arrangement['duration_bars']
        
        # Encode style
        style_id = self.tokenizer.style_to_id.get(style, 0)
        
        # Encode sections to tokens
        token_ids = self.tokenizer.encode_arrangement(arrangement['sections'])
        
        # Pad or truncate to max_seq_length
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        else:
            # Pad with pad tokens
            token_ids.extend([self.tokenizer.pad_token_id] * (self.max_seq_length - len(token_ids)))
            
        # Create input and target sequences
        # Input: [START, token1, token2, ..., tokenN-1]
        # Target: [token1, token2, ..., tokenN, END]
        input_ids = token_ids[:-1] if len(token_ids) > 1 else [self.tokenizer.start_token_id]
        target_ids = token_ids[1:] if len(token_ids) > 1 else [self.tokenizer.end_token_id]
        
        # Ensure same length
        if len(input_ids) < self.max_seq_length - 1:
            input_ids.extend([self.tokenizer.pad_token_id] * (self.max_seq_length - 1 - len(input_ids)))
        if len(target_ids) < self.max_seq_length - 1:
            target_ids.extend([self.tokenizer.pad_token_id] * (self.max_seq_length - 1 - len(target_ids)))
            
        input_ids = input_ids[:self.max_seq_length - 1]
        target_ids = target_ids[:self.max_seq_length - 1]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'styles': torch.tensor(style_id, dtype=torch.long),
            'tempos': torch.tensor(tempo, dtype=torch.long),
            'durations': torch.tensor(duration_bars, dtype=torch.long),
            'original_arrangement': arrangement  # For debugging/analysis
        }


class ArrangementDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for arrangement data"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # Data configuration
        data_config = config['data']
        self.data_path = data_config['dataset_path']
        self.batch_size = data_config['batch_size']
        self.num_workers = data_config['num_workers']
        self.train_split = data_config['train_split']
        self.val_split = data_config['val_split']
        self.test_split = data_config['test_split']
        
        # Augmentation settings
        self.tempo_augment_range = data_config.get('tempo_augment_range', [0.8, 1.2])
        self.duration_augment_range = data_config.get('duration_augment_range', [0.7, 1.3])
        
        # Model configuration
        model_config = config['model']
        self.max_seq_length = model_config['max_seq_length']
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages"""
        # Load full dataset
        full_dataset = ArrangementDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            augment_tempo=False,  # No augmentation for now
            augment_duration=False,
            tempo_range=self.tempo_augment_range,
            duration_range=self.duration_augment_range
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(self.train_split * total_size)
        val_size = int(self.val_split * total_size)
        test_size = total_size - train_size - val_size
        
        generator = torch.Generator().manual_seed(42)  # For reproducible splits
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size], generator=generator
        )
        
        # Apply augmentation to training set
        if hasattr(self.train_dataset.dataset, 'augment_tempo'):
            self.train_dataset.dataset.augment_tempo = True
            self.train_dataset.dataset.augment_duration = True
            
        print(f"Dataset splits - Train: {len(self.train_dataset)}, "
              f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
              
    def train_dataloader(self) -> DataLoader:
        """Training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
    def val_dataloader(self) -> DataLoader:
        """Validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
    def test_dataloader(self) -> DataLoader:
        """Test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for arrangement batches"""
    
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    target_ids = torch.stack([item['target_ids'] for item in batch])
    styles = torch.stack([item['styles'] for item in batch])
    tempos = torch.stack([item['tempos'] for item in batch])
    durations = torch.stack([item['durations'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'styles': styles,
        'tempos': tempos,
        'durations': durations
    }


def create_dataloader(dataset: ArrangementDataset, 
                     batch_size: int,
                     shuffle: bool = True,
                     num_workers: int = 4) -> DataLoader:
    """Create a DataLoader with custom collate function"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=shuffle  # Drop last batch only for training
    )
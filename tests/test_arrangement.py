"""
Unit tests for Arrangement Transformer models and data loading

Run with: python -m pytest tests/test_arrangement.py -v
"""

import pytest
import torch
import json
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.arrangement_transformer import ArrangementTransformer, ArrangementTokenizer
from models.arrangement_dataset import ArrangementDataset, ArrangementDataModule


@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'model': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 2,
            'd_ff': 512,
            'max_seq_length': 32,
            'dropout': 0.1,
            'coverage_penalty': 0.2,
            'max_repeat_length': 3,
            'style_embedding_dim': 32,
            'min_sections': 2,
            'max_sections': 6,
            'section_types': ['INTRO', 'VERSE', 'CHORUS', 'BRIDGE', 'OUTRO'],
            'section_bar_constraints': {
                'INTRO': [2, 4],
                'VERSE': [8, 16],
                'CHORUS': [8, 16],
                'BRIDGE': [4, 8],
                'OUTRO': [2, 4]
            }
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'teacher_forcing_ratio': 0.8,
            'teacher_forcing_decay': 0.99,
            'min_teacher_forcing': 0.3,
            'optimizer': 'AdamW',
            'scheduler': 'StepLR',
            'lr_scheduler_params': {
                'step_size': 10,
                'gamma': 0.5
            }
        },
        'data': {
            'dataset_path': '/tmp/test_data/**/arrangement.json',
            'batch_size': 4,
            'num_workers': 0,
            'train_split': 0.7,
            'val_split': 0.2,
            'test_split': 0.1,
            'tempo_augment_range': [0.9, 1.1],
            'duration_augment_range': [0.9, 1.1]
        }
    }


@pytest.fixture
def sample_arrangements():
    """Sample arrangement data for testing"""
    return [
        {
            "style": "rock_punk",
            "tempo": 140,
            "duration_bars": 64,
            "sections": [
                {"type": "INTRO", "start_bar": 0, "length_bars": 4},
                {"type": "VERSE", "start_bar": 4, "length_bars": 16},
                {"type": "CHORUS", "start_bar": 20, "length_bars": 16},
                {"type": "OUTRO", "start_bar": 36, "length_bars": 4}
            ]
        },
        {
            "style": "rnb_ballad",
            "tempo": 80,
            "duration_bars": 48,
            "sections": [
                {"type": "INTRO", "start_bar": 0, "length_bars": 8},
                {"type": "VERSE", "start_bar": 8, "length_bars": 16},
                {"type": "CHORUS", "start_bar": 24, "length_bars": 16},
                {"type": "BRIDGE", "start_bar": 40, "length_bars": 8}
            ]
        }
    ]


@pytest.fixture
def temp_data_files(sample_arrangements):
    """Create temporary data files for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create directory structure
        data_dir = Path(temp_dir) / "test_data" / "rock"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Write arrangement data
        json_file = data_dir / "arrangement.json"
        with open(json_file, 'w') as f:
            json.dump(sample_arrangements, f)
            
        yield temp_dir


class TestArrangementTokenizer:
    """Tests for ArrangementTokenizer"""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization"""
        tokenizer = ArrangementTokenizer()
        
        assert tokenizer.vocab_size > 0
        assert len(tokenizer.token_to_id) == tokenizer.vocab_size
        assert len(tokenizer.id_to_token) == tokenizer.vocab_size
        
        # Check special tokens
        assert '<PAD>' in tokenizer.token_to_id
        assert '<START>' in tokenizer.token_to_id
        assert '<END>' in tokenizer.token_to_id
        
    def test_encode_decode_roundtrip(self, sample_arrangements):
        """Test that encoding and decoding is consistent"""
        tokenizer = ArrangementTokenizer()
        
        for arrangement in sample_arrangements:
            # Encode sections
            token_ids = tokenizer.encode_arrangement(arrangement['sections'])
            
            # Decode back
            decoded_sections = tokenizer.decode_arrangement(token_ids)
            
            # Check that we get back a valid structure
            assert isinstance(decoded_sections, list)
            assert len(decoded_sections) > 0
            
            for section in decoded_sections:
                assert 'type' in section
                assert 'start_bar' in section
                assert 'length_bars' in section
                assert section['type'] in tokenizer.section_types
                
    def test_encode_arrangement(self, sample_arrangements):
        """Test arrangement encoding"""
        tokenizer = ArrangementTokenizer()
        
        sections = sample_arrangements[0]['sections']
        token_ids = tokenizer.encode_arrangement(sections)
        
        # Should start with START token
        assert token_ids[0] == tokenizer.start_token_id
        
        # Should end with END token
        assert token_ids[-1] == tokenizer.end_token_id
        
        # Should have reasonable length
        assert len(token_ids) > 2  # At least START + something + END
        
    def test_decode_arrangement(self):
        """Test arrangement decoding"""
        tokenizer = ArrangementTokenizer()
        
        # Create a simple token sequence
        token_ids = [
            tokenizer.start_token_id,
            tokenizer.token_to_id.get('INTRO_4', tokenizer.token_to_id['<UNK>']),
            tokenizer.token_to_id.get('VERSE_16', tokenizer.token_to_id['<UNK>']),
            tokenizer.end_token_id
        ]
        
        decoded = tokenizer.decode_arrangement(token_ids)
        
        assert len(decoded) == 2
        assert decoded[0]['type'] == 'INTRO'
        assert decoded[0]['length_bars'] == 4
        assert decoded[1]['type'] == 'VERSE'
        assert decoded[1]['length_bars'] == 16


class TestArrangementDataset:
    """Tests for ArrangementDataset"""
    
    def test_dataset_loading(self, temp_data_files, sample_arrangements):
        """Test dataset loading from files"""
        tokenizer = ArrangementTokenizer()
        data_path = str(Path(temp_data_files) / "test_data" / "**" / "arrangement.json")
        
        dataset = ArrangementDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=32
        )
        
        assert len(dataset) == len(sample_arrangements)
        
    def test_dataset_getitem(self, temp_data_files):
        """Test dataset item retrieval"""
        tokenizer = ArrangementTokenizer()
        data_path = str(Path(temp_data_files) / "test_data" / "**" / "arrangement.json")
        
        dataset = ArrangementDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=32
        )
        
        # Get first item
        item = dataset[0]
        
        # Check structure
        assert 'input_ids' in item
        assert 'target_ids' in item
        assert 'styles' in item
        assert 'tempos' in item
        assert 'durations' in item
        
        # Check tensor shapes and types
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['target_ids'], torch.Tensor)
        assert item['input_ids'].dtype == torch.long
        assert item['target_ids'].dtype == torch.long
        
        # Check sequence length
        assert len(item['input_ids']) == 31  # max_seq_length - 1
        assert len(item['target_ids']) == 31
        
    def test_dataset_validation(self):
        """Test arrangement validation"""
        tokenizer = ArrangementTokenizer()
        
        # Create invalid arrangement (missing required keys)
        invalid_arrangement = {
            "style": "rock_punk",
            "tempo": 140
            # Missing duration_bars and sections
        }
        
        dataset = ArrangementDataset(
            data_path="/nonexistent/path/*.json",  # Won't find files
            tokenizer=tokenizer,
            max_seq_length=32
        )
        
        # Test validation function directly
        assert not dataset._is_valid_arrangement(invalid_arrangement)
        
        # Valid arrangement
        valid_arrangement = {
            "style": "rock_punk",
            "tempo": 140,
            "duration_bars": 64,
            "sections": [
                {"type": "INTRO", "start_bar": 0, "length_bars": 4}
            ]
        }
        
        assert dataset._is_valid_arrangement(valid_arrangement)


class TestArrangementTransformer:
    """Tests for ArrangementTransformer"""
    
    def test_model_initialization(self, sample_config):
        """Test model initialization"""
        model = ArrangementTransformer(sample_config)
        
        assert model.vocab_size > 0
        assert model.d_model == sample_config['model']['d_model']
        assert model.n_heads == sample_config['model']['n_heads']
        assert model.n_layers == sample_config['model']['n_layers']
        
    def test_model_forward_pass(self, sample_config):
        """Test forward pass with dummy data"""
        model = ArrangementTransformer(sample_config)
        model.eval()
        
        batch_size = 2
        seq_length = 16
        
        # Create dummy inputs
        input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_length))
        styles = torch.randint(0, len(model.tokenizer.styles), (batch_size,))
        tempos = torch.randint(80, 160, (batch_size,))
        durations = torch.randint(32, 96, (batch_size,))
        
        # Forward pass
        with torch.no_grad():
            logits = model(input_ids, styles, tempos, durations)
            
        # Check output shape
        expected_shape = (batch_size, seq_length, model.vocab_size)
        assert logits.shape == expected_shape
        
    def test_condition_encoding(self, sample_config):
        """Test condition encoding"""
        model = ArrangementTransformer(sample_config)
        
        batch_size = 3
        styles = torch.tensor([0, 1, 2])  # Different styles
        tempos = torch.tensor([80, 120, 160])
        durations = torch.tensor([32, 64, 96])
        
        # Encode conditions
        with torch.no_grad():
            condition_encoding = model.encode_conditions(styles, tempos, durations)
            
        # Check shape
        expected_shape = (batch_size, 1, model.d_model)
        assert condition_encoding.shape == expected_shape
        
    def test_generation(self, sample_config):
        """Test arrangement generation"""
        model = ArrangementTransformer(sample_config)
        model.eval()
        
        # Generate arrangement
        with torch.no_grad():
            arrangement = model.generate_arrangement(
                style='rock_punk',
                tempo=140,
                target_duration=64,
                max_length=16,
                temperature=1.0
            )
            
        # Check output structure
        assert isinstance(arrangement, list)
        
        for section in arrangement:
            assert 'type' in section
            assert 'start_bar' in section
            assert 'length_bars' in section
            assert section['type'] in model.tokenizer.section_types


class TestDataModule:
    """Tests for ArrangementDataModule"""
    
    def test_datamodule_setup(self, sample_config, temp_data_files):
        """Test data module setup"""
        # Update config with temp data path
        sample_config['data']['dataset_path'] = str(Path(temp_data_files) / "test_data" / "**" / "arrangement.json")
        
        tokenizer = ArrangementTokenizer()
        data_module = ArrangementDataModule(sample_config, tokenizer)
        
        # Setup data
        data_module.setup()
        
        # Check splits
        assert len(data_module.train_dataset) > 0
        assert len(data_module.val_dataset) >= 0
        assert len(data_module.test_dataset) >= 0
        
        total_size = len(data_module.train_dataset) + len(data_module.val_dataset) + len(data_module.test_dataset)
        assert total_size > 0
        
    def test_dataloader_creation(self, sample_config, temp_data_files):
        """Test dataloader creation"""
        # Update config with temp data path
        sample_config['data']['dataset_path'] = str(Path(temp_data_files) / "test_data" / "**" / "arrangement.json")
        
        tokenizer = ArrangementTokenizer()
        data_module = ArrangementDataModule(sample_config, tokenizer)
        data_module.setup()
        
        # Test train dataloader
        train_loader = data_module.train_dataloader()
        assert train_loader is not None
        
        # Test getting a batch
        batch = next(iter(train_loader))
        assert 'input_ids' in batch
        assert 'target_ids' in batch
        assert 'styles' in batch
        assert 'tempos' in batch
        assert 'durations' in batch
        
        # Check batch dimensions
        batch_size = sample_config['data']['batch_size']
        seq_length = sample_config['model']['max_seq_length'] - 1
        
        assert batch['input_ids'].shape[0] <= batch_size  # May be smaller due to dataset size
        assert batch['input_ids'].shape[1] == seq_length


class TestShapeConsistency:
    """Tests for tensor shape consistency throughout the pipeline"""
    
    def test_end_to_end_shapes(self, sample_config, temp_data_files):
        """Test that tensor shapes are consistent from data loading to model output"""
        # Update config with temp data path
        sample_config['data']['dataset_path'] = str(Path(temp_data_files) / "test_data" / "**" / "arrangement.json")
        
        # Create model and data module
        model = ArrangementTransformer(sample_config)
        data_module = ArrangementDataModule(sample_config, model.tokenizer)
        data_module.setup()
        
        # Get a batch
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        
        # Run forward pass
        model.eval()
        with torch.no_grad():
            logits = model(
                batch['input_ids'],
                batch['styles'],
                batch['tempos'],
                batch['durations'],
                batch['target_ids']
            )
            
        # Check shapes are consistent
        batch_size, seq_length = batch['input_ids'].shape
        vocab_size = model.vocab_size
        
        assert logits.shape == (batch_size, seq_length, vocab_size)
        assert batch['target_ids'].shape == (batch_size, seq_length)
        
    def test_training_step_shapes(self, sample_config, temp_data_files):
        """Test training step with realistic data"""
        # Update config with temp data path
        sample_config['data']['dataset_path'] = str(Path(temp_data_files) / "test_data" / "**" / "arrangement.json")
        
        # Create model and data module
        model = ArrangementTransformer(sample_config)
        data_module = ArrangementDataModule(sample_config, model.tokenizer)
        data_module.setup()
        
        # Get a batch
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        
        # Run training step
        loss = model.training_step(batch, 0)
        
        # Check that loss is a scalar tensor
        assert loss.dim() == 0
        assert loss.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
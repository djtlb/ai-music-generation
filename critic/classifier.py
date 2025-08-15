"""
Adherence Classifier: Evaluates how well generated music matches the original prompt.
Scores prompt adherence based on control JSON vs actual tokens and audio features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AdherenceScore:
    """Structured output for adherence evaluation"""
    overall: float  # 0-1 score
    tempo_adherence: float
    key_adherence: float
    structure_adherence: float
    genre_adherence: float
    instrumentation_adherence: float
    details: Dict[str, Any]

class TextEncoder(nn.Module):
    """Encode prompt text to embeddings"""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings [batch_size, hidden_dim]"""
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                               return_tensors="pt", max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling over sequence length
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings

class ControlJSONEncoder(nn.Module):
    """Encode control JSON to fixed-size vectors"""
    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim
        
        # Learned embeddings for categorical features
        self.style_embedding = nn.Embedding(50, 32)  # 50 possible styles
        self.key_embedding = nn.Embedding(24, 16)    # 12 keys * 2 modes
        self.timefeel_embedding = nn.Embedding(20, 16)  # Various time feels
        
        # Linear layers for numerical features
        self.tempo_proj = nn.Linear(1, 16)
        self.structure_proj = nn.Linear(10, 32)  # Max 10 sections
        
        # Final projection
        self.final_proj = nn.Linear(32 + 16 + 16 + 16 + 32, output_dim)
        
    def forward(self, control_jsons: List[Dict]) -> torch.Tensor:
        """Encode control JSONs to embeddings [batch_size, output_dim]"""
        batch_size = len(control_jsons)
        device = next(self.parameters()).device
        
        # Extract and encode features
        style_ids = []
        key_ids = []
        timefeel_ids = []
        tempos = []
        structures = []
        
        for ctrl in control_jsons:
            # Style (hash to ID)
            style_id = hash(ctrl.get('style', 'unknown')) % 50
            style_ids.append(style_id)
            
            # Key (C=0, C#=1, ..., Am=12, A#m=13, etc.)
            key_str = ctrl.get('key', 'C')
            key_id = self._key_to_id(key_str)
            key_ids.append(key_id)
            
            # Time feel
            timefeel_id = hash(ctrl.get('timefeel', 'straight')) % 20
            timefeel_ids.append(timefeel_id)
            
            # Tempo (normalize to 0-1)
            tempo = ctrl.get('bpm', 120) / 200.0  # Assume max 200 BPM
            tempos.append(tempo)
            
            # Structure (section counts, padded to 10)
            arrangement = ctrl.get('arrangement', [])
            structure_vec = [0.0] * 10
            for i, section in enumerate(arrangement[:10]):
                structure_vec[i] = section.get('bars', 0) / 16.0  # Normalize by max bars
            structures.append(structure_vec)
        
        # Convert to tensors
        style_ids = torch.tensor(style_ids, device=device)
        key_ids = torch.tensor(key_ids, device=device)
        timefeel_ids = torch.tensor(timefeel_ids, device=device)
        tempos = torch.tensor(tempos, device=device).unsqueeze(1)
        structures = torch.tensor(structures, device=device)
        
        # Embed categorical features
        style_emb = self.style_embedding(style_ids)
        key_emb = self.key_embedding(key_ids)
        timefeel_emb = self.timefeel_embedding(timefeel_ids)
        
        # Project numerical features
        tempo_emb = self.tempo_proj(tempos)
        structure_emb = self.structure_proj(structures)
        
        # Concatenate and project
        combined = torch.cat([style_emb, key_emb, timefeel_emb, tempo_emb, structure_emb], dim=1)
        output = self.final_proj(combined)
        
        return output
    
    def _key_to_id(self, key_str: str) -> int:
        """Convert key string to ID"""
        key_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
            'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
            'Cm': 12, 'C#m': 13, 'Dbm': 13, 'Dm': 14, 'D#m': 15, 'Ebm': 15, 'Em': 16,
            'Fm': 17, 'F#m': 18, 'Gbm': 18, 'Gm': 19, 'G#m': 20, 'Abm': 20, 'Am': 21, 'A#m': 22, 'Bbm': 22, 'Bm': 23
        }
        return key_map.get(key_str, 0)

class TokenSequenceEncoder(nn.Module):
    """Encode 8-bar token windows to embeddings"""
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.output_dim = hidden_dim * 2
        
    def forward(self, token_sequences: torch.Tensor) -> torch.Tensor:
        """Encode token sequences [batch_size, seq_len] -> [batch_size, hidden_dim*2]"""
        # Embed tokens
        embeddings = self.embedding(token_sequences)  # [batch, seq, embed]
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(embeddings)
        
        # Use final hidden state (concatenate forward and backward)
        output = torch.cat([h_n[0], h_n[1]], dim=1)  # [batch, hidden_dim*2]
        
        return output

class AdherenceClassifier(nn.Module):
    """
    Main classifier that combines prompt text, control JSON, and tokens
    to predict adherence score between 0-1.
    """
    def __init__(
        self,
        vocab_size: int,
        text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 256
    ):
        super().__init__()
        
        # Component encoders
        self.text_encoder = TextEncoder(text_encoder_model)
        self.control_encoder = ControlJSONEncoder(output_dim=128)
        self.token_encoder = TokenSequenceEncoder(vocab_size, hidden_dim=hidden_dim)
        
        # Fusion network
        total_input_dim = (self.text_encoder.output_dim + 
                          self.control_encoder.output_dim + 
                          self.token_encoder.output_dim)
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output 0-1 adherence score
        )
        
        # Component-specific heads for detailed scoring
        self.tempo_head = nn.Linear(hidden_dim // 2, 1)
        self.key_head = nn.Linear(hidden_dim // 2, 1)
        self.structure_head = nn.Linear(hidden_dim // 2, 1)
        self.genre_head = nn.Linear(hidden_dim // 2, 1)
        self.instrumentation_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(
        self,
        prompt_texts: List[str],
        control_jsons: List[Dict],
        token_sequences: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass returning overall score and component scores.
        
        Args:
            prompt_texts: List of original prompt strings
            control_jsons: List of control JSON dicts
            token_sequences: Token sequences [batch_size, seq_len]
            
        Returns:
            overall_scores: [batch_size, 1] adherence scores
            component_scores: Dict with detailed component scores
        """
        # Encode all inputs
        text_emb = self.text_encoder(prompt_texts)
        control_emb = self.control_encoder(control_jsons)
        token_emb = self.token_encoder(token_sequences)
        
        # Fuse representations
        combined = torch.cat([text_emb, control_emb, token_emb], dim=1)
        
        # Pass through fusion network
        fused = combined
        for i, layer in enumerate(self.fusion_layers):
            fused = layer(fused)
            if i == 4:  # After 3rd linear layer, save for component heads
                component_features = fused
        
        overall_scores = fused  # Final sigmoid output
        
        # Component-specific predictions
        component_scores = {
            'tempo': torch.sigmoid(self.tempo_head(component_features)),
            'key': torch.sigmoid(self.key_head(component_features)),
            'structure': torch.sigmoid(self.structure_head(component_features)),
            'genre': torch.sigmoid(self.genre_head(component_features)),
            'instrumentation': torch.sigmoid(self.instrumentation_head(component_features))
        }
        
        return overall_scores, component_scores
    
    def predict_adherence(
        self,
        prompt_texts: List[str],
        control_jsons: List[Dict],
        token_sequences: torch.Tensor
    ) -> List[AdherenceScore]:
        """Predict structured adherence scores"""
        self.eval()
        with torch.no_grad():
            overall_scores, component_scores = self.forward(
                prompt_texts, control_jsons, token_sequences
            )
            
            results = []
            for i in range(len(prompt_texts)):
                score = AdherenceScore(
                    overall=overall_scores[i].item(),
                    tempo_adherence=component_scores['tempo'][i].item(),
                    key_adherence=component_scores['key'][i].item(),
                    structure_adherence=component_scores['structure'][i].item(),
                    genre_adherence=component_scores['genre'][i].item(),
                    instrumentation_adherence=component_scores['instrumentation'][i].item(),
                    details={
                        'prompt': prompt_texts[i],
                        'control': control_jsons[i],
                        'token_count': token_sequences[i].shape[0]
                    }
                )
                results.append(score)
            
            return results

# Training utilities
class AdherenceDataset(torch.utils.data.Dataset):
    """Dataset for training adherence classifier"""
    def __init__(self, data_path: str):
        self.samples = []
        self._load_data(data_path)
    
    def _load_data(self, data_path: str):
        """Load training data with prompt/control/tokens/score tuples"""
        # Format: each line is JSON with keys: prompt, control_json, tokens, adherence_score
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'prompt': sample['prompt'],
            'control_json': sample['control_json'],
            'tokens': torch.tensor(sample['tokens'], dtype=torch.long),
            'adherence_score': torch.tensor(sample['adherence_score'], dtype=torch.float),
            'component_scores': {
                k: torch.tensor(v, dtype=torch.float) 
                for k, v in sample.get('component_scores', {}).items()
            }
        }

def train_classifier(
    model: AdherenceClassifier,
    train_dataset: AdherenceDataset,
    val_dataset: AdherenceDataset,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
):
    """Train the adherence classifier"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            
            prompts = batch['prompt']
            controls = batch['control_json']
            tokens = batch['tokens'].to(device)
            targets = batch['adherence_score'].to(device)
            
            overall_pred, component_pred = model(prompts, controls, tokens)
            
            # Main loss
            loss = criterion(overall_pred.squeeze(), targets)
            
            # Component losses (if available)
            for comp_name, comp_pred in component_pred.items():
                if comp_name in batch['component_scores']:
                    comp_targets = batch['component_scores'][comp_name].to(device)
                    loss += 0.1 * criterion(comp_pred.squeeze(), comp_targets)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                prompts = batch['prompt']
                controls = batch['control_json']
                tokens = batch['tokens'].to(device)
                targets = batch['adherence_score'].to(device)
                
                overall_pred, _ = model(prompts, controls, tokens)
                loss = criterion(overall_pred.squeeze(), targets)
                val_loss += loss.item()
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss/len(train_loader):.4f}")
        logger.info(f"Val Loss: {val_loss/len(val_loader):.4f}")

if __name__ == "__main__":
    # Example usage
    vocab_size = 1000  # From tokenizer
    model = AdherenceClassifier(vocab_size)
    
    # Example data
    prompts = ["upbeat pop song with guitar and drums"]
    controls = [{
        'style': 'pop',
        'bpm': 120,
        'key': 'C',
        'timefeel': 'straight',
        'arrangement': [
            {'section': 'INTRO', 'bars': 4},
            {'section': 'VERSE', 'bars': 8},
            {'section': 'CHORUS', 'bars': 8}
        ]
    }]
    tokens = torch.randint(0, vocab_size, (1, 64))  # 8 bars worth
    
    scores = model.predict_adherence(prompts, controls, tokens)
    print(f"Adherence Score: {scores[0].overall:.3f}")
    print(f"Tempo: {scores[0].tempo_adherence:.3f}")
    print(f"Structure: {scores[0].structure_adherence:.3f}")
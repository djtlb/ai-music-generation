"""
Style adapters that apply LoRA adaptations for specific music styles.

Supports hierarchical style modeling:
- StyleAdapter: Single-style LoRA adaptation
- HierarchicalStyleAdapter: Parent + child style composition
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List
import yaml
from pathlib import Path

from .lora_layer import LoRALinear, apply_lora_to_model, get_lora_parameters


class StyleAdapter(nn.Module):
    """
    Single-style LoRA adapter that modifies model behavior for a specific style.
    
    Wraps a base model and applies style-specific LoRA adaptations to
    targeted layers (attention, feed_forward, etc.)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        style_name: str,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        target_modules: List[str] = None
    ):
        super().__init__()
        
        self.base_model = base_model
        self.style_name = style_name
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        
        if target_modules is None:
            target_modules = ['attention', 'feed_forward']
        self.target_modules = target_modules
        
        # Apply LoRA to target modules
        self.lora_layers = apply_lora_to_model(
            self.base_model,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Style-specific parameters 
        self.style_embedding = nn.Parameter(
            torch.randn(self.base_model.config.hidden_size) * 0.1
        )
        
        print(f"Applied LoRA to {len(self.lora_layers)} layers for style '{style_name}'")
        
    def forward(self, *args, **kwargs):
        """Forward pass through style-adapted model."""
        return self.base_model(*args, **kwargs)
        
    def get_style_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get state dict containing only style-specific parameters."""
        state_dict = {}
        
        # Add style embedding
        state_dict['style_embedding'] = self.style_embedding
        
        # Add LoRA parameters
        for name, lora_layer in self.lora_layers.items():
            lora_state = lora_layer.state_dict_lora_only()
            for key, value in lora_state.items():
                state_dict[f"{name}.{key}"] = value
                
        # Add metadata
        state_dict['_metadata'] = {
            'style_name': self.style_name,
            'rank': self.rank,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'target_modules': self.target_modules
        }
        
        return state_dict
        
    def load_style_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load style-specific parameters from state dict."""
        # Load style embedding
        if 'style_embedding' in state_dict:
            self.style_embedding.data = state_dict['style_embedding']
            
        # Load LoRA parameters
        lora_states = {}
        for key, value in state_dict.items():
            if key.startswith('_') or key == 'style_embedding':
                continue
                
            # Parse LoRA parameter: "layer_name.lora_param"
            parts = key.rsplit('.', 1)
            if len(parts) == 2:
                layer_name, param_name = parts
                if layer_name not in lora_states:
                    lora_states[layer_name] = {}
                lora_states[layer_name][param_name] = value
                
        # Apply LoRA states to layers
        for layer_name, lora_state in lora_states.items():
            if layer_name in self.lora_layers:
                self.lora_layers[layer_name].load_lora_state_dict(lora_state)
                
    def merge_and_save(self, output_path: str):
        """Merge LoRA weights and save full model."""
        # Merge all LoRA layers
        for lora_layer in self.lora_layers.values():
            lora_layer.merge_weights_()
            
        # Save merged model
        torch.save(self.base_model.state_dict(), output_path)
        
        # Unmerge for continued training
        for lora_layer in self.lora_layers.values():
            lora_layer.unmerge_weights()


class HierarchicalStyleAdapter(nn.Module):
    """
    Hierarchical style adapter that combines parent and child style adaptations.
    
    Uses composition: base_model + parent_adapter + child_adapter
    Allows child styles to inherit and modify parent style characteristics.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        parent_style: str,
        child_style: Optional[str] = None,
        parent_config: Optional[Dict] = None,
        child_config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.base_model = base_model
        self.parent_style = parent_style
        self.child_style = child_style
        
        # Load configurations
        self.parent_config = parent_config or self._load_style_config(parent_style)
        self.child_config = child_config or self._load_style_config(child_style) if child_style else {}
        
        # Create parent adapter
        parent_lora_config = self.parent_config.get('lora', {})
        self.parent_adapter = StyleAdapter(
            base_model=base_model,
            style_name=parent_style,
            rank=parent_lora_config.get('rank', 8),
            alpha=parent_lora_config.get('alpha', 16.0),
            dropout=parent_lora_config.get('dropout', 0.1),
            target_modules=parent_lora_config.get('target_modules')
        )
        
        # Create child adapter if specified
        self.child_adapter = None
        if child_style:
            child_lora_config = self.child_config.get('lora', {})
            # Child adapter operates on the parent-adapted model
            self.child_adapter = StyleAdapter(
                base_model=base_model,  # Same base model - LoRA stacks
                style_name=child_style,
                rank=child_lora_config.get('rank', 4),  # Smaller rank for child
                alpha=child_lora_config.get('alpha', 8.0),
                dropout=child_lora_config.get('dropout', 0.1),
                target_modules=child_lora_config.get('target_modules')
            )
            
    def _load_style_config(self, style_name: str) -> Dict:
        """Load style configuration from YAML file."""
        if not style_name:
            return {}
            
        # Try parent genre first
        config_path = Path(f"configs/genres/{style_name}.yaml")
        if not config_path.exists():
            # Try child style path
            parent_name = style_name.split('_')[0]  # e.g., 'dance_pop' -> 'pop'
            config_path = Path(f"configs/styles/{parent_name}/{style_name}.yaml")
            
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            print(f"Warning: No config found for style '{style_name}'")
            return {}
            
    def forward(self, *args, **kwargs):
        """Forward pass through hierarchical style adaptation."""
        return self.base_model(*args, **kwargs)
        
    def set_style_weights(self, parent_weight: float = 1.0, child_weight: float = 1.0):
        """Adjust the relative influence of parent and child adaptations."""
        # Update LoRA scaling factors
        for lora_layer in self.parent_adapter.lora_layers.values():
            lora_layer.scaling = (lora_layer.alpha / lora_layer.rank) * parent_weight
            
        if self.child_adapter:
            for lora_layer in self.child_adapter.lora_layers.values():
                lora_layer.scaling = (lora_layer.alpha / lora_layer.rank) * child_weight
                
    def get_hierarchical_state_dict(self) -> Dict[str, Any]:
        """Get state dict for the complete hierarchical adapter."""
        state_dict = {
            'parent_style': self.parent_style,
            'child_style': self.child_style,
            'parent_config': self.parent_config,
            'child_config': self.child_config,
            'parent_adapter': self.parent_adapter.get_style_state_dict()
        }
        
        if self.child_adapter:
            state_dict['child_adapter'] = self.child_adapter.get_style_state_dict()
            
        return state_dict
        
    def load_hierarchical_state_dict(self, state_dict: Dict[str, Any]):
        """Load hierarchical adapter from state dict."""
        self.parent_style = state_dict['parent_style']
        self.child_style = state_dict.get('child_style')
        self.parent_config = state_dict['parent_config']
        self.child_config = state_dict.get('child_config', {})
        
        # Load parent adapter
        self.parent_adapter.load_style_state_dict(state_dict['parent_adapter'])
        
        # Load child adapter if present
        if 'child_adapter' in state_dict and self.child_adapter:
            self.child_adapter.load_style_state_dict(state_dict['child_adapter'])
            
    def save_adapters(self, output_dir: str):
        """Save parent and child adapters separately."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save parent adapter
        parent_path = output_path / f"{self.parent_style}_adapter.pt"
        torch.save(self.parent_adapter.get_style_state_dict(), parent_path)
        
        # Save child adapter if present
        if self.child_adapter:
            child_path = output_path / f"{self.child_style}_adapter.pt"
            torch.save(self.child_adapter.get_style_state_dict(), child_path)
            
        # Save hierarchical config
        config_path = output_path / "hierarchical_config.yaml"
        config = {
            'parent_style': self.parent_style,
            'child_style': self.child_style,
            'parent_config': self.parent_config,
            'child_config': self.child_config
        }
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        print(f"Saved adapters to {output_dir}")


def load_style_adapter(
    base_model: nn.Module,
    adapter_path: str,
    adapter_type: str = 'style'
) -> StyleAdapter:
    """Load a style adapter from saved checkpoint."""
    state_dict = torch.load(adapter_path, map_location='cpu')
    
    if adapter_type == 'style':
        metadata = state_dict.get('_metadata', {})
        adapter = StyleAdapter(
            base_model=base_model,
            style_name=metadata.get('style_name', 'unknown'),
            rank=metadata.get('rank', 8),
            alpha=metadata.get('alpha', 16.0),
            dropout=metadata.get('dropout', 0.1),
            target_modules=metadata.get('target_modules')
        )
        adapter.load_style_state_dict(state_dict)
        return adapter
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
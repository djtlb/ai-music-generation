"""
LoRA (Low-Rank Adaptation) layer implementations.

Provides efficient parameter updates through low-rank decomposition:
- LoRALayer: Base class with rank-based weight updates  
- LoRALinear: Linear layer with LoRA adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


class LoRALayer(nn.Module):
    """
    Base LoRA layer that adds low-rank adaptation to existing layers.
    
    LoRA decomposes weight updates as W + BA where:
    - B: (out_features, rank) matrix  
    - A: (rank, in_features) matrix
    - rank << min(in_features, out_features)
    """
    
    def __init__(
        self,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        merge_weights: bool = True
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.merge_weights = merge_weights
        self.merged = False
        
        # Will be set by subclasses
        self.lora_A = None  
        self.lora_B = None
        self.scaling = None
        
    def reset_parameters(self):
        """Initialize LoRA weights using Kaiming initialization."""
        if hasattr(self, 'lora_A') and self.lora_A is not None:
            # Initialize A with small random values
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        if hasattr(self, 'lora_B') and self.lora_B is not None:
            # Initialize B to zero for stable training start
            nn.init.zeros_(self.lora_B)
            
    def train(self, mode: bool = True):
        """Override train mode to handle weight merging."""
        super().train(mode)
        if mode and self.merge_weights and self.merged:
            # Unmerge weights when entering training mode
            self.unmerge_weights()
        elif not mode and self.merge_weights and not self.merged:
            # Merge weights when entering eval mode
            self.merge_weights_()
        return self
            
    def merge_weights_(self):
        """Merge LoRA weights into base layer for inference efficiency."""
        raise NotImplementedError("Subclasses must implement merge_weights_")
        
    def unmerge_weights(self):
        """Unmerge LoRA weights from base layer.""" 
        raise NotImplementedError("Subclasses must implement unmerge_weights")


class LoRALinear(LoRALayer):
    """
    Linear layer with LoRA adaptation.
    
    Implements: output = (W + α/r * B @ A) @ input + bias
    where W is the frozen base weight matrix.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        merge_weights: bool = True
    ):
        super().__init__(rank, alpha, dropout, merge_weights)
        
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # Freeze base layer weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.scaling = self.alpha / self.rank
        
        # Dropout for LoRA path
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.reset_parameters()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA-adapted linear layer."""
        # Base layer computation
        result = self.base_layer(x)
        
        if not self.merged:
            # LoRA path: x -> A -> dropout -> B -> scale -> add
            lora_out = self.lora_dropout(x) @ self.lora_A.T
            lora_out = lora_out @ self.lora_B.T
            result = result + lora_out * self.scaling
            
        return result
        
    def merge_weights_(self):
        """Merge LoRA weights into base layer."""
        if not self.merged:
            # Compute LoRA weight update: α/r * B @ A
            delta_w = self.scaling * (self.lora_B @ self.lora_A)
            
            # Add to base weights
            self.base_layer.weight.data += delta_w
            self.merged = True
            
    def unmerge_weights(self):
        """Unmerge LoRA weights from base layer."""
        if self.merged:
            # Subtract LoRA weight update
            delta_w = self.scaling * (self.lora_B @ self.lora_A)
            self.base_layer.weight.data -= delta_w
            self.merged = False
            
    def state_dict_lora_only(self) -> Dict[str, torch.Tensor]:
        """Return state dict containing only LoRA parameters."""
        return {
            'lora_A': self.lora_A,
            'lora_B': self.lora_B,
            'alpha': torch.tensor(self.alpha),
            'rank': torch.tensor(self.rank),
            'scaling': torch.tensor(self.scaling)
        }
        
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load LoRA parameters from state dict."""
        self.lora_A.data = state_dict['lora_A']
        self.lora_B.data = state_dict['lora_B']
        self.alpha = state_dict['alpha'].item()
        self.rank = state_dict['rank'].item()
        self.scaling = state_dict['scaling'].item()


def apply_lora_to_model(
    model: nn.Module,
    target_modules: list = None,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0
) -> Dict[str, LoRALinear]:
    """
    Apply LoRA adaptation to specified linear layers in a model.
    
    Args:
        model: Base model to adapt
        target_modules: List of module name patterns to adapt (e.g., ['attention', 'feed_forward'])
        rank: LoRA rank
        alpha: LoRA scaling parameter  
        dropout: LoRA dropout rate
        
    Returns:
        Dictionary mapping module names to LoRA layers
    """
    if target_modules is None:
        target_modules = ['attention', 'feed_forward', 'linear']
        
    lora_layers = {}
    
    for name, module in model.named_modules():
        # Check if this module should get LoRA
        should_adapt = False
        if isinstance(module, nn.Linear):
            for target in target_modules:
                if target.lower() in name.lower():
                    should_adapt = True
                    break
                    
        if should_adapt:
            # Replace linear layer with LoRA version
            lora_layer = LoRALinear(
                base_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout
            )
            
            # Replace in parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, lora_layer)
            else:
                setattr(model, child_name, lora_layer)
                
            lora_layers[name] = lora_layer
            
    return lora_layers


def get_lora_parameters(model: nn.Module) -> Dict[str, nn.Parameter]:
    """Extract only LoRA parameters from a model."""
    lora_params = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_params[f"{name}.lora_A"] = module.lora_A
            lora_params[f"{name}.lora_B"] = module.lora_B
            
    return lora_params


def freeze_non_lora_parameters(model: nn.Module):
    """Freeze all parameters except LoRA parameters."""
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
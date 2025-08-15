"""
Adapter merging utilities for combining base models with hierarchical LoRA adapters.

Supports stacking: base → parent → child for inference with proper weight combination.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
import copy

from .lora_layer import LoRALinear
from .style_adapter import StyleAdapter, HierarchicalStyleAdapter


class AdapterMerger:
    """
    Merges LoRA adapters with base models for efficient inference.
    
    Supports:
    - Single adapter merging
    - Weight scaling and combination
    - Temporary merging with context managers
    """
    
    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.merged_adapters = []
        self.backup_weights = {}
        
    def merge_adapter(
        self,
        adapter: StyleAdapter,
        weight: float = 1.0,
        temporary: bool = False
    ) -> None:
        """
        Merge a style adapter into the base model.
        
        Args:
            adapter: StyleAdapter to merge
            weight: Scaling factor for adapter weights  
            temporary: If True, store backup for later restoration
        """
        if temporary and not self.backup_weights:
            self._backup_weights()
            
        for name, lora_layer in adapter.lora_layers.items():
            if isinstance(lora_layer, LoRALinear):
                # Compute weighted LoRA update
                delta_w = weight * lora_layer.scaling * (lora_layer.lora_B @ lora_layer.lora_A)
                
                # Apply to base layer
                lora_layer.base_layer.weight.data += delta_w
                
        self.merged_adapters.append((adapter.style_name, weight))
        print(f"Merged adapter '{adapter.style_name}' with weight {weight}")
        
    def unmerge_all(self) -> None:
        """Restore base model weights by removing all merged adapters."""
        if self.backup_weights:
            # Restore from backup
            for name, weight_backup in self.backup_weights.items():
                module = self._get_module_by_name(name)
                if hasattr(module, 'weight'):
                    module.weight.data = weight_backup.clone()
                    
            self.backup_weights.clear()
        else:
            # Manual unmerging - requires adapters to be available
            raise ValueError("No backup weights available for unmerging")
            
        self.merged_adapters.clear()
        print("Unmerged all adapters")
        
    def _backup_weights(self) -> None:
        """Backup current model weights."""
        for name, module in self.base_model.named_modules():
            if isinstance(module, (nn.Linear, LoRALinear)):
                if hasattr(module, 'weight'):
                    self.backup_weights[name] = module.weight.data.clone()
                    
    def _get_module_by_name(self, name: str) -> nn.Module:
        """Get module by dotted name path."""
        module = self.base_model
        for part in name.split('.'):
            module = getattr(module, part)
        return module
        
    def create_merged_model(self, adapters: List[Tuple[StyleAdapter, float]]) -> nn.Module:
        """
        Create a new model with adapters permanently merged.
        
        Args:
            adapters: List of (adapter, weight) tuples
            
        Returns:
            New model with merged weights
        """
        # Deep copy base model
        merged_model = copy.deepcopy(self.base_model)
        merger = AdapterMerger(merged_model)
        
        # Merge each adapter
        for adapter, weight in adapters:
            merger.merge_adapter(adapter, weight, temporary=False)
            
        return merged_model


class HierarchicalMerger:
    """
    Specialized merger for hierarchical parent/child adapter combinations.
    
    Properly stacks: base → parent → child with configurable blending.
    """
    
    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.parent_adapter = None
        self.child_adapter = None
        self.merged_state = None
        
    def load_parent_adapter(self, adapter_path: str, weight: float = 1.0) -> None:
        """Load parent adapter from checkpoint."""
        from .style_adapter import load_style_adapter
        
        self.parent_adapter = load_style_adapter(self.base_model, adapter_path)
        self.parent_weight = weight
        print(f"Loaded parent adapter: {self.parent_adapter.style_name}")
        
    def load_child_adapter(
        self,
        adapter_path: str,
        parent_adapter_path: Optional[str] = None,
        child_weight: float = 1.0
    ) -> None:
        """
        Load child adapter and optionally its parent dependency.
        
        Args:
            adapter_path: Path to child adapter
            parent_adapter_path: Optional path to parent adapter  
            child_weight: Scaling for child adapter
        """
        from .style_adapter import load_style_adapter
        
        # Load parent first if specified
        if parent_adapter_path and not self.parent_adapter:
            self.load_parent_adapter(parent_adapter_path)
            
        self.child_adapter = load_style_adapter(self.base_model, adapter_path)
        self.child_weight = child_weight
        print(f"Loaded child adapter: {self.child_adapter.style_name}")
        
    def merge_hierarchical(
        self,
        parent_weight: float = 1.0,
        child_weight: float = 1.0,
        blend_mode: str = 'additive'
    ) -> None:
        """
        Merge parent and child adapters hierarchically.
        
        Args:
            parent_weight: Scaling for parent adapter
            child_weight: Scaling for child adapter  
            blend_mode: 'additive' or 'interpolative'
        """
        if not self.parent_adapter:
            raise ValueError("Parent adapter must be loaded first")
            
        # Store original state for restoration
        self._backup_base_weights()
        
        if blend_mode == 'additive':
            # Base + parent + child
            self._merge_additive(parent_weight, child_weight)
        elif blend_mode == 'interpolative':
            # Base + interpolated(parent, child)
            self._merge_interpolative(parent_weight, child_weight)
        else:
            raise ValueError(f"Unknown blend mode: {blend_mode}")
            
        self.merged_state = {
            'parent_weight': parent_weight,
            'child_weight': child_weight,
            'blend_mode': blend_mode
        }
        
    def _merge_additive(self, parent_weight: float, child_weight: float) -> None:
        """Merge adapters additively: base + α*parent + β*child."""
        # Apply parent adapter
        for name, lora_layer in self.parent_adapter.lora_layers.items():
            if isinstance(lora_layer, LoRALinear):
                delta_w = parent_weight * lora_layer.scaling * (lora_layer.lora_B @ lora_layer.lora_A)
                lora_layer.base_layer.weight.data += delta_w
                
        # Apply child adapter (if available)
        if self.child_adapter:
            for name, lora_layer in self.child_adapter.lora_layers.items():
                if isinstance(lora_layer, LoRALinear):
                    delta_w = child_weight * lora_layer.scaling * (lora_layer.lora_B @ lora_layer.lora_A)
                    lora_layer.base_layer.weight.data += delta_w
                    
    def _merge_interpolative(self, parent_weight: float, child_weight: float) -> None:
        """Merge adapters with interpolation: base + (α*parent + β*child)/(α+β)."""
        total_weight = parent_weight + child_weight
        if total_weight == 0:
            return
            
        norm_parent = parent_weight / total_weight
        norm_child = child_weight / total_weight
        
        # Collect all target layers
        all_layers = set()
        if self.parent_adapter:
            all_layers.update(self.parent_adapter.lora_layers.keys())
        if self.child_adapter:
            all_layers.update(self.child_adapter.lora_layers.keys())
            
        # Apply interpolated weights
        for layer_name in all_layers:
            delta_w = torch.zeros_like(self._get_layer_weight(layer_name))
            
            # Add parent contribution
            if layer_name in self.parent_adapter.lora_layers:
                parent_lora = self.parent_adapter.lora_layers[layer_name]
                delta_w += norm_parent * parent_lora.scaling * (parent_lora.lora_B @ parent_lora.lora_A)
                
            # Add child contribution  
            if self.child_adapter and layer_name in self.child_adapter.lora_layers:
                child_lora = self.child_adapter.lora_layers[layer_name]
                delta_w += norm_child * child_lora.scaling * (child_lora.lora_B @ child_lora.lora_A)
                
            # Apply combined update
            target_layer = self._get_layer_by_name(layer_name)
            if hasattr(target_layer, 'weight'):
                target_layer.weight.data += delta_w
                
    def _backup_base_weights(self) -> None:
        """Backup base model weights before merging."""
        self.weight_backup = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, (nn.Linear, LoRALinear)):
                if hasattr(module, 'weight'):
                    self.weight_backup[name] = module.weight.data.clone()
                    
    def _get_layer_weight(self, layer_name: str) -> torch.Tensor:
        """Get weight tensor for a named layer.""" 
        layer = self._get_layer_by_name(layer_name)
        return layer.base_layer.weight if isinstance(layer, LoRALinear) else layer.weight
        
    def _get_layer_by_name(self, name: str) -> nn.Module:
        """Get module by dotted name path."""
        module = self.base_model
        for part in name.split('.'):
            module = getattr(module, part)
        return module
        
    def unmerge(self) -> None:
        """Restore base model to pre-merge state.""" 
        if hasattr(self, 'weight_backup'):
            for name, weight_backup in self.weight_backup.items():
                module = self._get_layer_by_name(name)
                if hasattr(module, 'weight'):
                    module.weight.data = weight_backup.clone()
                    
            del self.weight_backup
            
        self.merged_state = None
        print("Unmerged hierarchical adapters")
        
    def create_merged_checkpoint(self, output_path: str) -> None:
        """Save merged model as a checkpoint."""
        if not self.merged_state:
            raise ValueError("No adapters currently merged")
            
        checkpoint = {
            'model_state_dict': self.base_model.state_dict(),
            'merged_state': self.merged_state,
            'parent_style': self.parent_adapter.style_name if self.parent_adapter else None,
            'child_style': self.child_adapter.style_name if self.child_adapter else None
        }
        
        torch.save(checkpoint, output_path)
        print(f"Saved merged checkpoint to {output_path}")


def verify_adapter_compatibility(
    base_model: nn.Module,
    adapter_paths: List[str]
) -> bool:
    """
    Verify that adapters are compatible with base model architecture.
    
    Args:
        base_model: Target base model
        adapter_paths: List of adapter checkpoint paths
        
    Returns:
        True if all adapters are compatible
    """
    base_layers = {name: module for name, module in base_model.named_modules() 
                   if isinstance(module, nn.Linear)}
    
    for adapter_path in adapter_paths:
        try:
            adapter_state = torch.load(adapter_path, map_location='cpu')
            
            # Check LoRA layer compatibility
            for key in adapter_state:
                if '.lora_A' in key or '.lora_B' in key:
                    layer_name = key.split('.lora_')[0]
                    if layer_name not in base_layers:
                        print(f"Adapter {adapter_path} targets non-existent layer: {layer_name}")
                        return False
                        
                    # Check dimension compatibility
                    lora_tensor = adapter_state[key]
                    base_layer = base_layers[layer_name]
                    
                    if '.lora_A' in key:
                        if lora_tensor.shape[1] != base_layer.in_features:
                            print(f"LoRA A dimension mismatch in {layer_name}")
                            return False
                    elif '.lora_B' in key:
                        if lora_tensor.shape[0] != base_layer.out_features:
                            print(f"LoRA B dimension mismatch in {layer_name}")
                            return False
                            
        except Exception as e:
            print(f"Error checking adapter {adapter_path}: {e}")
            return False
            
    print("All adapters are compatible")
    return True
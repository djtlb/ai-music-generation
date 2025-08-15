"""
Hierarchical LoRA adapters for style-conditioned music generation.

Implements parent/child adapter training to learn style hierarchies:
- Parent adapters capture broad genre characteristics (e.g., pop)
- Child adapters learn specific sub-style variations (e.g., dance_pop)
- Merge utilities combine adapters for hierarchical inference
"""

from .lora_layer import LoRALayer, LoRALinear
from .style_adapter import StyleAdapter, HierarchicalStyleAdapter
from .adapter_merge import AdapterMerger, HierarchicalMerger
from .training_utils import ParentAdapterTrainer, ChildAdapterTrainer

__all__ = [
    'LoRALayer', 'LoRALinear',
    'StyleAdapter', 'HierarchicalStyleAdapter', 
    'AdapterMerger', 'HierarchicalMerger',
    'ParentAdapterTrainer', 'ChildAdapterTrainer'
]
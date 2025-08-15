#!/bin/bash
# Example usage scripts for hierarchical LoRA adapter training

echo "=== Hierarchical LoRA Adapter Training Examples ==="
echo

# Example 1: Train parent adapter for pop genre
echo "1. Training parent adapter for 'pop' genre:"
echo "python train_parent_adapter.py --parent pop --pack /style_packs/pop \\"
echo "    --epochs 15 --batch-size 8 --rank 16 --alpha 32.0"
echo

# Example 2: Train child adapter for dance_pop
echo "2. Training child adapter for 'dance_pop' (inherits from pop):"
echo "python train_child_adapter.py --parent pop --child dance_pop \\"
echo "    --pack /style_packs/pop/dance_pop \\"
echo "    --parent_lora checkpoints/adapters/pop/pop.lora \\"
echo "    --epochs 8 --batch-size 4 --rank 8 --alpha 16.0"
echo

# Example 3: Train multiple children for pop genre
echo "3. Training multiple child adapters for pop sub-styles:"

styles=("dance_pop" "synth_pop" "indie_pop" "pop_rock" "electro_pop")
for style in "${styles[@]}"; do
    echo "python train_child_adapter.py --parent pop --child $style \\"
    echo "    --pack /style_packs/pop/$style \\"
    echo "    --parent_lora checkpoints/adapters/pop/pop.lora"
done
echo

# Example 4: Merge adapters for inference
echo "4. Merging adapters for inference:"
echo "# Single parent adapter merge:"
echo "python merge_adapters.py --base-model checkpoints/base_model.pt \\"
echo "    --parent-adapter checkpoints/adapters/pop/pop.lora \\"
echo "    --output merged_models/pop_model.pt --create-config"
echo

echo "# Hierarchical (parent + child) merge:"
echo "python merge_adapters.py --base-model checkpoints/base_model.pt \\"
echo "    --parent-adapter checkpoints/adapters/pop/pop.lora \\"
echo "    --child-adapter checkpoints/adapters/pop/dance_pop/dance_pop.lora \\"
echo "    --output merged_models/dance_pop_model.pt \\"
echo "    --blend-mode additive --create-config"
echo

# Example 5: Test adapter compatibility and decode parity
echo "5. Testing adapter compatibility and decode parity:"
echo "python merge_adapters.py --base-model checkpoints/base_model.pt \\"
echo "    --parent-adapter checkpoints/adapters/pop/pop.lora \\"
echo "    --child-adapter checkpoints/adapters/pop/dance_pop/dance_pop.lora \\"
echo "    --output test_merged.pt \\"
echo "    --test-compatibility --test-decode-parity"
echo

# Example 6: Batch training script for full genre hierarchy
echo "6. Batch training for complete genre hierarchy:"
echo "#!/bin/bash"
echo "# Train all parent genres"
echo "parents=(\"pop\" \"rock\" \"country\" \"hiphop_rap\" \"rnb_soul\")"
echo "for parent in \"\${parents[@]}\"; do"
echo "    python train_parent_adapter.py --parent \$parent --pack /style_packs/\$parent"
echo "done"
echo
echo "# Train all child styles"
echo "python train_child_adapter.py --parent pop --child dance_pop --pack /style_packs/pop/dance_pop --parent_lora checkpoints/adapters/pop/pop.lora"
echo "python train_child_adapter.py --parent pop --child synth_pop --pack /style_packs/pop/synth_pop --parent_lora checkpoints/adapters/pop/pop.lora"
echo "python train_child_adapter.py --parent rock --child indie_rock --pack /style_packs/rock/indie_rock --parent_lora checkpoints/adapters/rock/rock.lora"
echo "# ... continue for all child styles"
echo

# Example 7: Advanced merging with custom weights
echo "7. Advanced merging with custom weight combinations:"
echo "# Emphasize parent characteristics:"
echo "python merge_adapters.py --base-model checkpoints/base_model.pt \\"
echo "    --parent-adapter checkpoints/adapters/pop/pop.lora \\"
echo "    --child-adapter checkpoints/adapters/pop/dance_pop/dance_pop.lora \\"
echo "    --parent-weight 1.0 --child-weight 0.3 \\"
echo "    --output dance_pop_parent_heavy.pt"
echo

echo "# Emphasize child characteristics:"
echo "python merge_adapters.py --base-model checkpoints/base_model.pt \\"
echo "    --parent-adapter checkpoints/adapters/pop/pop.lora \\"
echo "    --child-adapter checkpoints/adapters/pop/dance_pop/dance_pop.lora \\"
echo "    --parent-weight 0.7 --child-weight 1.2 \\"
echo "    --output dance_pop_child_heavy.pt"
echo

# Example 8: Testing and validation
echo "8. Running comprehensive adapter tests:"
echo "python test_adapters.py"
echo

echo "=== Configuration Files ==="
echo "LoRA configuration: configs/lora_adapter_config.yaml"
echo "Parent genre configs: configs/genres/<parent>.yaml"
echo "Child style configs: configs/styles/<parent>/<child>.yaml"
echo

echo "=== Directory Structure ==="
echo "checkpoints/adapters/"
echo "├── pop/"
echo "│   ├── pop.lora                    # Parent adapter"
echo "│   ├── dance_pop/"
echo "│   │   └── dance_pop.lora         # Child adapter"
echo "│   ├── synth_pop/"
echo "│   │   └── synth_pop.lora"
echo "│   └── ..."
echo "├── rock/"
echo "│   ├── rock.lora"
echo "│   ├── indie_rock/"
echo "│   └── ..."
echo "└── ..."
echo

echo "merged_models/"
echo "├── pop_model.pt                   # Parent-only merged model"
echo "├── dance_pop_model.pt             # Hierarchical merged model"
echo "├── dance_pop_model.json           # Inference config"
echo "└── ..."
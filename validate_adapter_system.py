#!/usr/bin/env python3
"""
Validation script for hierarchical LoRA adapter system integration.

This script performs static analysis and validation checks to ensure
the adapter training system is properly integrated and ready for use.
"""

import os
import sys
from pathlib import Path


def check_file_exists(file_path: str, description: str) -> bool:
    """Check if a file exists and report status."""
    exists = os.path.exists(file_path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {file_path}")
    return exists


def check_directory_structure() -> bool:
    """Validate the expected directory structure."""
    print("=== Directory Structure Check ===")
    
    required_dirs = [
        ("src/models/adapters", "Adapter module directory"),
        ("configs", "Configuration directory"),
        ("style_packs", "Style packs directory"),
    ]
    
    all_exist = True
    for dir_path, description in required_dirs:
        exists = check_file_exists(dir_path, description)
        all_exist = all_exist and exists
    
    return all_exist


def check_core_files() -> bool:
    """Validate core implementation files."""
    print("\n=== Core Implementation Files ===")
    
    core_files = [
        ("src/models/adapters/__init__.py", "Adapter module init"),
        ("src/models/adapters/lora_layer.py", "LoRA layer implementation"),
        ("src/models/adapters/style_adapter.py", "Style adapter classes"),
        ("src/models/adapters/adapter_merge.py", "Adapter merging utilities"),
        ("src/models/adapters/training_utils.py", "Training utilities"),
        ("train_parent_adapter.py", "Parent adapter training script"),
        ("train_child_adapter.py", "Child adapter training script"),
        ("merge_adapters.py", "Adapter merging script"),
        ("test_adapters.py", "Comprehensive test suite"),
    ]
    
    all_exist = True
    for file_path, description in core_files:
        exists = check_file_exists(file_path, description)
        all_exist = all_exist and exists
    
    return all_exist


def check_configuration_files() -> bool:
    """Validate configuration files."""
    print("\n=== Configuration Files ===")
    
    config_files = [
        ("configs/lora_adapter_config.yaml", "LoRA adapter configuration"),
        ("ADAPTER_TRAINING_README.md", "Adapter training documentation"),
        ("examples_lora_training.sh", "Usage examples script"),
    ]
    
    all_exist = True
    for file_path, description in config_files:
        exists = check_file_exists(file_path, description)
        all_exist = all_exist and exists
    
    return all_exist


def check_style_pack_structure() -> bool:
    """Check for example style pack structure."""
    print("\n=== Style Pack Structure ===")
    
    # Check for some example parent directories
    example_parents = ["pop", "rock", "country"]
    found_parents = []
    
    for parent in example_parents:
        parent_dir = f"style_packs/{parent}"
        if os.path.exists(parent_dir):
            found_parents.append(parent)
            check_file_exists(parent_dir, f"Parent style pack: {parent}")
            
            # Check for child directories
            parent_path = Path(parent_dir)
            if parent_path.exists():
                child_dirs = [d for d in parent_path.iterdir() if d.is_dir()]
                for child_dir in child_dirs:
                    check_file_exists(str(child_dir), f"Child style pack: {child_dir.name}")
    
    if found_parents:
        print(f"Found {len(found_parents)} parent style packs: {', '.join(found_parents)}")
        return True
    else:
        print("✗ No style pack directories found")
        return False


def validate_import_structure() -> bool:
    """Validate that import structure is correct."""
    print("\n=== Import Structure Validation ===")
    
    # Check that __init__.py contains expected exports
    init_file = "src/models/adapters/__init__.py"
    if not os.path.exists(init_file):
        print("✗ Adapter __init__.py missing")
        return False
    
    with open(init_file, 'r') as f:
        content = f.read()
    
    expected_exports = [
        "LoRALayer", "LoRALinear",
        "StyleAdapter", "HierarchicalStyleAdapter",
        "AdapterMerger", "HierarchicalMerger",
        "ParentAdapterTrainer", "ChildAdapterTrainer"
    ]
    
    all_exports_found = True
    for export in expected_exports:
        if export in content:
            print(f"✓ Export found: {export}")
        else:
            print(f"✗ Export missing: {export}")
            all_exports_found = False
    
    return all_exports_found


def check_script_executable_flags() -> bool:
    """Check that scripts have appropriate structure."""
    print("\n=== Script Structure Check ===")
    
    scripts = [
        "train_parent_adapter.py",
        "train_child_adapter.py", 
        "merge_adapters.py",
        "examples_lora_training.sh"
    ]
    
    all_valid = True
    for script in scripts:
        if os.path.exists(script):
            with open(script, 'r') as f:
                first_line = f.readline().strip()
                
            if script.endswith('.py'):
                if first_line.startswith('#!/usr/bin/env python'):
                    print(f"✓ Python script has shebang: {script}")
                else:
                    print(f"✗ Python script missing shebang: {script}")
                    all_valid = False
            elif script.endswith('.sh'):
                if first_line.startswith('#!/bin/bash'):
                    print(f"✓ Bash script has shebang: {script}")
                else:
                    print(f"✗ Bash script missing shebang: {script}")
                    all_valid = False
        else:
            print(f"✗ Script not found: {script}")
            all_valid = False
    
    return all_valid


def generate_usage_summary() -> None:
    """Generate a usage summary."""
    print("\n=== Usage Summary ===")
    print("1. Train parent adapter:")
    print("   python train_parent_adapter.py --parent pop --pack /style_packs/pop")
    print()
    print("2. Train child adapter:")
    print("   python train_child_adapter.py --parent pop --child dance_pop \\")
    print("       --pack /style_packs/pop/dance_pop --parent_lora checkpoints/pop.lora")
    print()
    print("3. Merge adapters:")
    print("   python merge_adapters.py --base-model checkpoints/base_model.pt \\")
    print("       --parent-adapter checkpoints/pop.lora \\")
    print("       --child-adapter checkpoints/dance_pop.lora \\")
    print("       --output merged_dance_pop.pt")
    print()
    print("4. Run tests:")
    print("   python test_adapters.py")
    print()
    print("5. See examples:")
    print("   bash examples_lora_training.sh")


def main():
    """Run all validation checks."""
    print("Hierarchical LoRA Adapter System Validation")
    print("=" * 50)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Core Files", check_core_files),
        ("Configuration Files", check_configuration_files),
        ("Style Pack Structure", check_style_pack_structure),
        ("Import Structure", validate_import_structure),
        ("Script Structure", check_script_executable_flags),
    ]
    
    all_passed = True
    results = {}
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
            all_passed = all_passed and result
        except Exception as e:
            print(f"✗ Error in {check_name}: {e}")
            results[check_name] = False
            all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    for check_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{check_name}: {status}")
    
    overall_status = "PASS" if all_passed else "FAIL"
    print(f"\nOverall: {overall_status}")
    
    if all_passed:
        print("\n✓ Hierarchical LoRA adapter system is ready for use!")
        generate_usage_summary()
    else:
        print("\n✗ Some validation checks failed. Please review the issues above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
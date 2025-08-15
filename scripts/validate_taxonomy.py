#!/usr/bin/env python3
"""
Configuration Validation Script

Validates all genre and style configuration files for proper YAML syntax
and required parameters.
"""

import yaml
import json
from pathlib import Path

def validate_configs():
    """Validate all configuration files."""
    
    base_path = Path("/workspaces/spark-template")
    genres_path = base_path / "configs" / "genres"
    styles_path = base_path / "configs" / "styles"
    packs_path = base_path / "style_packs"
    
    errors = []
    success_count = 0
    
    print("🔍 Validating Genre Taxonomy Configurations")
    print("=" * 50)
    
    # Validate parent genre files
    print("\n📁 Parent Genres:")
    for genre_file in genres_path.glob("*.yaml"):
        try:
            with open(genre_file, 'r') as f:
                genre_data = yaml.safe_load(f)
            
            # Check required fields
            required_fields = ['name', 'display_name', 'description', 'bpm', 'mix_targets']
            missing_fields = [field for field in required_fields if field not in genre_data]
            
            if missing_fields:
                errors.append(f"{genre_file.name}: Missing fields: {missing_fields}")
                print(f"  ❌ {genre_file.name} - Missing: {missing_fields}")
            else:
                print(f"  ✅ {genre_file.name}")
                success_count += 1
                
        except yaml.YAMLError as e:
            errors.append(f"{genre_file.name}: YAML syntax error: {e}")
            print(f"  ❌ {genre_file.name} - YAML Error: {e}")
        except Exception as e:
            errors.append(f"{genre_file.name}: {e}")
            print(f"  ❌ {genre_file.name} - Error: {e}")
    
    # Validate sub-genre files
    print("\n📂 Sub-Genres:")
    for parent_dir in styles_path.iterdir():
        if parent_dir.is_dir():
            print(f"\n  📁 {parent_dir.name}:")
            for style_file in parent_dir.glob("*.yaml"):
                try:
                    with open(style_file, 'r') as f:
                        style_data = yaml.safe_load(f)
                    
                    # Check required fields for sub-genres
                    required_fields = ['parent', 'name', 'display_name', 'description']
                    missing_fields = [field for field in required_fields if field not in style_data]
                    
                    if missing_fields:
                        errors.append(f"{style_file}: Missing fields: {missing_fields}")
                        print(f"    ❌ {style_file.name} - Missing: {missing_fields}")
                    else:
                        print(f"    ✅ {style_file.name}")
                        success_count += 1
                        
                except yaml.YAMLError as e:
                    errors.append(f"{style_file}: YAML syntax error: {e}")
                    print(f"    ❌ {style_file.name} - YAML Error: {e}")
                except Exception as e:
                    errors.append(f"{style_file}: {e}")
                    print(f"    ❌ {style_file.name} - Error: {e}")
    
    # Validate style pack meta.json files
    print("\n📦 Style Pack Metadata:")
    for pack_dir in packs_path.iterdir():
        if pack_dir.is_dir():
            meta_file = pack_dir / "meta.json"
            if meta_file.exists():
                try:
                    with open(meta_file, 'r') as f:
                        meta_data = json.load(f)
                    
                    required_fields = ['genre', 'display_name', 'description', 'reference_count']
                    missing_fields = [field for field in required_fields if field not in meta_data]
                    
                    if missing_fields:
                        errors.append(f"{meta_file}: Missing fields: {missing_fields}")
                        print(f"  ❌ {pack_dir.name}/meta.json - Missing: {missing_fields}")
                    else:
                        print(f"  ✅ {pack_dir.name}/meta.json")
                        success_count += 1
                        
                except json.JSONDecodeError as e:
                    errors.append(f"{meta_file}: JSON syntax error: {e}")
                    print(f"  ❌ {pack_dir.name}/meta.json - JSON Error: {e}")
                except Exception as e:
                    errors.append(f"{meta_file}: {e}")
                    print(f"  ❌ {pack_dir.name}/meta.json - Error: {e}")
            
            # Check for sub-genre meta files
            for sub_dir in pack_dir.iterdir():
                if sub_dir.is_dir() and sub_dir.name not in ['refs_audio', 'refs_midi']:
                    sub_meta_file = sub_dir / "meta.json"
                    if sub_meta_file.exists():
                        try:
                            with open(sub_meta_file, 'r') as f:
                                meta_data = json.load(f)
                            
                            required_fields = ['genre', 'sub_genre', 'parent', 'display_name']
                            missing_fields = [field for field in required_fields if field not in meta_data]
                            
                            if missing_fields:
                                errors.append(f"{sub_meta_file}: Missing fields: {missing_fields}")
                                print(f"  ❌ {pack_dir.name}/{sub_dir.name}/meta.json - Missing: {missing_fields}")
                            else:
                                print(f"  ✅ {pack_dir.name}/{sub_dir.name}/meta.json")
                                success_count += 1
                                
                        except json.JSONDecodeError as e:
                            errors.append(f"{sub_meta_file}: JSON syntax error: {e}")
                            print(f"  ❌ {pack_dir.name}/{sub_dir.name}/meta.json - JSON Error: {e}")
                        except Exception as e:
                            errors.append(f"{sub_meta_file}: {e}")
                            print(f"  ❌ {pack_dir.name}/{sub_dir.name}/meta.json - Error: {e}")
    
    # Summary
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 20)
    print(f"✅ Valid files: {success_count}")
    print(f"❌ Files with errors: {len(errors)}")
    
    if errors:
        print(f"\n🚨 ERRORS FOUND:")
        for error in errors:
            print(f"  • {error}")
        return False
    else:
        print(f"\n🎉 All configuration files are valid!")
        return True

if __name__ == "__main__":
    validate_configs()
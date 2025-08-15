#!/usr/bin/env python3
"""
Genre Taxonomy Structure Summary

This script provides an overview of the hierarchical genre taxonomy
implemented for the AI music composition system.
"""

import yaml
import json
import os
from pathlib import Path

def load_genre_taxonomy():
    """Load and display the complete genre taxonomy structure."""
    
    base_path = Path("/workspaces/spark-template")
    genres_path = base_path / "configs" / "genres"
    styles_path = base_path / "configs" / "styles"
    packs_path = base_path / "style_packs"
    
    print("🎵 AI Music Composer - Genre Taxonomy")
    print("=" * 50)
    print()
    
    # Load all parent genres
    parent_genres = {}
    for genre_file in genres_path.glob("*.yaml"):
        try:
            with open(genre_file, 'r') as f:
                genre_data = yaml.safe_load(f)
                parent_genres[genre_data['name']] = genre_data
        except Exception as e:
            print(f"Error loading {genre_file}: {e}")
    
    # Display taxonomy structure
    for genre_name, genre_data in parent_genres.items():
        print(f"📁 {genre_data['display_name']} ({genre_name})")
        print(f"   📝 {genre_data['description']}")
        print(f"   🎼 BPM: {genre_data['bpm']['min']}-{genre_data['bpm']['max']} (preferred: {genre_data['bpm']['preferred']})")
        print(f"   🎯 LUFS Target: {genre_data['mix_targets']['lufs']}")
        print(f"   🔊 Spectral Centroid: {genre_data['mix_targets']['spectral_centroid_hz']} Hz")
        
        # List sub-genres
        if 'sub_genres' in genre_data:
            print(f"   📂 Sub-genres:")
            for sub_genre in genre_data['sub_genres']:
                sub_genre_path = styles_path / genre_name / f"{sub_genre}.yaml"
                pack_path = packs_path / genre_name / sub_genre
                
                status = "✅" if sub_genre_path.exists() else "⏳"
                pack_status = "📦" if pack_path.exists() else "📦⏳"
                
                print(f"      {status} {sub_genre} {pack_status}")
                
                # Load sub-genre details if available
                if sub_genre_path.exists():
                    try:
                        with open(sub_genre_path, 'r') as f:
                            sub_data = yaml.safe_load(f)
                            if 'description' in sub_data:
                                print(f"         💭 {sub_data['description']}")
                    except Exception as e:
                        print(f"         ❌ Error loading sub-genre: {e}")
        
        print()
    
    # Summary statistics
    total_parents = len(parent_genres)
    total_sub_genres = sum(len(genre.get('sub_genres', [])) for genre in parent_genres.values())
    
    implemented_configs = 0
    implemented_packs = 0
    
    for genre_name in parent_genres.keys():
        genre_style_path = styles_path / genre_name
        if genre_style_path.exists():
            implemented_configs += len(list(genre_style_path.glob("*.yaml")))
        
        genre_pack_path = packs_path / genre_name
        if genre_pack_path.exists():
            # Count subdirectories (sub-genres) that have both refs_audio and refs_midi
            for sub_dir in genre_pack_path.iterdir():
                if sub_dir.is_dir() and sub_dir.name not in ['refs_audio', 'refs_midi']:
                    if (sub_dir / 'refs_audio').exists() and (sub_dir / 'refs_midi').exists():
                        implemented_packs += 1
    
    print("📊 TAXONOMY SUMMARY")
    print("=" * 20)
    print(f"Parent Genres: {total_parents}")
    print(f"Total Sub-genres: {total_sub_genres}")
    print(f"Implemented Configs: {implemented_configs}")
    print(f"Implemented Style Packs: {implemented_packs}")
    print()
    
    # Configuration structure info
    print("📋 CONFIGURATION STRUCTURE")
    print("=" * 30)
    print("Parent Genre Files: /configs/genres/<parent>.yaml")
    print("Sub-Genre Files:    /configs/styles/<parent>/<child>.yaml")
    print("Style Packs:        /style_packs/<parent>/{refs_audio,refs_midi,meta.json}")
    print("Sub-Style Packs:    /style_packs/<parent>/<child>/{refs_audio,refs_midi,meta.json}")
    print()
    
    # Key configuration parameters
    print("🔧 KEY PARAMETERS")
    print("=" * 20)
    print("• BPM ranges and preferred tempos")
    print("• Key signatures and chord progressions")
    print("• Instrument roles and arrangements")
    print("• Mix targets (LUFS, spectral characteristics)")
    print("• Frequency balance and dynamics")
    print("• QA thresholds for quality control")
    print("• Groove characteristics and timing")
    print()

if __name__ == "__main__":
    load_genre_taxonomy()
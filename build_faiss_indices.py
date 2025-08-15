#!/usr/bin/env python3
"""
Build FAISS indices for hierarchical style retrieval.
CLI tool for indexing tokenized MIDI patterns with parent-child relationships.
"""

import argparse
import logging
import json
from pathlib import Path

from style.faiss_index import build_hierarchical_indices, HierarchicalFAISSIndex


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS indices for hierarchical style pattern retrieval"
    )
    
    parser.add_argument(
        "--style_packs_dir",
        type=str,
        default="style_packs",
        help="Directory containing style packs with tokenized MIDI"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="indices",
        help="Output directory for FAISS indices"
    )
    
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=512,
        help="Dimension of pattern embeddings"
    )
    
    parser.add_argument(
        "--child_weight",
        type=float,
        default=1.5,
        help="Weight multiplier for child patterns during fusion"
    )
    
    parser.add_argument(
        "--parents",
        nargs="*",
        help="Specific parent genres to index (default: all)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Building hierarchical FAISS indices...")
    logger.info(f"Style packs: {args.style_packs_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Embedding dim: {args.embedding_dim}")
    
    # Build indices
    index = HierarchicalFAISSIndex(args.embedding_dim)
    
    # Build parent indices (optionally filtered)
    if args.parents:
        logger.info(f"Building indices for specified parents: {args.parents}")
        style_packs_path = Path(args.style_packs_dir)
        
        for parent in args.parents:
            parent_dir = style_packs_path / parent
            if parent_dir.exists():
                patterns = index._load_parent_patterns(parent_dir, parent)
                if patterns:
                    embeddings = index._create_pattern_embeddings(patterns)
                    
                    import faiss
                    faiss_index = faiss.IndexFlatIP(args.embedding_dim)
                    faiss_index.add(embeddings.astype('float32'))
                    
                    index.parent_indices[parent] = faiss_index
                    index.parent_patterns[parent] = patterns
                    
                    logger.info(f"Built index for {parent}: {len(patterns)} patterns")
                else:
                    logger.warning(f"No patterns found for {parent}")
            else:
                logger.error(f"Parent directory not found: {parent_dir}")
    else:
        # Build all parent indices
        index.build_parent_indices(args.style_packs_dir)
    
    # Register child patterns
    style_packs_path = Path(args.style_packs_dir)
    
    parents_to_process = args.parents if args.parents else [d.name for d in style_packs_path.iterdir() if d.is_dir()]
    
    for parent_genre in parents_to_process:
        parent_dir = style_packs_path / parent_genre
        if not parent_dir.exists():
            continue
            
        # Find child directories
        for child_dir in parent_dir.iterdir():
            if child_dir.is_dir() and child_dir.name not in ['refs_audio', 'refs_midi']:
                child_genre = child_dir.name
                logger.info(f"Registering child patterns: {parent_genre}/{child_genre}")
                
                index.register_child_patterns(
                    parent_genre, child_genre, 
                    args.style_packs_dir, 
                    args.child_weight
                )
    
    # Save indices
    index.save_indices(args.output_dir)
    
    # Save build info
    build_info = {
        "embedding_dim": args.embedding_dim,
        "child_weight": args.child_weight,
        "parent_genres": list(index.parent_indices.keys()),
        "child_patterns": list(index.child_patterns.keys()),
        "total_parent_patterns": sum(len(patterns) for patterns in index.parent_patterns.values()),
        "total_child_patterns": sum(len(patterns) for patterns in index.child_patterns.values())
    }
    
    with open(output_path / "build_info.json", 'w') as f:
        json.dump(build_info, f, indent=2)
    
    logger.info("=" * 50)
    logger.info("FAISS Index Build Complete!")
    logger.info(f"Parent genres: {len(index.parent_indices)}")
    logger.info(f"Child pattern groups: {len(index.child_patterns)}")
    logger.info(f"Total parent patterns: {build_info['total_parent_patterns']}")
    logger.info(f"Total child patterns: {build_info['total_child_patterns']}")
    logger.info(f"Output saved to: {args.output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
# Hierarchical FAISS Retrieval System

This module implements a hierarchical FAISS indexing system for style-aware pattern retrieval with parent-child bias weighting. The system enables sophisticated style conditioning during music generation by retrieving similar patterns from a hierarchical style taxonomy.

## Overview

The system supports:
- **Parent indices**: FAISS indices built from tokenized MIDI patterns in each parent genre
- **Child patterns**: Additional patterns with higher fusion weights for sub-genre bias
- **Retrieval fusion**: N-gram matching and similarity search with logit fusion during generation
- **Hierarchical bias**: Child patterns receive higher weights during retrieval and fusion

## Architecture

```
style_packs/
├── pop/
│   ├── refs_midi/           # Parent patterns for indexing
│   │   ├── *.tokens         # Tokenized sequences  
│   │   └── *.json          # JSON format with tokenized_bars
│   ├── dance_pop/          # Child genre
│   │   └── refs_midi/      # Child patterns (higher weight)
│   └── pop_rock/
└── rock/
    ├── refs_midi/
    └── punk/
        └── refs_midi/
```

## Core Components

### 1. HierarchicalFAISSIndex (`style/faiss_index.py`)

Main indexing system that:
- Loads tokenized patterns from `refs_midi` directories
- Creates embeddings using token hashing (replaceable with neural encoders)
- Builds FAISS indices for parent genres
- Registers child patterns with configurable weights
- Supports saving/loading indices

### 2. RetrievalFusion (`style/retrieval_fusion.py`)

Fusion system for biasing generation:
- Extracts n-grams from generated sequences
- Retrieves similar patterns with parent + child bias
- Applies weighted fusion to model logits
- Supports configurable fusion parameters

### 3. CLI Tools

**Build indices:**
```bash
python build_faiss_indices.py \
    --style_packs_dir style_packs \
    --output_dir indices \
    --embedding_dim 512 \
    --child_weight 1.5
```

**Generate with retrieval:**
```bash
python sample_with_retrieval.py \
    --model_path checkpoints/model.pt \
    --vocab_file models/vocab.json \
    --family_index pop \
    --child_bias 0.3 \
    --child_genre dance_pop \
    --fusion_weight 0.1
```

## Usage Examples

### Building Indices

```python
from style.faiss_index import build_hierarchical_indices

# Build all parent indices with child patterns
index = build_hierarchical_indices(
    style_packs_dir="style_packs",
    output_dir="indices", 
    embedding_dim=512
)

# Save for later use
index.save_indices("indices")
```

### Retrieval During Generation

```python
from style.retrieval_fusion import create_retrieval_fusion

# Create retrieval fusion system
retrieval_fusion = create_retrieval_fusion(
    faiss_index_dir="indices",
    vocab_file="models/vocab.json",
    family_index="pop",
    child_bias=0.3,
    child_genre="dance_pop",
    fusion_weight=0.1
)

# Apply during generation
biased_logits = retrieval_fusion.apply_retrieval_bias(
    logits=model_logits,
    generated_tokens=previous_tokens
)
```

### Pattern Retrieval

```python
# Retrieve similar patterns with bias
similar_patterns = index.retrieve_similar_patterns(
    query_tokens=["STYLE=pop", "TEMPO=120", "CHORD=C"],
    parent_genre="pop",
    child_genre="dance_pop", 
    child_bias=0.3,
    k=5
)

for pattern, similarity in similar_patterns:
    print(f"Pattern: {pattern.tokens[:5]}...")
    print(f"Similarity: {similarity:.3f}")
    print(f"Weight: {pattern.weight}")
```

## Configuration

### RetrievalConfig Parameters

- `family_index`: Parent genre name (e.g., "pop", "rock")
- `child_bias`: Bias weight for child patterns (0.0-1.0)  
- `child_genre`: Specific child genre for bias
- `fusion_weight`: Weight for retrieval vs. model logits (0.0-1.0)
- `ngram_size`: Size of n-grams for pattern matching (default: 3)
- `top_k_patterns`: Number of patterns to retrieve (default: 5)

### Style Pack Structure

Each parent genre should have:
- `refs_midi/`: Directory with tokenized patterns
  - `*.tokens`: Space-separated token files
  - `*.json`: JSON format with `tokenized_bars` array
- Child directories with same structure + higher weights

## Pattern Formats

### Token File Format (.tokens)
```
STYLE=pop TEMPO=120 KEY=C SECTION=VERSE BAR POS=1 CHORD=C NOTE_ON 60 VEL=80 DUR=4
```

### JSON Format (.json)
```json
{
  "style": "pop",
  "bars": 4,
  "tokenized_bars": [
    ["STYLE=pop", "TEMPO=120", "CHORD=C", "NOTE_ON", "60", "VEL=80"],
    ["STYLE=pop", "TEMPO=120", "CHORD=F", "NOTE_ON", "65", "VEL=75"]
  ]
}
```

## Algorithm Details

### Embedding Creation
Current implementation uses simple token hashing:
- Maps tokens to embedding dimensions via hash function
- Applies positional weighting to capture sequence order
- Normalizes embeddings for cosine similarity

**Note**: Can be replaced with neural encoders for better semantic representations.

### Retrieval Fusion
1. Extract recent n-gram from generated sequence
2. Look up n-gram in cached pattern transitions
3. Weight parent patterns normally, child patterns with bias
4. Create probability distribution over next tokens
5. Fuse with model logits using weighted interpolation

### Child Bias Application
- Child patterns get weight multiplier (default 1.5x)
- During retrieval, child similarities boosted by bias factor
- Final fusion weight combines base + bias: `fusion_weight * (1 + child_bias)`

## Testing

Run comprehensive tests:
```bash
python test_faiss_retrieval.py
```

Tests cover:
- Pattern loading from both token and JSON formats
- FAISS index building and querying
- Child bias application
- Retrieval fusion logic
- Save/load functionality

## Performance Considerations

- **Memory**: Each parent index stores all patterns in memory
- **Speed**: FAISS provides fast similarity search (~ms for thousands of patterns)
- **Scalability**: Consider approximate indices (IVF, HNSW) for large pattern sets
- **Embedding Quality**: Neural encoders would improve retrieval quality vs. simple hashing

## Integration Points

### With Tokenizer
Requires compatible vocabulary and token format:
```python
# Must match tokenizer vocab
vocab = {"STYLE=pop": 0, "TEMPO=120": 1, ...}
```

### With Generator Models
Apply during nucleus sampling:
```python
# Standard generation loop
logits = model(input_ids).logits[:, -1]

# Apply retrieval bias
if retrieval_fusion:
    logits = retrieval_fusion.apply_retrieval_bias(logits, generated_tokens)

# Continue with sampling
probs = F.softmax(logits / temperature, dim=-1)
```

### With Training Pipeline
Can be used for:
- Data augmentation during training
- Evaluation of style consistency
- Analysis of learned representations

## Future Improvements

1. **Neural Embeddings**: Replace token hashing with learned audio/MIDI encoders
2. **Dynamic Weighting**: Learn optimal fusion weights per context
3. **Multi-Modal**: Combine MIDI patterns with audio features
4. **Approximate Search**: Use FAISS approximate indices for better scalability
5. **Online Learning**: Update indices with new patterns during generation

## Dependencies

Install requirements:
```bash
pip install -r requirements-faiss.txt
```

Core dependencies:
- `faiss-cpu>=1.7.4`: Fast similarity search
- `numpy>=1.21.0`: Numerical operations
- `torch>=1.12.0`: Model integration
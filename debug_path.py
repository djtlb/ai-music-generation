import sys
import os

print("Current Working Directory:", os.getcwd())
print("sys.path:")
for p in sys.path:
    print(p)

# Check if the src directory is in the path
src_in_path = any('ai-music-generation/src' in p for p in sys.path)
print("\n'src' directory in sys.path:", src_in_path)

# Check for the existence of the module
tokenizer_path = os.path.join(os.getcwd(), 'src', 'models', 'tokenizer.py')
print(f"Expected tokenizer path: {tokenizer_path}")
print(f"Tokenizer path exists: {os.path.exists(tokenizer_path)}")

# Let's try to import it
try:
    from models.tokenizer import MIDITokenizer
    print("\nSuccessfully imported MIDITokenizer!")
except ImportError as e:
    print(f"\nFailed to import MIDITokenizer: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred during import: {e}")

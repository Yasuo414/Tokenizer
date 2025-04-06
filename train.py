import argparse
import pathlib
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer'))
from Tokenizer.tokenizer import Tokenizer
from Tokenizer.config import Config

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_text_files(directory):
    texts = []
    directory = pathlib.Path(directory)

    for file_path in directory.glob('**/*.txt'):
        try:
            texts.append(read_text_file(file_path))
            print(f"Read file: {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return texts

def train_tokenizer(args):
    config = Config(
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        max_word_len=args.max_word_len,
        min_subword_len=args.min_subword_len,
        lowercase=args.lowercase,
        normalize_unicode=args.normalize_unicode,
        remove_accents=args.remove_accents,
        clean_text=args.clean_text,
        merge_method=args.merge_method
    )
    
    tokenizer = Tokenizer(config)
    texts = []

    if args.input_file:
        print(f"Reading input file: {args.input_file}")
        texts.append(read_text_file(args.input_file))
    elif args.input_dir:
        print(f"Reading text files from directory: {args.input_dir}")
        texts.extend(read_text_files(args.input_dir))
    else:
        raise ValueError("Either --input_file or --input_dir must be provided")
    
    if not texts:
        raise ValueError("No text data found for training")
    
    print(f"Training tokenizer on {len(texts)} text samples")
    print(f"Configuration: {config.__dict__}")
    
    if len(texts) < 5 and config.min_freq > 1:
        print(f"Small dataset detected ({len(texts)} samples). Adjusting min_freq from {config.min_freq} to 1")
        config.min_freq = 1
    
    tokenizer.fit(texts)
    
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir)
    
    print(f"Tokenizer trained and saved to {output_dir}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    
    if args.test_text:
        test_text = ' '.join(args.test_text)
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print("\nTest encoding/decoding:")
        print(f"Original: {test_text}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {decoded}")

def main():
    parser = argparse.ArgumentParser(description="Train a tokenizer on text data")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_file", type=str, help="Path to a text file for training")
    input_group.add_argument("--input_dir", type=str, help="Path to a directory containing text files for training")

    parser.add_argument("--output_dir", type=str, default="tokenizer_model", help="Directory to save the trained tokenizer (default: tokenizer_model)")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Size of the vocabulary (default: 30000)")
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum frequency for a token to be included (default: 2)")
    parser.add_argument("--max_word_len", type=int, default=100, help="Maximum word length (default: 100)")
    parser.add_argument("--min_subword_len", type=int, default=2, help="Minimum subword length (default: 2)")
    parser.add_argument("--lowercase", action="store_true", default=True, help="Convert text to lowercase (default: True)")
    parser.add_argument("--no_lowercase", action="store_false", dest="lowercase", help="Do not convert text to lowercase")
    parser.add_argument("--normalize_unicode", action="store_true", default=True, help="Normalize Unicode characters (default: True)")
    parser.add_argument("--no_normalize_unicode", action="store_false", dest="normalize_unicode", help="Do not normalize Unicode characters")
    parser.add_argument("--remove_accents", action="store_true", default=False, help="Remove accents from characters (default: False)")
    parser.add_argument("--clean_text", action="store_true", default=True, help="Clean text before tokenization (default: True)")
    parser.add_argument("--no_clean_text", action="store_false", dest="clean_text", help="Do not clean text before tokenization")
    parser.add_argument("--merge_method", type=str, choices=["bpe", "wordpiece"], default="bpe", help="Tokenization method: 'bpe' or 'wordpiece' (default: bpe)")
    
    parser.add_argument("--test_text", type=str, nargs='+', help="Optional text to test the trained tokenizer (can be multiple words)")

    args = parser.parse_args()
    train_tokenizer(args)

if __name__ == "__main__":
    main()
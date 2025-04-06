# Tokenizer

A lightweight and customizable tokenizer implementation for natural language processing tasks, supporting both BPE (Byte Pair Encoding) and WordPiece tokenization strategies.

## Features

- Support for both BPE and WordPiece tokenization methods
- Customizable vocabulary size
- Special token handling
- Text normalization options:
  - Unicode normalization
  - Accent removal
  - Lowercasing
  - Whitespace normalization
- Entity recognition for URLs, emails, and numbers
- Easy serialization and loading of models

## Installation

```bash
git clone https://github.com/SamuelHusek/simple-tokenizer.git
cd simple-tokenizer
pip install -e .
```

## Quick Start

```python
from tokenizer import Tokenizer, Config

# Create a tokenizer with default configuration
tokenizer = Tokenizer()

# Or with custom configuration
config = Config(
    vocab_size=10000,
    min_freq=3,
    lowercase=True,
    merge_method="bpe"  # Use "wordpiece" for WordPiece tokenization
)
tokenizer = Tokenizer(config)

# Train the tokenizer on your texts
texts = ["This is an example sentence.", "Another example text to tokenize."]
tokenizer.fit(texts)

# Encode text to token IDs
tokens = tokenizer.encode("This is a test sentence.")
print(tokens)  # [2, 3, 4, 7, 8, 5]

# Decode token IDs back to text
decoded = tokenizer.decode(tokens)
print(decoded)  # "this is a test sentence"

# Save the tokenizer
tokenizer.save("path/to/save")

# Load the tokenizer
loaded_tokenizer = Tokenizer.load("path/to/save")
```

## Training Script Usage

The project includes a command-line script `train.py` for training the tokenizer on text data:

```bash
python train.py --input_file path/to/file.txt --output_dir ./my_tokenizer
```

Or use a directory containing multiple text files:

```bash
python train.py --input_dir path/to/text_directory --output_dir ./my_tokenizer
```

### Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input_file` | Path to a single text file for training | None |
| `--input_dir` | Path to a directory containing text files for training | None |
| `--output_dir` | Directory to save the trained tokenizer | `tokenizer_model` |
| `--vocab_size` | Size of the vocabulary | 30000 |
| `--min_freq` | Minimum frequency for a token to be included | 2 |
| `--max_word_len` | Maximum word length | 100 |
| `--min_subword_len` | Minimum subword length | 2 |
| `--lowercase` | Convert text to lowercase | True |
| `--no_lowercase` | Do not convert text to lowercase | False |
| `--normalize_unicode` | Normalize Unicode characters | True |
| `--no_normalize_unicode` | Do not normalize Unicode characters | False |
| `--remove_accents` | Remove accents from characters | False |
| `--clean_text` | Clean text before tokenization | True |
| `--no_clean_text` | Do not clean text before tokenization | False |
| `--merge_method` | Tokenization method: 'bpe' or 'wordpiece' | `bpe` |
| `--test_text` | Optional text to test the trained tokenizer | None |

### Example

Train a BPE tokenizer with a vocabulary size of 5000:

```bash
python train.py --input_dir ./my_texts --vocab_size 5000 --output_dir ./my_tokenizer
```

Train a WordPiece tokenizer and test it:

```bash
python train.py --input_file ./corpus.txt --merge_method wordpiece --test_text "This is a test sentence"
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `vocab_size` | Maximum vocabulary size | 30000 |
| `min_freq` | Minimum frequency for a word to be included | 2 |
| `special_tokens` | List of special tokens | `["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[MASK]"]` |
| `max_word_len` | Maximum word length | 100 |
| `min_subword_len` | Minimum subword length | 2 |
| `lowercase` | Convert text to lowercase | True |
| `normalize_unicode` | Apply Unicode normalization | True |
| `remove_accents` | Remove accent marks | False |
| `clean_text` | Apply text cleaning | True |
| `merge_method` | Tokenization method ("bpe" or "wordpiece") | "bpe" |

## Architecture

The project consists of the following main components:

- `Config`: Configuration class for tokenizer settings
- `Tokenizer`: Core tokenizer implementation
- `Normalizer`: Text normalization utilities
- `train.py`: Command-line script for training the tokenizer

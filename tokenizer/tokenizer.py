from . import config
from . import normalizer
import typing
import collections
import unicodedata
import tqdm
import pathlib
import pickle

class Tokenizer:
    def __init__(self, config_: config.Config = None):
        self.config = config_ or config.Config()
        self.normalizer = normalizer.Normalizer()

        self.vocab: typing.Dict[str, int] = {}
        self.reverse_vocab: typing.Dict[int, str] = {}
        self.subword_vocab: typing.Dict[str, int] = {}
        self.word_freqs: collections.Counter = collections.Counter()
        self.merges: typing.Dict[typing.Tuple[str, str], int] = {}

        self._initialize_special_tokens()
    
    def _initialize_special_tokens(self):
        for i, token in enumerate(self.config.special_tokens):
            self.vocab[token] = i
            self.reverse_vocab[i] = token
        
        self.pad_token_ID = self.vocab["[PAD]"]
        self.unk_token_ID = self.vocab["[UNK]"]
        self.bos_token_ID = self.vocab["[BOS]"]
        self.eos_token_ID = self.vocab["[EOS]"]
        self.sep_token_ID = self.vocab["[SEP]"]
        self.mask_token_ID = self.vocab["[MASK]"]
    
    def _clean_text(self, text: str) -> str:
        if not self.config.clean_text:
            return text
        
        if self.config.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        
        if self.config.remove_accents:
            text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

        words = text.split()
        cleaned_words = []

        for word in words:
            if self.normalizer.is_url(word):
                cleaned_words.append("[URL]")
            elif self.normalizer.is_email(word):
                cleaned_words.append("[EMAIL]")
            elif self.normalizer.is_number(word):
                cleaned_words.append("[NUM]")
            else:
                tokens = self.normalizer.split_on_punctuation(word)
                cleaned_words.extend(tokens)
        
        text = self.normalizer.normalize_whitespace(" ".join(cleaned_words))

        if self.config.lowercase:
            text = text.lower()
        
        return text
    
    def _learn_bpe_merges(self, words: typing.List[str], num_merges: int):
        word_splits = {
            word: list(word) for word in words
        }

        for i in tqdm.tqdm(range(num_merges), desc="Learning BPE merges"):
            pairs = collections.Counter()

            for splits in word_splits.values():
                for j in range(len(splits) - 1):
                    pairs[(splits[j], splits[j + 1])] += 1
            
            if not pairs:
                break

            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            self.merges[best_pair] = i

            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.vocab:
                index = len(self.vocab)
                self.vocab[merged_token] = index
                self.reverse_vocab[index] = merged_token

            new_word_splits = {}
            for word, splits in word_splits.items():
                new_splits = []
                j = 0

                while j < len(splits):
                    if j < len(splits) - 1 and (splits[j], splits[j + 1]) == best_pair:
                        new_splits.append(splits[j] + splits[j + 1])
                        j += 2
                    else:
                        new_splits.append(splits[j])
                        j += 1
                
                new_word_splits[word] = new_splits
            
            word_splits = new_word_splits
    
    def _get_subword_vocabulary(self, words: typing.List[str]) -> collections.Counter:
        subwords = collections.Counter()

        for word in words:
            word = f"##" + word + "##"
            word_len = len(word)

            for start in range(word_len):
                for end in range(start + self.config.min_subword_len, min(start + self.config.max_word_len + 1, word_len + 1)):
                    subword = word[start:end]

                    if "##" not in subword[2:-2]:
                        subwords[subword] += self.word_freqs[word[2:-2]]
        
        return subwords
    
    def fit(self, texts: typing.List[str]):
        processed_texts = [self._clean_text(text) for text in tqdm.tqdm(texts, desc="Preprocessing")]

        for text in processed_texts:
            self.word_freqs.update(text.split())
        
        self.word_freqs = collections.Counter({
            word: freq for word, freq in self.word_freqs.items()
            if freq >= self.config.min_freq
        })

        if self.config.merge_method == "bpe":
            words = list(self.word_freqs.keys())
            num_merges = self.config.vocab_size - len(self.config.special_tokens)
            self._learn_bpe_merges(words, num_merges)
        else:
            subwords = self._get_subword_vocabulary(list(self.word_freqs.keys()))

            vocab_size = self.config.vocab_size - len(self.config.special_tokens)
            selected_subwords = sorted(subwords.items(), key=lambda x: (-x[1], x[0]))[:vocab_size]

            for subword, _ in selected_subwords:
                index = len(self.vocab)
                self.vocab[subword] = index
                self.reverse_vocab[index] = subword
    
    def encode(self, text: str, add_special_tokens: bool = True) -> typing.List[int]:
        text = self._clean_text(text)
        words = text.split()

        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_ID)
        
        for word in words:
            if self.config.merge_method == "bpe":
                word_tokens = self._tokenize_word_bpe(word)
            else:
                word_tokens = self._tokenize_word_wordpiece(word)
            
            tokens.extend(self.vocab.get(t, self.unk_token_ID) for t in word_tokens)
        
        if add_special_tokens:
            tokens.append(self.eos_token_ID)
        
        return tokens
    
    def _tokenize_word_bpe(self, word: str) -> typing.List[str]:
        if not word:
            return []
        
        if word in self.vocab:
            return [word]
        
        chars = list(word)
        while len(chars) > 1:
            min_pair = None
            min_rank = float("inf")

            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])

                if pair in self.merges:
                    rank = self.merges[pair]

                    if rank < min_rank:
                        min_pair = pair
                        min_rank = rank
            
            if min_pair is None:
                break

            new_chars = []
            i = 0

            while i < len(chars):
                if i < len(chars) - 1 and (chars[i], chars[i + 1]) == min_pair:
                    new_chars.append(chars[i] + chars[i + 1])
                    i += 2
                else:
                    new_chars.append(chars[i])
                    i += 1
            
            chars = new_chars
        
        return chars
    
    def _tokenize_word_wordpiece(self, word: str) -> typing.List[str]:
        if not word:
            return []
        
        if word in self.vocab:
            return [word]
        
        tokens = []
        start = 0

        while start < len(word):
            end = len(word)
            current_substr = None

            while start < end:
                substr = word[start:end]

                if start > 0:
                    substr = "##" + substr
                
                if substr in self.vocab:
                    current_substr = substr
                    break

                end -= 1
            
            if current_substr is None:
                return ["[UNK]"]
            
            tokens.append(current_substr)
            start = end
        
        return tokens
    
    def decode(self, token_IDs: typing.List[int], remove_special_tokens: bool = True) -> str:
        tokens = []

        for token_ID in token_IDs:
            if token_ID in self.reverse_vocab:
                token = self.reverse_vocab[token_ID]

                if remove_special_tokens and token in self.config.special_tokens:
                    continue

                if token.startswith("##"):
                    token = token[2:]
                
                tokens.append(token)

        return " ".join(tokens)
    
    def save(self, path: typing.Union[str, pathlib.Path]):
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.config.save(path)

        data = {
            "vocab": self.vocab,
            "reverse": self.reverse_vocab,
            "subword": self.subword_vocab,
            "word_freqs": dict(self.word_freqs),
            "merges": dict(self.merges)
        }

        with open(path / "tokenizer.model", "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: typing.Union[str, pathlib.Path]) -> "Tokenizer":
        path = pathlib.Path(path)

        cfg = config.Config.load(path)
        tokenizer = cls(cfg)

        with open(path / "tokenizer.model", "rb") as f:
            data = pickle.load(f)
        
        tokenizer.vocab = data["vocab"]
        tokenizer.reverse_vocab = data["reverse"]
        tokenizer.subword_vocab = data["subword"]
        tokenizer.word_freqs = collections.Counter(data["word_freqs"])
        tokenizer.merges = dict(data["merges"])

        return tokenizer
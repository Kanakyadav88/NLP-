import json
from collections import defaultdict
import re


class WordPieceTokenizer:
    def __init__(self):
        self.vocab = ["[PAD]", "<UNK>"]
        self.word_freqs = defaultdict(int)
        self.splits = {}
        self.contractions = {
            "i'm": "im",
            "you're": "you are",
            "he's": "he is",
            "she's": "she is",
            "it's": "it is",
            "we're": "we are",
            "they're": "they are",
            "can't": "can not",
            "won't": "will not",
            "n't": " not",
            "couldn't": "could not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "doesn't": "does not",
            "don't": "do not",
            "didn't": "did not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
        }

    def preprocess_data(self, corpus_file):
        with open(corpus_file, "r", encoding="utf-8") as file:
            corpus = file.readlines()

        processed_corpus = []
        for line in corpus:
            line = line.lower().strip()
            line = self.expand_contractions(line)  # Expand contractions
            line = re.sub(
                r"[^a-z0-9\s']", "", line
            )  # Keep apostrophes for contractions
            words = line.split()
            for word in words:
                self.word_freqs[word] += 1
            processed_corpus.append(words)

        return processed_corpus

    def expand_contractions(self, text):
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)
        return text

    def tokenize(self, text):
        text = text.lower().strip()
        text = self.expand_contractions(text)
        text = re.sub(r"[^a-z0-9\s']", "", text)  # Keep apostrophes
        words = text.split()
        encoded_words = [self.encode_word(word) for word in words]
        return sum(encoded_words, [])

    def construct_vocabulary(self, vocab_file, vocab_size=5000):
        alphabet = set()
        for word in self.word_freqs.keys():
            alphabet.add(word[0])
            for letter in word[1:]:
                alphabet.add(f"##{letter}")

        self.vocab += sorted(alphabet)

        self.splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in self.word_freqs.keys()
        }

        while len(self.vocab) < vocab_size:
            scores = self.compute_pair_scores()
            if not scores:
                break
            best_pair = max(scores, key=scores.get)
            self.splits = self.merge_pair(*best_pair)
            new_token = (
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith("##")
                else best_pair[0] + best_pair[1]
            )
            self.vocab.append(new_token)

        with open(vocab_file, "w", encoding="utf-8") as file:
            for token in self.vocab:
                file.write(token + "\n")

    def compute_pair_scores(self):
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)

        for word, freq in self.word_freqs.items():
            split = self.splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        return {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]] + 1e-10)
            for pair, freq in pair_freqs.items()
        }

    def merge_pair(self, a, b):
        for word in self.word_freqs:
            split = self.splits[word]
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                    new_split.append(a + (b[2:] if b.startswith("##") else b))
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            self.splits[word] = new_split
        return self.splits

    def encode_word(self, word):
        tokens = []
        while word:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                return ["<UNK>"]
            tokens.append(word[:i])
            word = word[i:]
            if word:
                word = f"##{word}"
        return tokens

    def tokenize_test_data(self, test_file, output_file):
        with open(test_file, "r", encoding="utf-8") as file:
            test_data = json.load(file)

        tokenized_data = {
            entry["id"]: self.tokenize(entry["sentence"]) for entry in test_data
        }

        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(tokenized_data, file, indent=4)

    def tokenize_corpus(self, corpus_file, output_file):
        with open(corpus_file, "r", encoding="utf-8") as file:
            corpus = file.readlines()

        tokenized_corpus = [" ".join(self.tokenize(line)) for line in corpus]

        with open(output_file, "w", encoding="utf-8") as file:
            file.write("\n".join(tokenized_corpus))


if __name__ == "__main__":
    tokenizer = WordPieceTokenizer()
    tokenizer.preprocess_data("corpus.txt")
    size = int(input("Enter size of vocab : "))
    tokenizer.construct_vocabulary("vocabulary_54.txt", size)
    tokenizer.tokenize_test_data("test.json", "tokenized_54.json")
    tokenizer.tokenize_corpus("corpus.txt", "tokenized_corpus.txt")
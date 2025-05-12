import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

# from task1 import WordPieceTokenizer

from task1 import WordPieceTokenizer

import torch.nn.functional as F


class Word2VecDataset(Dataset):
    def __init__(self, corpus_file, tokenizer=None, window_size=2):
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = WordPieceTokenizer()
        self.window_size = window_size

        with open(corpus_file, "r", encoding="utf-8") as file:
            text = file.read()  
            tokenized_text = []  
            words = text.split()  

            for word in words:  
                tokenized_text.append(word) 
        num_tokens = len(tokenized_text)
        print(f"First 20 tokens: {tokenized_text[:20]}")

        self.data = []
        for i in range(self.window_size, num_tokens - self.window_size):

            context = (
                tokenized_text[i - self.window_size : i]
                + tokenized_text[i + 1 : i + 1 + self.window_size]
            )
            target = tokenized_text[i]
            self.data.append((context, target))

        print("This is the data: ", self.data)

        self.vocab = self.tokenizer.vocab


        if "<UNK>" not in self.vocab:
            print("<UNK> not found in vocab, adding it.")
            self.vocab.append("<UNK>")  

        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}

        print("Vocabulary:", self.vocab)
        print("Word to index mapping:", self.word_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]

        context_indices = [
            self.word_to_idx.get(word, self.word_to_idx["<UNK>"]) for word in context
        ]
        target_index = self.word_to_idx.get(target, self.word_to_idx["<UNK>"])

        if "<UNK>" in context_indices:
            print(f"Unknown word in context: {context}, target: {target}")

        return torch.tensor(context_indices, dtype=torch.long), torch.tensor(
            target_index, dtype=torch.long
        )


def collate_fn(batch):
    contexts, targets = zip(*batch)
    return torch.stack(contexts), torch.tensor(targets, dtype=torch.long)


class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.3)
        self.linear1 = nn.Linear(embed_size, embed_size // 2)
        self.linear2 = nn.Linear(embed_size // 2, vocab_size)

    def forward(self, context):
        embedded = self.embeddings(context)
        embedded = embedded.mean(dim=1)  # mean pooling
        embedded = self.dropout(embedded)
        embedded = F.relu(self.linear1(embedded))
        output = self.linear2(embedded)
        return output


def train(
    model, dataset, epochs=5, batch_size=16, lr=0.0001
): 
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=0.0001
    ) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=4, verbose=True
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience = 15 
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for context, target in train_loader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=0.25
            )  
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for context, target in val_loader:
                output = model(context)
                loss = criterion(output, target)
                val_loss = val_loss + loss.item()
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "word2vec_cbow_model.pth")
            patience_counter = 0
        else:
            patience_counter = patience_counter + 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        print(
            f"Epoch {epoch + 1} - Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_loss:.4f}"
        )

    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    plt.savefig("training_loss.png")
    plt.show()


def cosine_similarity(vec1, vec2):
    return F.cosine_similarity(vec1, vec2, dim=0).item()


def find_triplets(model, word_to_idx, vocab):
    word_list = list(vocab)
    word_embeddings = model.embeddings.weight
    similarities = {}
    for word1 in word_list:
        for word2 in word_list:
            if word1 != word2:
                similarities[(word1, word2)] = cosine_similarity(
                    word_embeddings[word_to_idx[word1]],
                    word_embeddings[word_to_idx[word2]],
                )
    sorted_pairs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    triplets = []
    for i in range(len(sorted_pairs)):
        if len(triplets) == 2:
            break
        (word1, word2), sim_high = sorted_pairs[i]
        for (word3, _), sim_low in reversed(sorted_pairs):
            if word3 != word1 and word3 != word2 and sim_low < 0.2:
                triplets.append((word1, word2, word3, sim_high, sim_low))
                break
    print("\nSelected Triplets:")
    for word1, word2, word3, sim_high, sim_low in triplets:
        print("Similar:", word1, word2, "Cosine Similarity:", round(sim_high, 4))
        print("Dissimilar:", word3, "Cosine Similarity:", round(sim_low, 4), "\n")
    return triplets


def preprocess_data(corpus_file, tokenizer):
    # Preprocessing steps moved here
    tokenizer.preprocess_data(corpus_file)
    tokenizer.construct_vocabulary("vocabulary_54.txt")
    tokenizer.tokenize_test_data("test.json", "tokenized_54.json")
    tokenizer.tokenize_corpus(corpus_file, "tokenized_corpus.txt")


if __name__ == "__main__":
    # calling task 1

    tokenizer = WordPieceTokenizer()
    preprocess_data("corpus.txt", tokenizer)

    dataset = Word2VecDataset("tokenized_corpus.txt", tokenizer)

    vocab_size = len(tokenizer.vocab)
    embed_size = 200
    model = Word2VecModel(vocab_size, embed_size)
    train(model, dataset, epochs=5, batch_size=16)
    model.load_state_dict(torch.load("word2vec_cbow_model.pth"))
    model.eval()
    find_triplets(model, dataset.word_to_idx, dataset.vocab)
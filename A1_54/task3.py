import torch
from torch.utils.data import Dataset
from task1 import WordPieceTokenizer 
from task2 import Word2VecModel  
import numpy as np
import math
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralLMDataset(Dataset):
    def __init__(
        self,
        corpus_file,
        tokenizer=None,
        window_size=5,
        embed_size=200,
        word2vec_model=None,
    ):

        self.window_size = window_size
        self.embed_size = embed_size
        self.tokenizer = tokenizer if tokenizer else WordPieceTokenizer()

        with open(corpus_file, "r", encoding="utf-8") as file:
            self.corpus = file.read().split()

        if word2vec_model is None:
            raise ValueError("Word2Vec model is required")

        self.word2vec_model = word2vec_model
        self.vocab = self.tokenizer.vocab

        if "<UNK>" not in self.vocab:
            self.vocab.append("<UNK>")

        # Initialize word_to_idx without using enumerate
        self.word_to_idx = {}
        index = 0
        for word in self.vocab:
            self.word_to_idx[word] = index
            index += 1

        self.data = self.create_data()
    def create_data(self):

        data = []
        num_tokens = len(self.corpus)

        for i in range(self.window_size, num_tokens - self.window_size):
            context = (
                self.corpus[i - self.window_size : i]
                + self.corpus[i + 1 : i + 1 + self.window_size]
            )
            target = self.corpus[i]
            data.append((context, target))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]

        context_indices = [
            self.word_to_idx.get(word, self.word_to_idx["<UNK>"]) for word in context
        ]
        target_index = self.word_to_idx.get(target, self.word_to_idx["<UNK>"])

        context_tensor = torch.tensor(context_indices, dtype=torch.long)
        target_tensor = torch.tensor(target_index, dtype=torch.long)

        return context_tensor, target_tensor


class NeuralLM1(nn.Module):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size * 2 * window_size, 512)
        self.dropout = nn.Dropout(0.3)  
        self.fc2 = nn.Linear(512, vocab_size)

    def forward(self, context):
        embedded = self.embeddings(context)
        embedded = embedded.view(embedded.size(0), -1)
        x = F.relu(self.fc1(embedded))
        x = self.dropout(x)  
        output = self.fc2(x)
        return output


class NeuralLM2(nn.Module):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size * 2 * window_size, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, vocab_size)

    def forward(self, context):
        embedded = self.embeddings(context)
        embedded = embedded.view(embedded.size(0), -1)  # Flatten for MLP
        x = F.tanh(self.fc1(embedded)) 

        x = F.relu(self.fc2(x))  
        x = self.dropout(x)
        output = self.fc3(x)
        return output


class NeuralLM3(nn.Module):
    def __init__(self, vocab_size, embed_size, window_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size * 2 * window_size, 1024)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc4 = nn.Linear(256, vocab_size)

    def forward(self, context):
        embedded = self.embeddings(context)
        embedded = embedded.view(embedded.size(0), -1)  # Flatten for MLP
        x = F.relu(self.fc1(embedded))  # ReLU activation
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))  # Sigmoid activation
        x = F.relu(self.fc3(x))  # ReLU activation
        x = self.dropout(x)
        output = self.fc4(x)
        return output


def accuracy(model, dataset):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for context, target in dataset:
            output = model(context)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    return correct / total


def perplexity(model, dataset):
    log_likelihood = 0
    total_words = 0
    model.eval()

    with torch.no_grad():
        for context, target in dataset:
            output = model(context)
            log_likelihood += F.cross_entropy(output, target, reduction="sum").item()
            total_words += len(target)

    return math.exp(log_likelihood / total_words)


def train(model, dataset, epochs=10, batch_size=16, lr=0.00005):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_losses = []
    val_losses = []

    # Split dataset into train and validation sets
    val_size = int(0.15 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    patience = 3  
    counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for context, target in train_loader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for context, target in val_loader:
                output = model(context)
                loss = criterion(output, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        train_accuracy = accuracy(model, train_loader)
        val_accuracy = accuracy(model, val_loader)
        train_perplexity = perplexity(model, train_loader)
        val_perplexity = perplexity(model, val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, "
            f"Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}, Train Perplexity: {train_perplexity:.4f}, "
            f"Val Perplexity: {val_perplexity:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), f"neural_lm_model_{model.__class__.__name__}.pth")


def predict_next_tokens(model, test_file, top_k=3):
    with open(test_file, "r") as f:
        test_sentences = f.readlines()

    model.eval()
    with torch.no_grad():
        for sentence in test_sentences:
            words = sentence.strip().split()
            context = words[-5:]  # Use last 5 words as context (adjustable)
            context_indices = [
                model.word_to_idx.get(word, model.word_to_idx["<UNK>"])
                for word in context
            ]
            context_tensor = torch.tensor(context_indices, dtype=torch.long).unsqueeze(
                0
            )

            output = model(context_tensor)
            top_tokens = torch.topk(output, top_k, dim=1).indices[0].tolist()

            predicted_tokens = [model.vocab[idx] for idx in top_tokens]
            print(f"Context: {' '.join(words)}")
            print(f"Predicted next tokens: {predicted_tokens}\n")


if __name__ == "__main__":

    tokenizer = WordPieceTokenizer()
    tokenizer.preprocess_data("corpus.txt")
    tokenizer.construct_vocabulary("vocabulary_54.txt")
    dataset = NeuralLMDataset(
        "tokenized_corpus.txt", tokenizer, window_size=5, word2vec_model=Word2VecModel
    )

    model1 = NeuralLM1(len(tokenizer.vocab), 200, window_size=5)
    model2 = NeuralLM2(len(tokenizer.vocab), 200, window_size=5)
    model3 = NeuralLM3(len(tokenizer.vocab), 200, window_size=5)

    train(model1, dataset, epochs=5, batch_size=16)
    train(model2, dataset, epochs=5, batch_size=16)
    train(model3, dataset, epochs=5, batch_size=16)

    predict_next_tokens(model1, "test.txt")
    predict_next_tokens(model2, "test.txt")
    predict_next_tokens(model3, "test.txt")
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from conlleval import evaluate  


with open("train.json", "r") as file:
    train_data = json.load(file)
with open("val.json", "r") as file:
    val_data = json.load(file)

train_data
val_data
len(train_data), len(val_data)
print(train_data[0])
print(val_data[0])
print(train_data[0].keys())
print(val_data[0].keys())
print(len(train_data), len(val_data))

def BIO(sentence, aspect_terms):
    tokens = sentence.split()
    labels = []
    for i in range(len(tokens)):
        labels.append("O")

    if not aspect_terms:
        return tokens, labels

    map_char_tokenIndex = {}
    charIndex = 0

    for tokenIndex, token in enumerate(tokens):
        for i in range(len(token)):
            map_char_tokenIndex[charIndex + i] = tokenIndex
        charIndex += len(token) + 1  

    for aspect_term in aspect_terms:
        if isinstance(aspect_term, dict):
            start, end = int(aspect_term.get("from", -1)), int(aspect_term.get("to", -1))
            if start == -1 or end == -1:
                continue

            overlapIndex = sorted(set(map_char_tokenIndex[i] for i in range(start, end) if i in map_char_tokenIndex))
            if overlapIndex:
                labels[overlapIndex[0]] = "B" 
                for i in overlapIndex[1:]:
                    labels[i] = "I"  

    return tokens, labels

tokens, labels = BIO("But the staff was so horrible to us.", [{"term": "staff", "from": 8, "to": 13}])
(tokens, labels)
print(tokens)
print(labels)
print(tokens, labels)

def preprocess(data):
    preprocessed_data = []
    for item in data:
        sentence = item.get("sentence", "")
        aspect_terms = item.get("aspect_terms", [])
        tokens, labels = BIO(sentence, aspect_terms)
        preprocessed_data.append({"sentence": sentence, "tokens": tokens, "labels": labels})
    return preprocessed_data

train1 = preprocess(train_data)
val1 = preprocess(val_data)

with open("train_task_1.json", "w") as file:
    json.dump(train1, file)
with open("val_task_1.json", "w") as file:
    json.dump(val1, file)

with open("train_task_1.json", "r") as file:
  train_data = json.load(file)
with open("val_task_1.json", "r") as file:
  val_data = json.load(file)

train_data
val_data
print(train_data[0])
print(val_data[0])
print(train_data[0].keys())
print(val_data[0].keys())
print(len(train_data), len(val_data))

def word2vec(file):
    embedded_words = {}
    with open(file, "r", encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embedded_words[word] = vector
    return embedded_words

embedded_words = word2vec("glove.6B.100d.txt")
embedded_words
len(embedded_words)

embedded_words_fast = word2vec("cc.en.300.vec")
embedded_words_fast
len(embedded_words_fast)


label_map = {"O": 0, "B": 1, "I": 2}
index_to_label = {0: "O", 1: "B", 2: "I"}  


def prepare_data(data, word_index):
    X, Y = [], []
    tokens_list = []  
    for item in data:
        tokens = item["tokens"]
        labels = item["labels"]

        token_ids = [word_index.get(token.lower(), 0) for token in tokens] 
        label_ids = [label_map[label] for label in labels]  

        X.append(token_ids)
        Y.append(label_ids)
        tokens_list.append(tokens)  

    return X, Y, tokens_list

def create_tf_dataset(X, Y, batch_size=32):
    seq_lengths = [len(x) for x in X]
    
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(X, padding="post")
    Y_padded = tf.keras.preprocessing.sequence.pad_sequences(Y, padding="post")
    
    dataset = tf.data.Dataset.from_tensor_slices((X_padded, Y_padded))
    return dataset.batch(batch_size), seq_lengths

# Add a function to create a TF dataset without padding for the model evaluation
def create_tf_dataset_no_padding(X, Y, batch_size=1):
    # Here we use batch_size=1 to handle each sequence individually
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.ragged.constant(X), tf.ragged.constant(Y))
    )
    return dataset.batch(batch_size)

def flatten_list(nested_list):
    """Helper function to flatten a list of lists."""
    return [item for sublist in nested_list for item in sublist]

def inputLayer(size_vocab, matrix_embedding):
    return tf.keras.layers.Embedding(input_dim=size_vocab, output_dim=300, weights=[matrix_embedding], trainable=False)

def rnnLayer(rnn):
    if rnn == "RNN":
        return tf.keras.layers.SimpleRNN(128, return_sequences=True)
    elif rnn == "GRU":
        return tf.keras.layers.GRU(128, return_sequences=True)

def outputLayer():
    return tf.keras.layers.Dense(3, activation="softmax")


# Define AspectTermModel class for model loading
class AspectTermModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_matrix, rnn_type):
        super(AspectTermModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, 
            output_dim=300, 
            weights=[embedding_matrix], 
            trainable=False
        )
        
        if rnn_type == "RNN":
            self.rnn = tf.keras.layers.SimpleRNN(128, return_sequences=True)
        elif rnn_type == "GRU":
            self.rnn = tf.keras.layers.GRU(128, return_sequences=True)
        
        self.output_layer = tf.keras.layers.Dense(3, activation="softmax")
    
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        x = self.rnn(x)
        return self.output_layer(x)


def build(size_vocab, matrix_embedding, rnn):
    i = tf.keras.Input(shape=(None,), dtype="int32")
    x = inputLayer(size_vocab, matrix_embedding)(i)
    x = rnnLayer(rnn)(x)
    x = outputLayer()(x)
    return tf.keras.Model(inputs=i, outputs=x)

def prepare_for_conlleval(tokens_list, true_tags, pred_tags):
    conll_lines = []
    tag_j = 0
    
    for tokens in tokens_list:
        for token in tokens:
            if tag_j < len(true_tags):
                conll_lines.append(f"{token} {true_tags[tag_j]} {pred_tags[tag_j]}")
                tag_j += 1

        conll_lines.append("")
    
    return conll_lines

def load_model_and_evaluate(test_file, model_path, word_index, index_to_label):
    with open(test_file, "r") as f:
        test_data = json.load(f)
    
    preprocessed_test = preprocess(test_data)

    test_X, test_Y, test_tokens = prepare_data(preprocessed_test, word_index)
    test_ds = create_tf_dataset_no_padding(test_X, test_Y)

    model = tf.keras.models.load_model(model_path, custom_objects={'AspectTermModel': AspectTermModel})
    
    all_gold = []
    all_pred = []
    
    for x_batch, y_batch in test_ds:
        preds = model(x_batch, training=False)  
        gold_seq = y_batch.flat_values.numpy().tolist()  
        pred_seq = tf.argmax(preds[0], axis=-1).numpy().tolist()  
        
        all_gold.extend(gold_seq)
        all_pred.extend(pred_seq)
    
    all_gold_str = [index_to_label[l] for l in all_gold]
    all_pred_str = [index_to_label[l] for l in all_pred]
    
    precision, recall, f1 = evaluate(all_gold_str, all_pred_str)
    
    print(f"Test Evaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    with open(f"test_evaluation_{model_path.split('/')[-1].replace('.h5', '')}.txt", "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
    
    conll_output = prepare_for_conlleval(test_tokens, all_gold_str, all_pred_str)
    with open(f"test_conll_{model_path.split('/')[-1].replace('.h5', '')}.txt", "w") as f:
        f.write("\n".join(conll_output))
    
    return precision, recall, f1

def train_and_evaluate(vocab_size, embedding_matrix, rnn_type, embedding_type, train_ds, val_ds, val_tokens, val_X, val_Y, val_seq_lengths, epochs=10):
    model = build(vocab_size, embedding_matrix, rnn_type)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


    best_f1 = 0
    best_epoch = 0

    histories = {
        "train_loss": [], 
        "train_accuracy": [], 
        "val_loss": [],
        "val_accuracy": [],
        "f1": []
    }

    model_name = f"{embedding_type}_{rnn_type}"

    final_conll_output = None
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        history = model.fit(train_ds, epochs=1, verbose=0)

        train_loss = history.history['loss'][0]
        train_acc = history.history['accuracy'][0]
        histories["train_loss"].append(train_loss)
        histories["train_accuracy"].append(train_acc)

        all_true_tags, all_pred_tags = [], []
        

        val_X_padded = tf.keras.preprocessing.sequence.pad_sequences(val_X, padding="post")
        val_Y_padded = tf.keras.preprocessing.sequence.pad_sequences(val_Y, padding="post")

        predictions = model.predict(val_X_padded, verbose=0)
        pred_labels = np.argmax(predictions, axis=-1)
    
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        histories["val_loss"].append(val_loss)
        histories["val_accuracy"].append(val_acc)

        for i, length in enumerate(val_seq_lengths):

            true_seq = [index_to_label[j] for j in val_Y_padded[i][:length]]
            pred_seq = [index_to_label[j] for j in pred_labels[i][:length]]
  
            all_true_tags.extend(true_seq)
            all_pred_tags.extend(pred_seq)
        
        tag_accuracy = np.mean([t == p for t, p in zip(all_true_tags, all_pred_tags)])
        print(f"Tag-level accuracy: {tag_accuracy:.4f}")
        

        chunk_prec, chunk_rec, chunk_f1 = evaluate(all_true_tags, all_pred_tags, verbose=True)
        print(f"Chunk-level metrics - Precision: {chunk_prec:.2f}, Recall: {chunk_rec:.2f}, F1: {chunk_f1:.2f}")
   
        if epoch == epochs - 1:
            final_conll_output = prepare_for_conlleval(val_tokens, all_true_tags, all_pred_tags)
        
 
        f1 = chunk_f1
        histories["f1"].append(f1)
        
        print(f"Epoch {epoch+1}: Chunk F1-Score = {f1:.4f}, Tag Accuracy = {tag_accuracy:.4f}, Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        if f1 > best_f1:
            model.save(f"best_model_{model_name}.h5")
            best_f1 = f1
            best_epoch = epoch + 1
            print(f"New best model saved with F1-Score: {best_f1:.4f} at epoch {best_epoch}")

    with open(f"final_evaluation_{model_name}.txt", "w") as f:
        f.write("\n".join(final_conll_output))

    plt.figure(figsize=(15, 10))
    

    plt.subplot(2, 2, 1)
    plt.plot(histories["train_loss"], label='Training Loss')
    plt.plot(histories["val_loss"], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(histories["train_accuracy"], label='Training Accuracy')
    plt.plot(histories["val_accuracy"], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(histories["f1"])
    plt.title('Validation F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')

    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0.1, 0.5, 
             f"Best Model Summary:\n"
             f"- Model: {model_name}\n"
             f"- Best F1-Score: {best_f1:.4f}\n"
             f"- Best Epoch: {best_epoch}\n"
             f"- Final Train Loss: {histories['train_loss'][-1]:.4f}\n"
             f"- Final Val Loss: {histories['val_loss'][-1]:.4f}\n"
             f"- Final Train Acc: {histories['train_accuracy'][-1]:.4f}\n"
             f"- Final Val Acc: {histories['val_accuracy'][-1]:.4f}",
             fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"training_history_{model_name}.png")
    plt.close()
    
    print(f"\nTraining Summary for {model_name}:")
    print(f"Best F1-Score: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Final Train/Val Loss: {histories['train_loss'][-1]:.4f}/{histories['val_loss'][-1]:.4f}")
    print(f"Final Train/Val Accuracy: {histories['train_accuracy'][-1]:.4f}/{histories['val_accuracy'][-1]:.4f}")
    
    return best_f1


glove_embeddings = word2vec("glove.6B.300d.txt")
fasttext_embeddings = word2vec("cc.en.300.vec")

vocab = set(word.lower() for word in glove_embeddings.keys()).union(set(word.lower() for word in fasttext_embeddings.keys()))
word_index = {word: i+1 for i, word in enumerate(vocab)}  


train_X, train_Y, train_tokens = prepare_data(train1, word_index)
val_X, val_Y, val_tokens = prepare_data(val1, word_index)


train_ds, train_seq_lengths = create_tf_dataset(train_X, train_Y)
val_ds, val_seq_lengths = create_tf_dataset(val_X, val_Y)

results = {}
for embedding_type, embeddings in [("GloVe", glove_embeddings), ("fastText", fasttext_embeddings)]:
    for rnn_type in ["RNN", "GRU"]:
        print(f"\n{'='*50}")
        print(f"Training {rnn_type} with {embedding_type} Embeddings:")
        print(f"{'='*50}")
        
        vocab_size = len(word_index) + 1
        embedding_matrix = np.zeros((vocab_size, 300)) 

        for word, j in word_index.items():
            if word in embeddings:
                embedding_matrix[j] = embeddings[word]
            elif word.lower() in embeddings:
                embedding_matrix[j] = embeddings[word.lower()]
        
        best_f1 = train_and_evaluate(vocab_size, embedding_matrix, rnn_type, embedding_type, train_ds, val_ds, 
                                  val_tokens, val_X, val_Y, val_seq_lengths, epochs=10)
        results[f"{embedding_type}_{rnn_type}"] = best_f1


print("\n")
print("Results Summary:")
print("=")
for model, f1 in results.items():
    print(f"{model}: F1-Score = {f1:.4f}")

print("\n")
print("Saved Files Summary:")
print("="*50)
for model_name in results.keys():
    print(f"Model: {model_name}")
    print(f"  - Best model saved at: best_model_{model_name}.h5")
    print(f"  - Training history plot: training_history_{model_name}.png")
    print(f"  - Final evaluation: final_evaluation_{model_name}.txt")

print("\n")
print("Evaluating Models on Test Data:")
print("="*50)


test_file = "test.json"
test_results = {}

for model_name in results.keys():
    model_path = f"best_model_{model_name}.h5"
    print(f"\nEvaluating {model_name} on test data...")
    try:
        precision, recall, f1 = load_model_and_evaluate(test_file, model_path, word_index, index_to_label)
        test_results[model_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")

if test_results:
    print("\n")
    print("Test Results Summary:")
    print("="*50)
    for model_name, metrics in test_results.items():
        print(f"{model_name}:")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall: {metrics['recall']:.4f}")
        print(f"  - F1-Score: {metrics['f1']:.4f}")
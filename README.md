# Deception Detection in Diplomatic Communications

## Project Overview

This project explores the detection of deception in diplomatic messages exchanged in the strategy game *Diplomacy*. Messages are labeled based on sender truthfulness and recipient suspicion, creating a challenging dataset with significant class imbalance (only \~5% deceptive messages). The project compares traditional machine learning classifiers with advanced deep learning architectures that fuse textual, contextual, and metadata information.

## Authors

* Garvit Singh – [garvit24034@iiitd.ac.in](mailto:garvit24034@iiitd.ac.in)
* Sayantan Dasgupta – [sayantan24084@iiitd.ac.in](mailto:sayantan24084@iiitd.ac.in)
* Kanak Yadav – [kanak22611@iiitd.ac.in](mailto:kanak22611@iiitd.ac.in)

## Course

CSE556: Natural Language Processing (Spring 2025)

## Dataset

The dataset comprises 189 messages across 11 Diplomacy games. Each message includes metadata such as sender and receiver IDs, game season and year, scores, and message content.

* Training/Validation Games: Game IDs 1 to 10
* Test Game: Game ID 11 (used as an unseen evaluation set)

Key features:

* Text message content
* Game and player metadata (e.g., score, rank, time)
* Temporal context of previous messages

## Problem Statement

Given the high class imbalance and the subtle nature of deception, traditional classifiers often fail to detect deceptive messages. This project aims to build robust models that go beyond accuracy and emphasize macro-F1 and deceptive-class recall.

## Methodology

### Data Preprocessing

* Null handling, padding, and context window construction
* Metadata feature engineering: punctuation density, score differentials, positional features
* Standardization using `StandardScaler`

### Model Architecture

1. **Text Encoding**: RoBERTa for message and context encoding
2. **Context Modeling**: BiLSTM with attention on message history
3. **Metadata Processing**: Feedforward neural network with optional attention
4. **Feature Fusion**: Concatenation of all representations followed by dense layers
5. **Output**: Binary classification logit indicating deception

### Training Strategy

* Focal loss with adjustable alpha and gamma for class imbalance
* Adversarial training using Fast Gradient Method (FGM)
* Optimizer: AdamW with OneCycleLR scheduler
* Early stopping based on macro-F1 score

### Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score for both classes
* Macro-F1 score (for class balance)
* Confusion matrices for error analysis

## Models Compared

* **Classical Models**: Logistic Regression, Random Forest, Naive Bayes, SVM (with TF-IDF and metadata features)
* **DistilBERT**: Enhanced with threshold calibration for optimal F1 score
* **Hierarchical Attention Models**: Combining RoBERTa embeddings, LSTM attention, and metadata fusion

## Key Results

* Classical models showed high nominal accuracy but failed to detect any deceptive messages.
* DistilBERT with calibrated threshold (0.6881) achieved significantly improved recall and F1 score for the deceptive class.
* Hierarchical attention models achieved the highest overall accuracy (up to 90%) but continued to struggle with deceptive-class recall.


## Future Work

* Per-class threshold optimization
* Synthetic data generation for deceptive messages
* Graph-based modeling of player interactions
* Ensemble techniques to improve generalizability and recall

## References

1. Peskov, D. et al. *It Takes Two to Lie: One to Lie and One to Listen*. ACL 2020.
2. Devlin, J. et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
3. Stanford CS224N. *Deception Detection in Online Mafia Game Interactions*.
4. Hochreiter, S. & Schmidhuber, J. *Long Short-Term Memory*. Neural Computation, 1997.
5. *Deception Detection Accuracy*. Wiley Online Library.
6. AAAI. *Towards Deception Detection in a Language-Driven Game*.

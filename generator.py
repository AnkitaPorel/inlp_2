import sys
import re
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

def load_corpus(corpus_path):
    if not os.path.isfile(corpus_path):
        raise ValueError("Invalid corpus path provided.")
    with open(corpus_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_corpus(corpus_text):
    sentences = re.split(r'(?<=[.!?])\s+', corpus_text)
    return sentences

def split_data(sentences, test_size=1000):
    test_data = random.sample(sentences, test_size)
    training_data = [s for s in sentences if s not in test_data]
    return training_data, test_data

class TextDataset(Dataset):
    def __init__(self, sentences, vocab, n_gram=3):
        self.data = []
        for sentence in sentences:
            tokens = re.findall(r'\b\w+\b', sentence.lower())
            for i in range(len(tokens) - n_gram):
                context = tokens[i:i + n_gram - 1]
                target = tokens[i + n_gram - 1]
                self.data.append((context, target))
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_idxs = torch.tensor([self.vocab.get(word, self.vocab['<UNK>']) for word in context], dtype=torch.long)
        target_idx = torch.tensor(self.vocab.get(target, self.vocab['<UNK>']), dtype=torch.long)
        return context_idxs, target_idx

class FFNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, context_size):
        super(FFNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * (context_size - 1), hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / len(train_loader)

def compute_perplexity(model, dataset, file_name):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    with open(file_name, 'w') as f:
        with torch.no_grad():
            for inputs, targets in dataset:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                perplexity = torch.exp(loss).item()
                f.write(f"{inputs.tolist()} - {perplexity}\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 generator.py <lm_type> <corpus_path> <k>")
        sys.exit(0)
    
    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    k = int(sys.argv[3])
    
    corpus_text = load_corpus(corpus_path)
    sentences = preprocess_corpus(corpus_text)
    print(type(sentences))
    print(len(sentences))
    training_data, test_data = split_data(sentences)
    
    word_counts = Counter([word for sentence in training_data for word in re.findall(r'\b\w+\b', sentence.lower())])
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}
    vocab['<UNK>'] = len(vocab)
    
    if(lm_type == '-f'):
        for n_gram in [3, 5]:
            train_dataset = TextDataset(training_data, vocab, n_gram=n_gram)
            test_dataset = TextDataset(test_data, vocab, n_gram=n_gram)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            vocab_size = len(vocab)
            model = FFNNLanguageModel(vocab_size, embed_dim=15, hidden_dim=20, context_size=n_gram)
            
            train_model(model, train_loader, epochs=10)
            
            train_file = f"{os.path.basename(corpus_path).split('.')[0]}_{n_gram}_train.txt"
            test_file = f"{os.path.basename(corpus_path).split('.')[0]}_{n_gram}_test.txt"
            
            compute_perplexity(model, train_loader, train_file)
            compute_perplexity(model, test_loader, test_file)
            
            print(f"Saved Train Perplexity to {train_file}")
            print(f"Saved Test Perplexity to {test_file}") 


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# Hyperparameters
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 10
TOP_K = 5  # Number of words to predict

# Sample dataset
train_data = ["the cat sits on the mat", "a dog runs in the park", "the bird flies over the tree"]
test_data = ["the fish swims in the pond"]

# Tokenization
def tokenize(sentences):
    return [sentence.lower().split() for sentence in sentences]

train_tokens = tokenize(train_data)
test_tokens = tokenize(test_data)

# Vocabulary and word frequencies
all_words = [word for sentence in train_tokens for word in sentence]
word_counts = Counter(all_words)  # Count word occurrences
vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), start=1)}
vocab["<UNK>"] = 0
idx_to_word = {idx: word for word, idx in vocab.items()}  # Reverse mapping

# Compute word weights (Inverse frequency)
word_weights = torch.ones(len(vocab))  # Default all weights to 1
for word, idx in vocab.items():
    word_weights[idx] = 1.0 / word_counts[word] if word in word_counts else 1.0

# Convert words to indices
def word_to_index(word):
    return vocab.get(word, vocab["<UNK>"])

def create_ngrams(tokenized_sentences, n):
    pairs = []
    for sentence in tokenized_sentences:
        if len(sentence) < n:
            continue
        for i in range(len(sentence) - n + 1):
            ngram = sentence[i:i + n]
            context = ngram[:-1]
            target = ngram[-1]
            pairs.append((context, target))
    return pairs

# Create training pairs for n=3
train_trigram_pairs = create_ngrams(train_tokens, 3)
test_trigram_pairs = create_ngrams(test_tokens, 3)

# Dataset class
class WordDataset(Dataset):
    def __init__(self, pairs):
        self.data = [(torch.tensor([word_to_index(w) for w in context]), word_to_index(target)) for context, target in pairs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Create datasets and dataloaders
train_trigram_dataset = WordDataset(train_trigram_pairs)
test_trigram_dataset = WordDataset(test_trigram_pairs)

train_trigram_loader = DataLoader(train_trigram_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_trigram_loader = DataLoader(test_trigram_dataset, batch_size=BATCH_SIZE, shuffle=False)

# FFNN Model
class FeedForwardLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n):
        super(FeedForwardLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * (n - 1), hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x).view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train function
def train_model(model, train_loader, optimizer, criterion, n):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for context, target in train_loader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"n={n}-gram Model - Epoch {epoch+1}, Loss: {total_loss:.4f}")


def predict_top_k(model, context_words, n, k=TOP_K):
    model.eval()
    
    context_idx = torch.tensor([word_to_index(w) for w in context_words]).unsqueeze(0)

    with torch.no_grad():
        output = model(context_idx)
        top_k_probs, top_k_indices = torch.topk(output, k, dim=1)

    predicted_words = [idx_to_word[idx.item()] for idx in top_k_indices[0]]

    return predicted_words

trigram_model = FeedForwardLM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, 3)
optimizer = optim.Adam(trigram_model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss(weight=word_weights)

train_model(trigram_model, train_trigram_loader, optimizer, criterion, 3)

context_example = ["a", "dog"]
predicted_words = predict_top_k(trigram_model, context_example, 3)
print(f"Context: {context_example}")
print(f"Top-{TOP_K} Predicted Words: {predicted_words}")


import os
import sys
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 32
EPOCHS = 10

def tokenize(sentences):
    return [sentence.lower().split() for sentence in sentences]

def word_to_index(word):
    return vocab.get(word, vocab["<UNK>"])

def create_ngrams(tokenized_sentences, n):
    pairs = []
    i = 0
    for sentence in tokenized_sentences:
        if len(sentence) < n:
            i += 1
            continue
        for i in range(len(sentence) - n + 1):
            ngram = sentence[i:i + n]
            context = ngram[:-1]
            target = ngram[-1]
            pairs.append((context, target))

    print(i)
    return pairs

class WordDataset(Dataset):
    def __init__(self, pairs):
        self.data = [(torch.tensor([word_to_index(w) for w in context]), word_to_index(target)) for context, target in pairs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class FeedForwardLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n):
        super(FeedForwardLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * (n - 1), hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x).view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, optimizer, criterion, n):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for context, target in train_loader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"n={n}-gram Model - Epoch {epoch+1}, Loss: {total_loss:.4f}")

def predict_top_k(model, context_words, n, k):
    model.eval()
    
    context_idx = torch.tensor([word_to_index(w) for w in context_words]).unsqueeze(0)

    with torch.no_grad():
        output = model(context_idx)
        top_k_probs, top_k_indices = torch.topk(output, k, dim=1)

    predicted_words = [idx_to_word[idx.item()] for idx in top_k_indices[0]]

    return predicted_words


def compute_sentence_loss(model, sentence_tokens, criterion, n):
    model.eval()
    pairs = create_ngrams([sentence_tokens], n)  # Generate n-gram pairs for a single sentence
    if not pairs:
        return None  # Skip empty sentences or those too short for n-grams

    total_loss = 0.0
    count = 0
    
    for context, target in pairs:
        context_tensor = torch.tensor([word_to_index(w) for w in context]).unsqueeze(0)  # Add batch dimension
        target_tensor = torch.tensor([word_to_index(target)])  # Target should be a single value
        
        with torch.no_grad():
            output = model(context_tensor)
            loss = criterion(output, target_tensor)
            total_loss += loss.item()
            count += 1
    
    return total_loss / count if count > 0 else float('inf')


def evaluate_model(model, test_sentences, criterion, n):
    sentence_losses = []
    for sentence in test_sentences:
        sentence_tokens = tokenize([sentence])[0]  # Tokenize sentence
        loss = compute_sentence_loss(model, sentence_tokens, criterion, n)
        if loss is not None:
            sentence_losses.append((sentence, loss))
    
    return sentence_losses


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 generator.py <lm_type> <corpus_path> <k>")
        sys.exit(0)
    
    lm_type = sys.argv[1]
    corpus_path = sys.argv[2]
    k = int(sys.argv[3])
    corpus_text = ""
    if not os.path.isfile(corpus_path):
        raise ValueError("Invalid corpus path provided.")
    with open(corpus_path, 'r', encoding='utf-8') as file:
        corpus_text = file.read()

    url_pattern = r'(https?://\S+?)([.,!?;])?(?=\s|$)'
    www_url_pattern = r'www\.\S+?(\s|$)'
    hashtag_pattern = r'#\w+'
    mention_pattern = r'@\w+'
    percentage_pattern = r'\b\d+(\.\d+)?%'
    time_pattern = r'\b\d{1,2}:\d{2}(?:\s?[APap][Mm])?'
    age_pattern = r'\b\d{1,3}\s?(?:years?\s?old|y/o)\b'
    time_period_pattern = r'\b\d{1,4}\s?(?:BC|AD|BCE|CE)\b'

    corpus_text = re.sub(r'\bchapter\s+[IVXLCDM]+\b.*(?:\n|$)', '', corpus_text, flags=re.IGNORECASE)

    corpus_text = re.sub(url_pattern, r'<URL>\2', corpus_text)
    corpus_text = re.sub(www_url_pattern, '', corpus_text)
    corpus_text = re.sub(hashtag_pattern, '<HASHTAG>', corpus_text)
    corpus_text = re.sub(mention_pattern, '<MENTION>', corpus_text)
    corpus_text = re.sub(percentage_pattern, '<PERCENTAGE>', corpus_text)
    corpus_text = re.sub(time_pattern, '<TIME>', corpus_text)
    corpus_text = re.sub(age_pattern, '<AGE>', corpus_text)
    corpus_text = re.sub(time_period_pattern, '<TIME_PERIOD>', corpus_text)

    corpus_text = re.sub(r'(<[A-Z_]+>)([.,!?;])', r'\1', corpus_text)
    corpus_text = re.sub(r'_([^_]+)_', r'\1', corpus_text)
    corpus_text = re.sub(r'\n+', ' ', corpus_text)
    corpus_text = re.sub(r'\d+', '', corpus_text)
    corpus_text = re.sub(r'\s+', ' ', corpus_text).strip()

    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s', corpus_text)
    sentences = [sentence.strip() for sentence in sentences]
    sentences = [re.sub(r'[.,?,!,;,*]', '', sentence) for sentence in sentences]
    sentences = [re.sub(r'-+', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r':', ' ', sentence) for sentence in sentences]
    sentences = [re.sub(r'\s+', ' ', sentence).strip() for sentence in sentences]
    
    test_data = random.sample(sentences, 1000)
    train_data = [s for s in sentences if s not in test_data]    

    if (lm_type == '-f'):
        train_tokens = tokenize(train_data)
        test_tokens = tokenize(test_data)

        all_words = [word for sentence in train_tokens for word in sentence]
        word_counts = Counter(all_words)

        vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items(), start=1)}
        vocab["<UNK>"] = 0

        idx_to_word = {idx: word for word, idx in vocab.items()}
        word_weights = torch.ones(len(vocab))
        for word, idx in vocab.items():
            word_weights[idx] = 1.0 / (word_counts[word] ** 0.5) if word in word_counts else 1.0

        train_trigram_pairs = create_ngrams(train_tokens, 3)
        test_trigram_pairs = create_ngrams(test_tokens, 3)

        train_fivegram_pairs = create_ngrams(train_tokens, 5)
        test_fivegram_pairs = create_ngrams(train_tokens, 5)

        train_trigram_dataset = WordDataset(train_trigram_pairs)
        test_trigram_dataset = WordDataset(test_trigram_pairs)

        train_fivegram_dataset = WordDataset(train_fivegram_pairs)
        test_fivegram_dataset = WordDataset(test_fivegram_pairs)

        train_trigram_loader = DataLoader(train_trigram_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_trigram_loader = DataLoader(test_trigram_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # train_fivegram_loader = DataLoader(train_fivegram_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # test_fivegram_loader = DataLoader(test_fivegram_dataset, batch_size=BATCH_SIZE, shuffle=False)

        trigram_model = FeedForwardLM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, 3)
        
        # fivegram_model = FeedForwardLM(len(vocab), EMBEDDING_DIM, HIDDEN_DIM, 5)

        optimizer = optim.Adam(trigram_model.parameters(), lr=0.001)

        # optimizer_five = optim.Adam(fivegram_model.parameters(), lr=0.001)

        criterion = nn.CrossEntropyLoss()

        train_model(trigram_model, train_trigram_loader, optimizer, criterion, 3)

        # train_model(fivegram_model, train_fivegram_loader, optimizer_five, criterion, 5)

        trigram_losses = evaluate_model(trigram_model, test_data, criterion, 3)
        # fivegram_losses = evaluate_model(fivegram_model, test_data, criterion, 5)

        # Print or save results
        for sentence, loss in trigram_losses[:10]:  # Print first 10 losses
            print(f"Sentence: {sentence}\nTrigram Loss: {loss:.4f}\n")

        # for sentence, loss in fivegram_losses[:10]:  # Print first 10 losses
        #     print(f"Sentence: {sentence}\nFivegram Loss: {loss:.4f}\n")
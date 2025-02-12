import os
import sys
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from torch.nn import init

CONTRACTIONS = {
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not"
}

def expand_contractions(text):
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r'\b' + contraction + r'\b', expansion, text, flags=re.IGNORECASE)
    return text

class NGramDataset(Dataset):
    def __init__(self, corpus, vocab, context_size):
        self.data = []
        self.vocab = vocab
        self.context_size = context_size

        for i in range(context_size, len(corpus)):
            context = corpus[i-context_size:i]
            target = corpus[i]
            self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        context_ids = [self.vocab[word] for word in context]
        target_id = self.vocab[target]
        return torch.tensor(context_ids, dtype=torch.long), torch.tensor(target_id, dtype=torch.long)

def build_vocab(corpus):
    vocab = defaultdict(lambda: len(vocab))
    vocab['<unk>'] = 0
    for word in corpus:
        vocab[word]
    return vocab

class FFNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, context_size, dropout_rate=0.3):
        super(FFNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        init.xavier_uniform_(self.embedding.weight)
        
        self.layer_norm1 = nn.LayerNorm(context_size * embedding_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self.fc1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, vocab_size)
        
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x):
        embedded = self.embedding(x).view(x.size(0), -1)
        embedded = self.layer_norm1(embedded)
        
        out = self.fc1(embedded)
        out = self.layer_norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        return out

def train_with_validation(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0
    training_losses = []
    validation_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_idx, (context, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()

            # if batch_idx % 100 == 0:
            #     print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for context, target in val_loader:
                output = model(context)
                loss = criterion(output, target)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        
        # print(f'Epoch {epoch+1}/{epochs}')
        # print(f'Training Loss: {avg_train_loss:.4f}')
        # print(f'Validation Loss: {avg_val_loss:.4f}')
        # print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            torch.save(model.state_dict(), 'best_ffnn_lm.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
    
    return training_losses, validation_losses

def setup_training(train_data_token, vocab, context_size):
    embedding_dim = 64
    hidden_dim = 96
    
    dataset = NGramDataset(train_data_token, vocab, context_size)
    
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = FFNNLanguageModel(len(vocab), embedding_dim, hidden_dim, context_size, dropout_rate=0.3)
    
    word_counts = Counter(train_data_token)
    total_words = sum(word_counts.values())
    weights = [1.0 / (word_counts.get(word, 1) / total_words + 1e-5) for word in vocab.keys()]
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * len(weights)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.05)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True,
        min_lr=1e-6
    )
    
    return model, train_loader, val_loader, criterion, optimizer, scheduler

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

    corpus_text = expand_contractions(corpus_text)

    corpus_text = re.sub(r'[\'"]', '', corpus_text)

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
    
    sentences = [sentence.lower() for sentence in sentences]
    
    test_data = random.sample(sentences, 1000)
    train_data = [s for s in sentences if s not in test_data]

    train_data_string = ""
    for s in train_data:
        train_data_string += s
    
    train_data_token = train_data_string.split()
    if lm_type == '-f':
        embedding_dim = 100
        hidden_dim = 128
        batch_size = 32
        learning_rate = 0.001
        epochs = 10

        vocab = build_vocab(train_data_token)

        word_counts = Counter(train_data_token)
        
        total_words = len(train_data_token)

        weights = [1.0 / (word_counts[word] + 1e-5) for word in vocab.keys()]
        weights = torch.tensor(weights, dtype=torch.float32)

        train_data_token = [word if word_counts[word] >= 2 else '<unk>' for word in train_data_token]

        for context_size in [3, 5]:
            model, train_loader, val_loader, criterion, optimizer, scheduler = setup_training(
                train_data_token, vocab, context_size
            )
            
            training_losses, validation_losses = train_with_validation(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs=10, patience=5
            )
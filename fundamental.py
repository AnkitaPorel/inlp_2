class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        dropout_rate=0.5,
    ):
        super(LSTMLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        # x: (batch_size, context_size)
        embedded = self.embedding(x)  # (batch_size, context_size, embedding_dim)
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        output, hidden = self.lstm(embedded, hidden)  # output: (batch_size, context_size, hidden_dim)
        output = self.dropout(output)
        logits = self.fc(output)  # (batch_size, context_size, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state with zeros
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim),
        )


def setup_training_lstm(
    train_data_token, vocab, context_size, embedding_dim=100, hidden_dim=128
):
    dataset = NGramDataset(train_data_token, vocab, context_size)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMLanguageModel(
        len(vocab), embedding_dim, hidden_dim, num_layers=1, dropout_rate=0.2
    )

    word_counts = Counter(train_data_token)
    total_words = len(train_data_token)
    weights = torch.tensor(
        [1 - (word_counts[word] / total_words) for word in vocab], dtype=torch.float
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.08)

    return model, train_loader, val_loader, criterion, optimizer


def train_with_validation_lstm(
    model, train_loader, val_loader, criterion, optimizer, epochs=10, patience=3
):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=1
    )
    best_val_loss = float("inf")
    patience_counter = 0
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for context, target in train_loader:
            optimizer.zero_grad()
            hidden = model.init_hidden(context.size(0))
            output, hidden = model(context, hidden)
            output = output[:, -1, :]
            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for context, target in val_loader:
                hidden = model.init_hidden(context.size(0))
                output, hidden = model(context, hidden)
                output = output[:, -1, :]
                loss = criterion(output, target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_lstm_lm.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

    return training_losses, validation_losses


def calculate_perplexity_lstm(model, sentence, vocab, context_size):
    model.eval()
    words = sentence.lower().split()
    
    words = [word if word in vocab else "<unk>" for word in words]
    
    if len(words) <= context_size:
        return float('inf')
    
    total_log_prob = 0.0
    num_predictions = 0
    
    with torch.no_grad():
        hidden = model.init_hidden(1)
        for i in range(context_size, len(words)):
            # Get context
            context = words[i - context_size:i]
            target = words[i]
            
            # Convert to tensor
            context_ids = [vocab[word] for word in context]
            context_tensor = torch.tensor([context_ids], dtype=torch.long)
            target_id = vocab[target]
            
            # Get model output
            output, hidden = model(context_tensor, hidden)
            output = output[:, -1, :]  # Take the last output for prediction
            
            # Calculate log probability
            log_probs = F.log_softmax(output, dim=1)
            total_log_prob += log_probs[0][target_id].item()
            num_predictions += 1
    
    # Calculate average negative log likelihood
    avg_neg_log_likelihood = -total_log_prob / num_predictions
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(avg_neg_log_likelihood)).item()
    
    return perplexity


def evaluate_model_perplexity_lstm(model, test_sentences, vocab, context_size, filename):
    total_perplexity = 0
    valid_sentences = 0
    with open(filename, 'w', encoding='utf-8') as f:
        for sentence in test_sentences:
            perplexity = calculate_perplexity_lstm(model, sentence, vocab, context_size)
            f.write(f"{sentence} | {perplexity}\n")
            if perplexity != float('inf') and perplexity < 10000:
                total_perplexity += perplexity
                valid_sentences += 1
    
    return total_perplexity / valid_sentences if valid_sentences > 0 else float('inf')

def predict_top_k_words_lstm(model, sentence, vocab, context_size, k=5):
    model.eval()
    words = sentence.lower().split()
    
    # Replace unknown words with <unk> token
    words = [word if word in vocab else "<unk>" for word in words]
    
    if len(words) < context_size:
        raise ValueError(f"Input sentence must have at least {context_size} words.")
    
    # Get the last `context_size` words as context
    context = words[-context_size:]
    context_ids = [vocab[word] for word in context]
    context_tensor = torch.tensor([context_ids], dtype=torch.long)  # Shape: (1, context_size)
    
    with torch.no_grad():
        hidden = model.init_hidden(1)  # Batch size is 1
        output, hidden = model(context_tensor, hidden)
        output = output[:, -1, :]  # Take the last output for prediction (shape: (1, vocab_size))
        
        # Compute probabilities
        probs = F.softmax(output, dim=1)  # Shape: (1, vocab_size)
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=1)  # Get top k probabilities and indices
        
        # Convert indices to words
        idx_to_word = {idx: word for word, idx in vocab.items()}
        top_k_words = [idx_to_word[idx.item()] for idx in top_k_indices[0]]
        
        return list(zip(top_k_words, top_k_probs[0].tolist()))  # Return list of (word, probability) tuples


class RNNLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        dropout_rate=0.5,
    ):
        super(RNNLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        # x: (batch_size, context_size)
        embedded = self.embedding(x)  # (batch_size, context_size, embedding_dim)
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        output, hidden = self.rnn(embedded, hidden)  # output: (batch_size, context_size, hidden_dim)
        output = self.dropout(output)
        logits = self.fc(output)  # (batch_size, context_size, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)


def setup_training_rnn(
    train_data_token, vocab, context_size, embedding_dim=100, hidden_dim=128
):
    dataset = NGramDataset(train_data_token, vocab, context_size)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = total_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = RNNLanguageModel(
        len(vocab), embedding_dim, hidden_dim, num_layers=1, dropout_rate=0.2
    )

    word_counts = Counter(train_data_token)
    total_words = len(train_data_token)
    weights = torch.tensor(
        [1 - (word_counts[word] / total_words) for word in vocab], dtype=torch.float
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.08)

    return model, train_loader, val_loader, criterion, optimizer


def train_with_validation_rnn(
    model, train_loader, val_loader, criterion, optimizer, epochs=10, patience=3
):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=1
    )
    best_val_loss = float("inf")
    patience_counter = 0
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for context, target in train_loader:
            optimizer.zero_grad()
            hidden = model.init_hidden(context.size(0))
            output, hidden = model(context, hidden)
            output = output[:, -1, :]  # Take the last output for prediction
            loss = criterion(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for context, target in val_loader:
                hidden = model.init_hidden(context.size(0))
                output, hidden = model(context, hidden)
                output = output[:, -1, :]
                loss = criterion(output, target)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_rnn_lm.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

    return training_losses, validation_losses


def calculate_perplexity_rnn(model, sentence, vocab, context_size, filename):
    model.eval()
    words = sentence.lower().split()
    
    # Replace unknown words with <unk> token
    words = [word if word in vocab else "<unk>" for word in words]
    
    if len(words) <= context_size:
        return float('inf')  # Cannot calculate perplexity for sentences shorter than context size
    
    total_log_prob = 0.0
    num_predictions = 0
    
    with torch.no_grad():
        hidden = model.init_hidden(1)  # Batch size is 1
        # Iterate through the sentence
        for i in range(context_size, len(words)):
            # Get context
            context = words[i - context_size:i]
            target = words[i]
            
            # Convert to tensor
            context_ids = [vocab[word] for word in context]
            context_tensor = torch.tensor([context_ids], dtype=torch.long)
            target_id = vocab[target]
            
            # Get model output
            output, hidden = model(context_tensor, hidden)
            output = output[:, -1, :]  # Take the last output for prediction
            
            # Calculate log probability
            log_probs = F.log_softmax(output, dim=1)
            total_log_prob += log_probs[0][target_id].item()
            num_predictions += 1
    
    # Calculate average negative log likelihood
    avg_neg_log_likelihood = -total_log_prob / num_predictions
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(avg_neg_log_likelihood)).item()
    
    return perplexity


def evaluate_model_perplexity_rnn(model, test_sentences, vocab, context_size):
    total_perplexity = 0
    valid_sentences = 0
    with open(filename, 'w', encoding='utf-8') as f:
        for sentence in test_sentences:
            perplexity = calculate_perplexity_rnn(model, sentence, vocab, context_size)
            f.write(f"{sentence} | {perplexity}\n")
            if perplexity != float('inf') and perplexity < 10000:
                total_perplexity += perplexity
                valid_sentences += 1
    
    return total_perplexity / valid_sentences if valid_sentences > 0 else float('inf')

def predict_top_k_words_rnn(model, sentence, vocab, context_size, k=5):
    model.eval()
    words = sentence.lower().split()
    
    # Replace unknown words with <unk> token
    words = [word if word in vocab else "<unk>" for word in words]
    
    if len(words) < context_size:
        raise ValueError(f"Input sentence must have at least {context_size} words.")
    
    # Get the last `context_size` words as context
    context = words[-context_size:]
    context_ids = [vocab[word] for word in context]
    context_tensor = torch.tensor([context_ids], dtype=torch.long)  # Shape: (1, context_size)
    
    with torch.no_grad():
        hidden = model.init_hidden(1)  # Batch size is 1
        output, hidden = model(context_tensor, hidden)
        output = output[:, -1, :]  # Take the last output for prediction (shape: (1, vocab_size))
        
        # Compute probabilities
        probs = F.softmax(output, dim=1)  # Shape: (1, vocab_size)
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=1)  # Get top k probabilities and indices
        
        # Convert indices to words
        idx_to_word = {idx: word for word, idx in vocab.items()}
        top_k_words = [idx_to_word[idx.item()] for idx in top_k_indices[0]]
        
        return list(zip(top_k_words, top_k_probs[0].tolist()))  # Return list of (word, probability) tuples
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
import os
from datetime import datetime
import math

class AdvancedLyricsDataset(Dataset):
    """Gelişmiş şarkı sözü dataset'i (tüm özellikler dahil)"""
    def __init__(self, csv_path, vocab_path, max_length=300, chord_max_length=50):
        self.data = pd.read_csv(csv_path)
        
        # Vocabulary yükle
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = vocab_data['idx_to_word']
        self.vocab_size = len(self.word_to_idx)
        self.max_length = max_length
        
        self.chord_to_idx = vocab_data['chord_to_idx']
        self.idx_to_chord = vocab_data['idx_to_chord']
        self.chord_vocab_size = len(self.chord_to_idx)
        self.chord_max_length = chord_max_length
        
        # Genre encoder
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(vocab_data['genre_classes'])}
        self.num_genres = len(self.genre_to_idx)
        
        print(f"Advanced dataset yüklendi: {len(self.data)} şarkı")
        print(f"Vocabulary boyutu: {self.vocab_size}")
        print(f"Chord vocabulary boyutu: {self.chord_vocab_size}")
        print(f"Genre sayısı: {self.num_genres}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Şarkı sözlerini sequence'e çevir
        lyrics = eval(row['sequence'])
        lyrics = lyrics[:self.max_length]
        
        # Padding
        if len(lyrics) < self.max_length:
            lyrics.extend([self.word_to_idx['<PAD>']] * (self.max_length - len(lyrics)))
        
        # Akorları sequence'e çevir
        chords = eval(row['chord_sequence'])
        chords = chords[:self.chord_max_length]
        
        # Padding
        if len(chords) < self.chord_max_length:
            chords.extend([self.chord_to_idx['<PAD>']] * (self.chord_max_length - len(chords)))
        
        # Akor özellikleri
        chord_features = [
            row['chord_total_chords'],
            row['chord_unique_chords'],
            row['chord_major_count'],
            row['chord_minor_count'],
            row['chord_seventh_count'],
            row['chord_complex_count'],
            row['chord_diminished_count'],
            row['chord_augmented_count'],
            row['chord_suspended_count'],
            row['chord_chord_diversity'],
            row['chord_chord_complexity'],
            row['chord_chord_transitions']
        ]
        
        # Duygusal özellikler
        emotional_features = [
            row['emotional_positive_words'],
            row['emotional_negative_words'],
            row['emotional_love_words'],
            row['emotional_sadness_words'],
            row['emotional_music_words'],
            row['emotional_emotional_intensity']
        ]
        
        # Ritim özellikleri
        rhythm_features = [
            row['rhythm_avg_line_length'],
            row['rhythm_line_count'],
            row['rhythm_rhythm_variation'],
            row['rhythm_repetition_ratio']
        ]
        
        # Genre
        genre = self.genre_to_idx[row['genre']]
        
        return {
            'lyrics': torch.tensor(lyrics, dtype=torch.long),
            'chords': torch.tensor(chords, dtype=torch.long),
            'chord_features': torch.tensor(chord_features, dtype=torch.float),
            'emotional_features': torch.tensor(emotional_features, dtype=torch.float),
            'rhythm_features': torch.tensor(rhythm_features, dtype=torch.float),
            'genre': torch.tensor(genre, dtype=torch.long),
            'title': row['title'],
            'artist': row['artist']
        }

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class CNNBranch(nn.Module):
    """CNN branch for hybrid model"""
    def __init__(self, vocab_size, embedding_dim, num_filters=100, filter_sizes=[3, 4, 5], dropout=0.3):
        super(CNNBranch, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = embedded.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # [batch_size, num_filters, new_seq_len]
            pooled = self.pool(conv_out).squeeze(-1)  # [batch_size, num_filters]
            conv_outputs.append(pooled)
        
        # Concatenate all conv outputs
        cnn_output = torch.cat(conv_outputs, dim=1)  # [batch_size, num_filters * len(filter_sizes)]
        cnn_output = self.dropout(cnn_output)
        
        return cnn_output

class LSTMBranch(nn.Module):
    """LSTM branch for hybrid model"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim=256, n_layers=2, dropout=0.3):
        super(LSTMBranch, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state from both directions
        lstm_output = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_dim * 2]
        lstm_output = self.dropout(lstm_output)
        
        return lstm_output

class TransformerBranch(nn.Module):
    """Transformer branch for hybrid model"""
    def __init__(self, vocab_size, embedding_dim, num_heads=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.3, max_seq_length=300):
        super(TransformerBranch, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.pos_encoding(embedded.transpose(0, 1)).transpose(0, 1)
        embedded = self.dropout(embedded)
        
        transformer_out = self.transformer(embedded)  # [batch_size, seq_len, embedding_dim]
        
        # Global average pooling
        transformer_output = torch.mean(transformer_out, dim=1)  # [batch_size, embedding_dim]
        transformer_output = self.dropout(transformer_output)
        
        return transformer_output

class FeatureFusion(nn.Module):
    """Feature fusion module"""
    def __init__(self, cnn_dim, lstm_dim, transformer_dim, feature_dim, output_dim, dropout=0.3):
        super(FeatureFusion, self).__init__()
        
        self.cnn_proj = nn.Linear(cnn_dim, output_dim)
        self.lstm_proj = nn.Linear(lstm_dim, output_dim)
        self.transformer_proj = nn.Linear(transformer_dim, output_dim)
        self.feature_proj = nn.Linear(feature_dim, output_dim)
        
        self.attention = nn.MultiheadAttention(output_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, cnn_out, lstm_out, transformer_out, features):
        # Project all inputs to same dimension
        cnn_proj = self.cnn_proj(cnn_out)  # [batch_size, output_dim]
        lstm_proj = self.lstm_proj(lstm_out)  # [batch_size, output_dim]
        transformer_proj = self.transformer_proj(transformer_out)  # [batch_size, output_dim]
        feature_proj = self.feature_proj(features)  # [batch_size, output_dim]
        
        # Stack for attention
        stacked = torch.stack([cnn_proj, lstm_proj, transformer_proj, feature_proj], dim=1)  # [batch_size, 4, output_dim]
        
        # Self-attention
        attended, _ = self.attention(stacked, stacked, stacked)  # [batch_size, 4, output_dim]
        
        # Residual connection and layer norm
        attended = self.layer_norm(attended + stacked)
        attended = self.dropout(attended)
        
        # Global average pooling
        fused_output = torch.mean(attended, dim=1)  # [batch_size, output_dim]
        
        return fused_output

class HybridModel(nn.Module):
    """Hibrid model: CNN + LSTM + Transformer + Features"""
    def __init__(self, vocab_size, chord_vocab_size, embedding_dim=256, 
                 cnn_filters=100, cnn_filter_sizes=[3, 4, 5],
                 lstm_hidden=256, lstm_layers=2,
                 transformer_heads=8, transformer_layers=3, transformer_ff=512,
                 chord_features_dim=12, emotional_features_dim=6, rhythm_features_dim=4,
                 num_genres=6, dropout=0.3, max_seq_length=300):
        super(HybridModel, self).__init__()
        
        # Individual branches
        self.cnn_branch = CNNBranch(vocab_size, embedding_dim, cnn_filters, cnn_filter_sizes, dropout)
        self.lstm_branch = LSTMBranch(vocab_size, embedding_dim, lstm_hidden, lstm_layers, dropout)
        self.transformer_branch = TransformerBranch(vocab_size, embedding_dim, transformer_heads, 
                                                   transformer_layers, transformer_ff, dropout, max_seq_length)
        
        # Feature dimensions
        cnn_output_dim = cnn_filters * len(cnn_filter_sizes)
        lstm_output_dim = lstm_hidden * 2  # bidirectional
        transformer_output_dim = embedding_dim
        total_features_dim = chord_features_dim + emotional_features_dim + rhythm_features_dim
        
        # Feature fusion
        fusion_dim = 256
        self.feature_fusion = FeatureFusion(cnn_output_dim, lstm_output_dim, transformer_output_dim, 
                                           total_features_dim, fusion_dim, dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 4, num_genres)
        )
        
    def forward(self, lyrics, chords, chord_features, emotional_features, rhythm_features):
        # Process lyrics through all branches
        cnn_out = self.cnn_branch(lyrics)
        lstm_out = self.lstm_branch(lyrics)
        transformer_out = self.transformer_branch(lyrics)
        
        # Combine all features
        all_features = torch.cat([chord_features, emotional_features, rhythm_features], dim=-1)
        
        # Feature fusion
        fused_features = self.feature_fusion(cnn_out, lstm_out, transformer_out, all_features)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output

def train_hybrid_model(model, train_loader, val_loader, optimizer, criterion, scheduler, 
                      num_epochs, device, model_save_path, history_plot_path):
    """Hibrid model eğitimi"""
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_accuracy = 0.0
    patience_counter = 0
    patience = 10
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            lyrics = batch['lyrics'].to(device)
            chords = batch['chords'].to(device)
            chord_features = batch['chord_features'].to(device)
            emotional_features = batch['emotional_features'].to(device)
            rhythm_features = batch['rhythm_features'].to(device)
            genres = batch['genre'].to(device)
            
            optimizer.zero_grad()
            outputs = model(lyrics, chords, chord_features, emotional_features, rhythm_features)
            loss = criterion(outputs, genres)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += genres.size(0)
            train_correct += (predicted == genres).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                lyrics = batch['lyrics'].to(device)
                chords = batch['chords'].to(device)
                chord_features = batch['chord_features'].to(device)
                emotional_features = batch['emotional_features'].to(device)
                rhythm_features = batch['rhythm_features'].to(device)
                genres = batch['genre'].to(device)
                
                outputs = model(lyrics, chords, chord_features, emotional_features, rhythm_features)
                loss = criterion(outputs, genres)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += genres.size(0)
                val_correct += (predicted == genres).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        scheduler.step(val_accuracy)
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        print(f'Best Val Acc: {best_val_accuracy:.4f}')
        print('-' * 50)
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Hibrid Model Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Hibrid Model Training History - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save training history
    history_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy,
        'num_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    history_json_path = history_plot_path.replace('.png', '.json')
    with open(history_json_path, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)
    
    return best_val_accuracy

def evaluate_hybrid_model(model, test_loader, device, idx_to_genre):
    """Hibrid model değerlendirmesi"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            lyrics = batch['lyrics'].to(device)
            chords = batch['chords'].to(device)
            chord_features = batch['chord_features'].to(device)
            emotional_features = batch['emotional_features'].to(device)
            rhythm_features = batch['rhythm_features'].to(device)
            genres = batch['genre'].to(device)
            
            outputs = model(lyrics, chords, chord_features, emotional_features, rhythm_features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(genres.cpu().numpy())
    
    # Convert indices to genre names
    predicted_genres = [idx_to_genre[pred] for pred in all_predictions]
    true_genres = [idx_to_genre[label] for label in all_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(true_genres, predicted_genres)
    f1_macro = f1_score(true_genres, predicted_genres, average='macro')
    f1_weighted = f1_score(true_genres, predicted_genres, average='weighted')
    
    # Classification report
    report = classification_report(true_genres, predicted_genres, output_dict=True)
    
    return accuracy, report, f1_macro, f1_weighted

def main():
    print("HİBRİD MODEL EĞİTİMİ BAŞLIYOR")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load vocabulary and genre mapping
    with open("data/processed/vocabulary_advanced.json", 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    vocab_size = len(vocab_data['word_to_idx'])
    chord_vocab_size = len(vocab_data['chord_to_idx'])
    
    # Genre mapping
    genre_labels = vocab_data['genre_classes']
    genre_to_idx = {genre: i for i, genre in enumerate(genre_labels)}
    idx_to_genre = {i: genre for genre, i in genre_to_idx.items()}
    num_genres = len(genre_labels)
    
    # Load datasets
    train_dataset = AdvancedLyricsDataset("data/processed/train_advanced.csv", "data/processed/vocabulary_advanced.json")
    val_dataset = AdvancedLyricsDataset("data/processed/validation_advanced.csv", "data/processed/vocabulary_advanced.json")
    test_dataset = AdvancedLyricsDataset("data/processed/test_advanced.csv", "data/processed/vocabulary_advanced.json")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Smaller batch size for hybrid model
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Model parameters
    embedding_dim = 256
    cnn_filters = 128
    cnn_filter_sizes = [3, 4, 5, 6]
    lstm_hidden = 256
    lstm_layers = 2
    transformer_heads = 8
    transformer_layers = 3
    transformer_ff = 512
    dropout = 0.3
    max_seq_length = 300
    
    model = HybridModel(
        vocab_size, chord_vocab_size, embedding_dim,
        cnn_filters, cnn_filter_sizes,
        lstm_hidden, lstm_layers,
        transformer_heads, transformer_layers, transformer_ff,
        12, 6, 4,  # chord, emotional, rhythm feature dimensions
        num_genres, dropout, max_seq_length
    ).to(device)
    
    print(f"Hibrid model parametreleri: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    num_epochs = 40
    model_save_path = "models/hybrid/best_model.pth"
    history_plot_path = "results/hybrid_training_history.png"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_plot_path), exist_ok=True)
    
    print("Hibrid model eğitimi başlıyor...")
    best_accuracy = train_hybrid_model(model, train_loader, val_loader, optimizer, criterion, scheduler, 
                                     num_epochs, device, model_save_path, history_plot_path)
    print(f"Eğitim tamamlandı! En iyi validation accuracy: {best_accuracy:.4f}")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(model_save_path))
    accuracy, report, f1_macro, f1_weighted = evaluate_hybrid_model(model, test_loader, device, idx_to_genre)
    
    print("\nFinal Evaluation:")
    print("========================================")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro Avg F1: {f1_macro:.4f}")
    print(f"Weighted Avg F1: {f1_weighted:.4f}")
    print("\nClassification Report:")
    for genre in genre_labels:
        if genre in report:
            print(f"{genre}: Precision={report[genre]['precision']:.3f}, Recall={report[genre]['recall']:.3f}, F1={report[genre]['f1-score']:.3f}")
    
    print("\n" + "="*70)
    print("HİBRİD MODEL EĞİTİMİ TAMAMLANDI!")

if __name__ == "__main__":
    main()

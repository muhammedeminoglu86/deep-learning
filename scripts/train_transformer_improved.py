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

class ImprovedLyricsDataset(Dataset):
    """Geliştirilmiş şarkı sözü dataset'i (akorlar dahil)"""
    def __init__(self, csv_path, vocab_path, max_length=200, chord_max_length=30):
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
        
        print(f"Improved dataset yüklendi: {len(self.data)} şarkı")
        print(f"Vocabulary boyutu: {self.vocab_size}")
        print(f"Chord vocabulary boyutu: {self.chord_vocab_size}")
        print(f"Genre sayısı: {self.num_genres}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Şarkı sözlerini sequence'e çevir
        lyrics = eval(row['sequence'])  # String olarak kaydedilmiş listeyi parse et
        lyrics = lyrics[:self.max_length]  # Truncate
        
        # Padding
        if len(lyrics) < self.max_length:
            lyrics.extend([self.word_to_idx['<PAD>']] * (self.max_length - len(lyrics)))
        
        # Akorları sequence'e çevir
        chords = eval(row['chord_sequence'])  # String olarak kaydedilmiş listeyi parse et
        chords = chords[:self.chord_max_length]  # Truncate
        
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
            row['chord_chord_diversity'],
            row['chord_chord_complexity']
        ]
        
        # Genre
        genre = self.genre_to_idx[row['genre']]
        
        return {
            'lyrics': torch.tensor(lyrics, dtype=torch.long),
            'chords': torch.tensor(chords, dtype=torch.long),
            'chord_features': torch.tensor(chord_features, dtype=torch.float),
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

class ImprovedTransformerClassifier(nn.Module):
    """Geliştirilmiş Transformer sınıflandırıcı"""
    def __init__(self, vocab_size, embedding_dim, num_heads, num_encoder_layers, 
                 dim_feedforward, chord_vocab_size, chord_features_dim, num_genres, 
                 dropout=0.1, max_seq_length=200):
        super(ImprovedTransformerClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Lyrics embeddings
        self.lyrics_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Chord embeddings
        self.chord_embedding = nn.Embedding(chord_vocab_size, embedding_dim // 2)
        
        # Chord features projection
        self.chord_features_proj = nn.Linear(chord_features_dim, embedding_dim // 2)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Multi-modal fusion
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_genres)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, lyrics, chords, chord_features):
        batch_size = lyrics.size(0)
        
        # Lyrics processing
        lyrics_emb = self.lyrics_embedding(lyrics)  # [batch, seq_len, embedding_dim]
        lyrics_emb = self.pos_encoding(lyrics_emb.transpose(0, 1)).transpose(0, 1)
        lyrics_emb = self.dropout(lyrics_emb)
        
        # Transformer encoding for lyrics
        lyrics_encoded = self.transformer_encoder(lyrics_emb)  # [batch, seq_len, embedding_dim]
        lyrics_pooled = torch.mean(lyrics_encoded, dim=1)  # [batch, embedding_dim]
        
        # Chord processing
        chord_emb = self.chord_embedding(chords)  # [batch, chord_seq_len, embedding_dim//2]
        chord_features_emb = self.chord_features_proj(chord_features)  # [batch, embedding_dim//2]
        
        # Combine chord embeddings and features
        chord_features_expanded = chord_features_emb.unsqueeze(1).expand(-1, chord_emb.size(1), -1)
        chord_combined = torch.cat([chord_emb, chord_features_expanded], dim=-1)  # [batch, chord_seq_len, embedding_dim]
        
        # Chord pooling
        chord_pooled = torch.mean(chord_combined, dim=1)  # [batch, embedding_dim]
        
        # Multi-modal fusion
        lyrics_expanded = lyrics_pooled.unsqueeze(1)  # [batch, 1, embedding_dim]
        chord_expanded = chord_pooled.unsqueeze(1)  # [batch, 1, embedding_dim]
        
        # Attention between lyrics and chords
        fused_features, _ = self.fusion_layer(lyrics_expanded, chord_expanded, chord_expanded)
        fused_features = fused_features.squeeze(1)  # [batch, embedding_dim]
        
        # Combine features
        combined_features = torch.cat([lyrics_pooled, fused_features], dim=-1)  # [batch, embedding_dim*2]
        
        # Classification
        output = self.classifier(combined_features)
        
        return output

def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs, device, model_save_path, history_plot_path):
    """Model eğitimi"""
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    best_val_accuracy = 0.0
    patience_counter = 0
    patience = 7
    
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
            genres = batch['genre'].to(device)
            
            optimizer.zero_grad()
            outputs = model(lyrics, chords, chord_features)
            loss = criterion(outputs, genres)
            loss.backward()
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
                genres = batch['genre'].to(device)
                
                outputs = model(lyrics, chords, chord_features)
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
    plt.title('Training History - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Training History - Accuracy')
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

def evaluate_model(model, test_loader, device, idx_to_genre):
    """Model değerlendirmesi"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            lyrics = batch['lyrics'].to(device)
            chords = batch['chords'].to(device)
            chord_features = batch['chord_features'].to(device)
            genres = batch['genre'].to(device)
            
            outputs = model(lyrics, chords, chord_features)
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
    print("GELİŞTİRİLMİŞ TRANSFORMER MODEL EĞİTİMİ")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load vocabulary and genre mapping
    with open("data/processed/vocabulary_improved.json", 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    vocab_size = len(vocab_data['word_to_idx'])
    chord_vocab_size = len(vocab_data['chord_to_idx'])
    
    # Genre mapping
    genre_labels = vocab_data['genre_classes']
    genre_to_idx = {genre: i for i, genre in enumerate(genre_labels)}
    idx_to_genre = {i: genre for genre, i in genre_to_idx.items()}
    num_genres = len(genre_labels)
    
    # Load datasets
    train_dataset = ImprovedLyricsDataset("data/processed/train_improved.csv", "data/processed/vocabulary_improved.json")
    val_dataset = ImprovedLyricsDataset("data/processed/validation_improved.csv", "data/processed/vocabulary_improved.json")
    test_dataset = ImprovedLyricsDataset("data/processed/test_improved.csv", "data/processed/vocabulary_improved.json")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Model parameters
    embedding_dim = 256
    num_heads = 8
    num_encoder_layers = 4
    dim_feedforward = 1024
    dropout = 0.2
    max_seq_length = 200
    chord_features_dim = 8
    
    model = ImprovedTransformerClassifier(
        vocab_size, embedding_dim, num_heads, num_encoder_layers, dim_feedforward,
        chord_vocab_size, chord_features_dim, num_genres, dropout, max_seq_length
    ).to(device)
    
    print(f"Model parametreleri: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    num_epochs = 30
    model_save_path = "models/transformer_improved/best_model.pth"
    history_plot_path = "results/transformer_improved_training_history.png"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_plot_path), exist_ok=True)
    
    print("Geliştirilmiş Transformer model eğitimi başlıyor...")
    best_accuracy = train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs, device, model_save_path, history_plot_path)
    print(f"Eğitim tamamlandı! En iyi validation accuracy: {best_accuracy:.4f}")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(model_save_path))
    accuracy, report, f1_macro, f1_weighted = evaluate_model(model, test_loader, device, idx_to_genre)
    
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
    print("GELİŞTİRİLMİŞ TRANSFORMER MODEL EĞİTİMİ TAMAMLANDI!")

if __name__ == "__main__":
    main()

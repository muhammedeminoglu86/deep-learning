import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
from datetime import datetime
import math

class EnhancedLyricsDataset(Dataset):
    """Gelişmiş şarkı sözü dataset'i (akorlar dahil)"""
    def __init__(self, csv_path, vocab_path, max_length=100, chord_max_length=20):
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
        
        print(f"Enhanced dataset yüklendi: {len(self.data)} şarkı")
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
            row['chord_chord_diversity']
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

class EnhancedTransformerModel(nn.Module):
    """Gelişmiş Transformer Model (Akorlar Dahil)"""
    def __init__(self, vocab_size, chord_vocab_size, embedding_dim=128, hidden_dim=256, 
                 num_heads=8, num_layers=6, num_genres=6, max_length=100, chord_max_length=20, dropout=0.1):
        super(EnhancedTransformerModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.chord_vocab_size = chord_vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_genres = num_genres
        self.max_length = max_length
        self.chord_max_length = chord_max_length
        
        # Embedding layers
        self.lyrics_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.chord_embedding = nn.Embedding(chord_vocab_size, embedding_dim, padding_idx=0)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_length)
        self.chord_pos_encoding = PositionalEncoding(embedding_dim, chord_max_length)
        
        # Transformer encoder for lyrics
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.lyrics_transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Transformer encoder for chords
        chord_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.chord_transformer = nn.TransformerEncoder(chord_encoder_layer, num_layers)
        
        # Chord features processing
        self.chord_feature_processor = nn.Sequential(
            nn.Linear(7, hidden_dim // 2),  # 7 chord features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
        # Multi-modal fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),  # lyrics + chords + features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_genres)
        )
        
        # Attention weights for interpretability
        self.attention_weights = None
        
    def forward(self, lyrics, chords, chord_features):
        """
        Args:
            lyrics: Input lyrics tensor [batch_size, seq_len]
            chords: Input chords tensor [batch_size, chord_seq_len]
            chord_features: Chord features tensor [batch_size, 7]
        """
        batch_size = lyrics.size(0)
        
        # Lyrics processing
        lyrics_embedded = self.lyrics_embedding(lyrics)  # [batch_size, seq_len, embedding_dim]
        lyrics_embedded = self.pos_encoding(lyrics_embedded.transpose(0, 1)).transpose(0, 1)
        
        # Create padding mask for lyrics
        lyrics_mask = (lyrics == 0)  # Assuming 0 is padding token
        lyrics_output = self.lyrics_transformer(lyrics_embedded, src_key_padding_mask=lyrics_mask)
        
        # Global average pooling for lyrics
        lyrics_pooled = torch.mean(lyrics_output, dim=1)  # [batch_size, embedding_dim]
        
        # Chords processing
        chords_embedded = self.chord_embedding(chords)  # [batch_size, chord_seq_len, embedding_dim]
        chords_embedded = self.chord_pos_encoding(chords_embedded.transpose(0, 1)).transpose(0, 1)
        
        # Create padding mask for chords
        chords_mask = (chords == 0)  # Assuming 0 is padding token
        chords_output = self.chord_transformer(chords_embedded, src_key_padding_mask=chords_mask)
        
        # Global average pooling for chords
        chords_pooled = torch.mean(chords_output, dim=1)  # [batch_size, embedding_dim]
        
        # Chord features processing
        chord_features_processed = self.chord_feature_processor(chord_features)  # [batch_size, embedding_dim]
        
        # Multi-modal fusion
        fused_features = torch.cat([lyrics_pooled, chords_pooled, chord_features_processed], dim=1)
        fused_output = self.fusion_layer(fused_features)
        
        # Classification
        genre_logits = self.classifier(fused_output)
        
        return genre_logits

class EnhancedTransformerTrainer:
    """Gelişmiş Transformer Model Trainer"""
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """Bir epoch eğitim"""
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            lyrics = batch['lyrics'].to(self.device)
            chords = batch['chords'].to(self.device)
            chord_features = batch['chord_features'].to(self.device)
            genres = batch['genre'].to(self.device)
            
            # Forward pass
            genre_logits = self.model(lyrics, chords, chord_features)
            
            # Loss
            loss = self.criterion(genre_logits, genres)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                lyrics = batch['lyrics'].to(self.device)
                chords = batch['chords'].to(self.device)
                chord_features = batch['chord_features'].to(self.device)
                genres = batch['genre'].to(self.device)
                
                # Forward pass
                genre_logits = self.model(lyrics, chords, chord_features)
                
                # Loss
                loss = self.criterion(genre_logits, genres)
                total_loss += loss.item()
                
                # Accuracy
                _, predicted = torch.max(genre_logits.data, 1)
                total += genres.size(0)
                correct += (predicted == genres).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(genres.cpu().numpy())
        
        accuracy = correct / total
        return total_loss / len(self.val_loader), accuracy, all_predictions, all_targets
    
    def train(self, num_epochs=30):
        """Model eğitimi"""
        print(f"Gelişmiş Transformer model eğitimi başlıyor...")
        print(f"Device: {self.device}")
        print(f"Epoch sayısı: {num_epochs}")
        print("="*60)
        
        best_val_acc = 0
        patience_counter = 0
        patience = 7
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, val_acc, predictions, targets = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # History
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model(f"models/transformer/best_model_enhanced.pth")
                print(f"  -> Yeni en iyi model kaydedildi! (Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  -> Early stopping! Patience: {patience_counter}")
                break
            
            print("-" * 40)
        
        print(f"Eğitim tamamlandı! En iyi validation accuracy: {best_val_acc:.4f}")
        
        # Plot training history
        self.plot_training_history()
        
        # Final evaluation
        self.evaluate_final(predictions, targets)
    
    def evaluate_final(self, predictions, targets):
        """Final evaluation with detailed metrics"""
        print("\nFinal Evaluation:")
        print("="*40)
        
        # Load vocabulary for genre names
        with open('data/processed/vocabulary_enhanced.json', 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        genre_names = vocab_data['genre_classes']
        
        # Classification report
        report = classification_report(targets, predictions, target_names=genre_names, output_dict=True, zero_division=0)
        
        print("Classification Report:")
        for genre in genre_names:
            if genre in report:
                precision = report[genre]['precision']
                recall = report[genre]['recall']
                f1 = report[genre]['f1-score']
                support = report[genre]['support']
                print(f"  {genre:10}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Support={support}")
        
        print(f"\nOverall Accuracy: {report['accuracy']:.4f}")
        print(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")
    
    def save_model(self, path):
        """Model kaydet"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }, path)
    
    def plot_training_history(self):
        """Training history plot"""
        os.makedirs("results", exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Enhanced Transformer Model - Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Enhanced Transformer Model - Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/transformer_enhanced_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Training history plot kaydedildi: results/transformer_enhanced_training_history.png")

def main():
    """Ana fonksiyon"""
    print("Gelişmiş Transformer Model Eğitimi Başlıyor...")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset'leri yükle
    train_dataset = EnhancedLyricsDataset('data/processed/train_enhanced.csv', 'data/processed/vocabulary_enhanced.json')
    val_dataset = EnhancedLyricsDataset('data/processed/validation_enhanced.csv', 'data/processed/vocabulary_enhanced.json')
    
    # DataLoader'lar
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Smaller batch size for memory
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Model oluştur
    model = EnhancedTransformerModel(
        vocab_size=train_dataset.vocab_size,
        chord_vocab_size=train_dataset.chord_vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,  # Reduced layers for efficiency
        num_genres=train_dataset.num_genres,
        max_length=100,
        chord_max_length=20,
        dropout=0.1
    )
    
    print(f"Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer oluştur
    trainer = EnhancedTransformerTrainer(model, train_loader, val_loader, device)
    
    # Eğitim başlat
    trainer.train(num_epochs=25)
    
    print("\n" + "="*70)
    print("Gelişmiş Transformer model eğitimi tamamlandı!")

if __name__ == "__main__":
    main()


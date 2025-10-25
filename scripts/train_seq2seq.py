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

class LyricsDataset(Dataset):
    """Şarkı sözü dataset'i"""
    def __init__(self, csv_path, vocab_path, max_length=100):
        self.data = pd.read_csv(csv_path)
        
        # Vocabulary yükle
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = vocab_data['idx_to_word']
        self.vocab_size = len(self.word_to_idx)
        self.max_length = max_length
        
        # Genre encoder
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(vocab_data['genre_classes'])}
        self.num_genres = len(self.genre_to_idx)
        
        print(f"Dataset yüklendi: {len(self.data)} şarkı")
        print(f"Vocabulary boyutu: {self.vocab_size}")
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
        
        # Genre
        genre = self.genre_to_idx[row['genre']]
        
        return {
            'lyrics': torch.tensor(lyrics, dtype=torch.long),
            'genre': torch.tensor(genre, dtype=torch.long),
            'title': row['title'],
            'artist': row['artist']
        }

class Seq2SeqModel(nn.Module):
    """Seq2Seq (Encoder-Decoder) Model"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_genres=6, max_length=100):
        super(Seq2SeqModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_genres = num_genres
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoder (LSTM)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Decoder (LSTM)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Genre classification head
        self.genre_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 because bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_genres)
        )
        
        # Lyrics generation head
        self.lyrics_generator = nn.Linear(hidden_dim, vocab_size)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        
    def forward(self, lyrics, genre=None, mode='classify'):
        """
        Args:
            lyrics: Input lyrics tensor [batch_size, seq_len]
            genre: Target genre for generation mode
            mode: 'classify' or 'generate'
        """
        batch_size = lyrics.size(0)
        
        # Embedding
        embedded = self.embedding(lyrics)  # [batch_size, seq_len, embedding_dim]
        
        if mode == 'classify':
            # Encoder
            encoder_outputs, (hidden, cell) = self.encoder(embedded)
            
            # Use last hidden state for classification
            # Concatenate forward and backward hidden states
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_dim*2]
            
            # Genre classification
            genre_logits = self.genre_classifier(last_hidden)
            
            return genre_logits
            
        elif mode == 'generate':
            # For generation, we'll use a simpler approach
            # Use encoder output and generate lyrics
            encoder_outputs, (hidden, cell) = self.encoder(embedded)
            
            # Use last hidden state
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
            
            # Generate lyrics
            generated_logits = self.lyrics_generator(last_hidden)
            
            return generated_logits

class Seq2SeqTrainer:
    """Seq2Seq Model Trainer"""
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)
        
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
            genres = batch['genre'].to(self.device)
            
            # Forward pass
            genre_logits = self.model(lyrics, mode='classify')
            
            # Loss
            loss = self.criterion(genre_logits, genres)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                lyrics = batch['lyrics'].to(self.device)
                genres = batch['genre'].to(self.device)
                
                # Forward pass
                genre_logits = self.model(lyrics, mode='classify')
                
                # Loss
                loss = self.criterion(genre_logits, genres)
                total_loss += loss.item()
                
                # Accuracy
                _, predicted = torch.max(genre_logits.data, 1)
                total += genres.size(0)
                correct += (predicted == genres).sum().item()
        
        accuracy = correct / total
        return total_loss / len(self.val_loader), accuracy
    
    def train(self, num_epochs=10):
        """Model eğitimi"""
        print(f"Seq2Seq model eğitimi başlıyor...")
        print(f"Device: {self.device}")
        print(f"Epoch sayısı: {num_epochs}")
        print("="*50)
        
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
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
                self.save_model(f"models/seq2seq/best_model.pth")
                print(f"  -> Yeni en iyi model kaydedildi! (Acc: {val_acc:.4f})")
            
            print("-" * 30)
        
        print(f"Eğitim tamamlandı! En iyi validation accuracy: {best_val_acc:.4f}")
        
        # Plot training history
        self.plot_training_history()
    
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
        ax1.set_title('Seq2Seq Model - Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Seq2Seq Model - Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/seq2seq_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Training history plot kaydedildi: results/seq2seq_training_history.png")

def main():
    """Ana fonksiyon"""
    print("Seq2Seq Model Eğitimi Başlıyor...")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset'leri yükle
    train_dataset = LyricsDataset('data/processed/train.csv', 'data/processed/vocabulary.json')
    val_dataset = LyricsDataset('data/processed/validation.csv', 'data/processed/vocabulary.json')
    
    # DataLoader'lar
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model oluştur
    model = Seq2SeqModel(
        vocab_size=train_dataset.vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_genres=train_dataset.num_genres,
        max_length=100
    )
    
    print(f"Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer oluştur
    trainer = Seq2SeqTrainer(model, train_loader, val_loader, device)
    
    # Eğitim başlat
    trainer.train(num_epochs=15)
    
    print("\n" + "="*60)
    print("Seq2Seq model eğitimi tamamlandı!")

if __name__ == "__main__":
    main()


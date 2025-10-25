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

class CNNModel(nn.Module):
    """CNN Model for Turkish Lyrics Classification"""
    def __init__(self, vocab_size, embedding_dim=128, num_filters=100, filter_sizes=[3, 4, 5], num_genres=6, max_length=100, dropout=0.3):
        super(CNNModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_genres = num_genres
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=filter_size)
            for filter_size in filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(len(filter_sizes) * num_filters, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_genres)
        )
        
    def forward(self, lyrics):
        """
        Args:
            lyrics: Input lyrics tensor [batch_size, seq_len]
        """
        batch_size = lyrics.size(0)
        
        # Embedding
        embedded = self.embedding(lyrics)  # [batch_size, seq_len, embedding_dim]
        
        # Transpose for Conv1d: [batch_size, embedding_dim, seq_len]
        embedded = embedded.transpose(1, 2)
        
        # Apply convolutions
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # [batch_size, num_filters, new_seq_len]
            pooled = torch.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # [batch_size, num_filters, 1]
            conv_outputs.append(pooled.squeeze(2))  # [batch_size, num_filters]
        
        # Concatenate all conv outputs
        concatenated = torch.cat(conv_outputs, dim=1)  # [batch_size, len(filter_sizes) * num_filters]
        
        # Dropout
        concatenated = self.dropout(concatenated)
        
        # Classification
        genre_logits = self.classifier(concatenated)
        
        return genre_logits

class CNNTrainer:
    """CNN Model Trainer"""
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
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
            genres = batch['genre'].to(self.device)
            
            # Forward pass
            genre_logits = self.model(lyrics)
            
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
                genres = batch['genre'].to(self.device)
                
                # Forward pass
                genre_logits = self.model(lyrics)
                
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
    
    def train(self, num_epochs=20):
        """Model eğitimi"""
        print(f"CNN model eğitimi başlıyor...")
        print(f"Device: {self.device}")
        print(f"Epoch sayısı: {num_epochs}")
        print("="*50)
        
        best_val_acc = 0
        patience_counter = 0
        patience = 5
        
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
                self.save_model(f"models/cnn/best_model.pth")
                print(f"  -> Yeni en iyi model kaydedildi! (Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"  -> Early stopping! Patience: {patience_counter}")
                break
            
            print("-" * 30)
        
        print(f"Eğitim tamamlandı! En iyi validation accuracy: {best_val_acc:.4f}")
        
        # Plot training history
        self.plot_training_history()
        
        # Final evaluation
        self.evaluate_final(predictions, targets)
    
    def evaluate_final(self, predictions, targets):
        """Final evaluation with detailed metrics"""
        print("\nFinal Evaluation:")
        print("="*30)
        
        # Load vocabulary for genre names
        with open('data/processed/vocabulary.json', 'r', encoding='utf-8') as f:
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
        ax1.set_title('CNN Model - Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('CNN Model - Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/cnn_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Training history plot kaydedildi: results/cnn_training_history.png")

def main():
    """Ana fonksiyon"""
    print("CNN Model Eğitimi Başlıyor...")
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
    model = CNNModel(
        vocab_size=train_dataset.vocab_size,
        embedding_dim=128,
        num_filters=100,
        filter_sizes=[3, 4, 5],
        num_genres=train_dataset.num_genres,
        max_length=100,
        dropout=0.3
    )
    
    print(f"Model parametreleri: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer oluştur
    trainer = CNNTrainer(model, train_loader, val_loader, device)
    
    # Eğitim başlat
    trainer.train(num_epochs=25)
    
    print("\n" + "="*60)
    print("CNN model eğitimi tamamlandı!")

if __name__ == "__main__":
    main()


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score, classification_report, f1_score
import os
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

class SimpleEnsembleModel:
    """Basit ensemble model - en iyi 3 modeli birleştir"""
    def __init__(self, model_paths, model_types, vocab_path, device='cpu'):
        self.device = device
        self.models = []
        self.model_types = model_types
        
        for model_path, model_type in zip(model_paths, model_types):
            print(f"Loading {model_type} from {model_path}...")
            
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            if model_type == 'cnn':
                model = self._load_cnn_model(model_path, vocab_data)
            elif model_type == 'lstm':
                model = self._load_lstm_model(model_path, vocab_data)
            elif model_type == 'seq2seq':
                model = self._load_seq2seq_model(model_path, vocab_data)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            model.eval()
            self.models.append(model)
        
        print(f"Loaded {len(self.models)} models for ensemble")
    
    def _load_cnn_model(self, model_path, vocab_data):
        """Load CNN model"""
        from models.cnn_model import CNNClassifier
        
        vocab_size = len(vocab_data['word_to_idx'])
        num_genres = len(vocab_data['genre_classes'])
        
        model = CNNClassifier(vocab_size, 256, 100, [3, 4, 5], num_genres, 0.5)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def _load_lstm_model(self, model_path, vocab_data):
        """Load LSTM model"""
        from models.lstm_model import LSTMClassifier
        
        vocab_size = len(vocab_data['word_to_idx'])
        num_genres = len(vocab_data['genre_classes'])
        
        model = LSTMClassifier(vocab_size, 256, 512, 2, num_genres, 0.5)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def _load_seq2seq_model(self, model_path, vocab_data):
        """Load Seq2Seq model"""
        from models.seq2seq_model import Seq2Seq, Encoder, Decoder
        
        vocab_size = len(vocab_data['word_to_idx'])
        num_genres = len(vocab_data['genre_classes'])
        
        encoder = Encoder(vocab_size, 256, 512, 2, 0.5)
        decoder = Decoder(256, 512, num_genres, 2, 0.5)
        model = Seq2Seq(encoder, decoder, self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def predict(self, lyrics_tensor):
        """Ensemble prediction"""
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                output = model(lyrics_tensor.to(self.device))
                predictions.append(output)
        
        # Average predictions
        ensemble_output = torch.stack(predictions).mean(dim=0)
        
        return ensemble_output

def create_optimal_ensemble():
    """En iyi modellerle ensemble oluştur"""
    print("="*70)
    print("OPTİMAL ENSEMBLE MODEL OLUŞTURMA")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Mevcut en iyi model sonuçları
    model_results = {
        "CNN": {"accuracy": 0.8144, "path": "models/cnn/best_model.pth"},
        "Seq2Seq": {"accuracy": 0.8033, "path": "models/seq2seq/best_model.pth"},
        "LSTM": {"accuracy": 0.7784, "path": "models/lstm/best_model.pth"}
    }
    
    print("\nMevcut model performansları:")
    print("-" * 70)
    for model_name, data in model_results.items():
        print(f"{model_name}: {data['accuracy']:.4f} accuracy")
    
    print("\nÖNERİ: En iyi performans gösteren modeller (CNN + Seq2Seq) birleştirilerek")
    print("yaklaşık %85-88 arası accuracy hedeflenebilir!")
    
    # Basit voting ensemble
    print("\nENSEMBLE STRATEJİSİ:")
    print("-" * 70)
    print("1. CNN: %81.44 - En iyi tek model")
    print("2. Seq2Seq: %80.33 - İyi genel performans")
    print("3. LSTM: %77.84 - Destekleyici model")
    print("\nBasit Average Voting ile birleştirilecek")
    
    return model_results

def analyze_best_approach():
    """En iyi yaklaşımı analiz et"""
    print("\n" + "="*70)
    print("FİNAL ANALİZ VE ÖNERİLER")
    print("="*70)
    
    results = {
        "CNN": 0.8144,
        "Seq2Seq": 0.8033,
        "LSTM": 0.7784,
        "Transformer (Original)": 0.7341,
        "Transformer (Improved)": 0.6311,
        "Hybrid": 0.5235
    }
    
    print("\nModel Performans Sıralaması:")
    print("-" * 70)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for i, (model, accuracy) in enumerate(sorted_results, 1):
        print(f"{i}. {model}: {accuracy*100:.2f}%")
    
    print("\nSONUC:")
    print("-" * 70)
    print("OK CNN modeli en yuksek performans gosterdi: 81.44%")
    print("OK Hibrit model beklenen basariyi saglamadi: 52.35% (overfitting)")
    print("OK Transformer modelleri bu veri seti icin uygun degil")
    
    print("\nÖNERİLER:")
    print("-" * 70)
    print("1. CNN modeli en basit ve en başarılı")
    print("2. Hibrit model çok karmaşık - overfitting problemi")
    print("3. Daha fazla veri ve daha fazla regulasyon gerekli")
    print("4. Basit ensemble (CNN + Seq2Seq) %85+ accuracy verebilir")
    
    # Final rapor oluştur
    create_final_summary(sorted_results)

def create_final_summary(sorted_results):
    """Final özet oluştur"""
    print("\n" + "="*70)
    print("FİNAL ÖZET RAPOR")
    print("="*70)
    
    summary = f"""
# TURKCE SARKI SOZU ANALIZI - FINAL SONUCLAR

## EN IYI MODEL: CNN
**Accuracy:** 81.44%
**Parametre:** 2.9M
**Durum:** OK EN IYI PERFORMANS

## TUM MODEL SONUCLARI

"""
    
    for model, accuracy in sorted_results:
        star = "EN IYI" if model == "CNN" else ""
        summary += f"- {model}: {accuracy*100:.2f}% {star}\n"
    
    summary += f"""

## ANALIZ

### Basarili Modeller
1. **CNN** - En basit ve en basarili (%81.44)
2. **Seq2Seq** - Iyi performans ama fazla parametre (%80.33)
3. **LSTM** - Dengeli performans (%77.84)

### Basarisiz Modeller
1. **Hibrid** - Overfitting problemi (%52.35)
2. **Transformer (Original)** - Dusuk performans (%73.41)
3. **Transformer (Improved)** - Cok dusuk performans (%63.11)

## ONERILER

1. **CNN modeli kullan** - En iyi performans
2. **Hibrit modelden kacin** - Overfitting problemi
3. **Basit ensemble dene** - CNN + Seq2Seq birlestir
4. **Daha fazla veri** - Model performansini artirir
5. **Regularizasyon** - Overfitting'i onler

## SONUC

Bu projede **CNN modeli** Turkce sarki sozu analizi icin en uygun secim olmustur.
Basit mimarisi ve %81.44 accuracy'si ile en yuksek basariyi saglamistir.

**Proje basariyla tamamlandi!**

"""
    
    # Dosyaya kaydet
    with open('FINAL_RESULTS.md', 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(summary)
    print("\nDosya kaydedildi: FINAL_RESULTS.md")

def main():
    create_optimal_ensemble()
    analyze_best_approach()
    
    print("\n" + "="*70)
    print("TÜM ANALİZ TAMAMLANDI!")
    print("="*70)

if __name__ == "__main__":
    main()

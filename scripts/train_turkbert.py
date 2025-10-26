import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Check if transformers and datasets are installed
try:
    import transformers
    import datasets
    print(f"Transformers version: {transformers.__version__}")
    print(f"Datasets version: {datasets.__version__}")
except ImportError:
    print("Installing transformers and datasets...")
    os.system("pip install transformers datasets accelerate scikit-learn")
    import transformers
    import datasets

class TurkishLyricsDataset:
    """Türkçe şarkı sözü dataset'i TurkBERT için"""
    def __init__(self, df, tokenizer, max_length=512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Genre mapping
        self.genre_labels = sorted(df['genre'].unique())
        self.genre_to_idx = {genre: i for i, genre in enumerate(self.genre_labels)}
        self.idx_to_genre = {i: genre for genre, i in self.genre_to_idx.items()}
        
        print(f"TurkBERT dataset yüklendi: {len(df)} şarkı")
        print(f"Genre sayısı: {len(self.genre_labels)}")
        print(f"Genre'ler: {self.genre_labels}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Lyrics text
        lyrics = str(row['lyrics'])
        
        # Tokenize
        encoding = self.tokenizer(
            lyrics,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get genre label
        genre = self.genre_to_idx[row['genre']]
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(genre, dtype=torch.long),
            'text': lyrics,
            'title': row['title'],
            'artist': row['artist']
        }

def compute_metrics(eval_pred):
    """Metrikleri hesapla"""
    predictions, labels = eval_pred
    
    # Logits'ten sınıfları al
    predictions = np.argmax(predictions, axis=-1)
    
    # Metrikleri hesapla
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f1_macro': f1_macro
    }

def main():
    print("="*70)
    print("TURKBERT MODEL EĞİTİMİ")
    print("="*70)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    print("\n1. Veri yükleniyor...")
    train_df = pd.read_csv('data/processed/train_enhanced.csv', encoding='utf-8')
    val_df = pd.read_csv('data/processed/validation_enhanced.csv', encoding='utf-8')
    test_df = pd.read_csv('data/processed/test_enhanced.csv', encoding='utf-8')
    
    print(f"Train: {len(train_df)} şarkı")
    print(f"Validation: {len(val_df)} şarkı")
    print(f"Test: {len(test_df)} şarkı")
    
    # Load TurkBERT tokenizer and model
    print("\n2. TurkBERT modeli yükleniyor...")
    model_name = "dbmdz/bert-base-turkish-cased"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Genre sayısını belirle
    genre_labels = sorted(train_df['genre'].unique())
    num_labels = len(genre_labels)
    
    print(f"Genre sayısı: {num_labels}")
    print(f"Genre'ler: {genre_labels}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    # Check model size
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parametreleri: {num_params:,}")
    print(f"Model adı: {model_name}")
    
    # Create datasets
    print("\n3. Dataset oluşturuluyor...")
    train_dataset = TurkishLyricsDataset(train_df, tokenizer)
    val_dataset = TurkishLyricsDataset(val_df, tokenizer)
    test_dataset = TurkishLyricsDataset(test_df, tokenizer)
    
    # Training arguments
    print("\n4. Training başlatılıyor...")
    training_args = TrainingArguments(
        output_dir='./turkbert_results',
        num_train_epochs=5,  # TurkBERT için daha az epoch (pre-trained olduğu için)
        per_device_train_batch_size=8,  # GPU memory için küçük batch
        per_device_eval_batch_size=8,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,  # CPU için False
        report_to="none"
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("\n5. Eğitim başlıyor...")
    print("-" * 70)
    trainer.train()
    print("-" * 70)
    print("\nEğitim tamamlandı!")
    
    # Save model
    print("\n6. Model kaydediliyor...")
    model_save_path = "models/turkbert/best_model"
    os.makedirs(model_save_path, exist_ok=True)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Save genre labels
    with open(os.path.join(model_save_path, "genre_labels.json"), 'w', encoding='utf-8') as f:
        json.dump(genre_labels, f, ensure_ascii=False, indent=2)
    
    print(f"Model kaydedildi: {model_save_path}")
    
    # Evaluate on test set
    print("\n7. Test seti üzerinde değerlendirme...")
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    
    print("\n" + "="*70)
    print("FINAL TEST SONUÇLARI")
    print("="*70)
    print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test Precision: {test_results['eval_precision']:.4f}")
    print(f"Test Recall: {test_results['eval_recall']:.4f}")
    print(f"Test F1: {test_results['eval_f1']:.4f}")
    print(f"Test F1 (Macro): {test_results['eval_f1_macro']:.4f}")
    
    # Detailed classification report
    print("\n8. Detaylı sınıflandırma raporu...")
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    
    # Classification report
    report = classification_report(
        true_labels, 
        pred_labels, 
        target_names=genre_labels,
        output_dict=True
    )
    
    print("\nGenre bazlı performans:")
    print("-" * 70)
    for genre in genre_labels:
        if genre in report:
            metrics = report[genre]
            print(f"{genre:10s} - Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1-score']:.3f}")
    
    print("\nGenel Metrikler:")
    print(f"Macro Avg - Precision: {report['macro avg']['precision']:.3f}, Recall: {report['macro avg']['recall']:.3f}, F1: {report['macro avg']['f1-score']:.3f}")
    print(f"Weighted Avg - Precision: {report['weighted avg']['precision']:.3f}, Recall: {report['weighted avg']['recall']:.3f}, F1: {report['weighted avg']['f1-score']:.3f}")
    
    # Save results
    results = {
        'model_name': model_name,
        'num_parameters': num_params,
        'accuracy': test_results['eval_accuracy'],
        'precision': test_results['eval_precision'],
        'recall': test_results['eval_recall'],
        'f1': test_results['eval_f1'],
        'f1_macro': test_results['eval_f1_macro'],
        'classification_report': report,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('results/turkbert_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nSonuçlar kaydedildi: results/turkbert_results.json")
    
    print("\n" + "="*70)
    print("TURKBERT EĞİTİMİ TAMAMLANDI!")
    print("="*70)
    
    # Comparison with previous models
    print("\n" + "="*70)
    print("MODEL KARŞILAŞTIRMASI")
    print("="*70)
    print(f"CNN:            %81.44")
    print(f"Seq2Seq:        %80.33")
    print(f"LSTM:           %77.84")
    print(f"Transformer:    %73.41")
    print(f"TurkBERT:       %{test_results['eval_accuracy']*100:.2f} ⭐")
    
    improvement = (test_results['eval_accuracy'] - 0.8144) * 100
    if improvement > 0:
        print(f"\nTurKBERt, CNN'den {improvement:+.2f} puan daha iyi! 🎉")
    else:
        print(f"\nTurkBERT, CNN'den {improvement:.2f} puan daha düşük...")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

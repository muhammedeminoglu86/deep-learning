# TRANSFORMER'I Daha İYİ Hale GETİRME REHBERİ
## LLaMA, BERT, GPT Kullanımı ve Optimizasyon Stratejileri

### 🎯 HEDEF: Transformer accuracy'sini %73'ten %85+ yapmak

---

## 📊 İKİ ANA YAKLAŞIM

### 1. MEVCUT TRANSFORMER'I İYİLEŞTİRME
- Pre-trained model kullanmadan
- Sadece veri ve hiperparametre optimizasyonu
- Daha iyi preprocessing
- Daha uzun sequence

### 2. PRE-TRAINED MODELLER KULLANMA
- LLaMA (LLaMA-3.1-8B)
- TurkBERT
- mBERT (Multilingual BERT)
- Türkçe GPT modelleri

---

## 🚀 YAKLAŞIM 1: MEVCUT TRANSFORMER'I İYİLEŞTİRME

### 1. Veri Seti Genişletme

**Şu An:**
- 3,605 şarkı
- Train: 2,523
- Test: 721

**Hedef:**
```python
# Veri kaynakları
- Genius.com API
- Spotify API (lyrics + audio features)
- Musixmatch API
- Turkish music databases
- Web scraping (1000+ şarkı daha)

# Hedef boyut
Hedef: 10,000+ şarkı
Bu: Transformer için ideal
```

### 2. Sequence Length Artırma

**Şu An:**
- Max length: 300 token
- Ortalama şarkı: 149.7 kelime

**İyileştirme:**
```python
# Yeni ayarlar
max_length = 512  # Transformer için ideal
# Memory-efficient attention kullan
# Flash Attention veya Sparse Attention

# Kod örneği
class ImprovedTransformer(nn.Module):
    def __init__(self, max_len=512):
        # Flash Attention kullan
        self.attn = FlashAttention(8, 512)
        # veya
        self.attn = SparseAttention(8, 512, sparsity=0.1)
```

### 3. Preprocessing İyileştirmeleri

```python
# Character-level + Word-level hybrid
def advanced_tokenization(text):
    # 1. Character-level features
    char_emb = char_embedding(text)
    
    # 2. Subword tokenization (BPE/SentencePiece)
    subword_emb = subword_embedding(text)
    
    # 3. Morphological features (Türkçe için kritik!)
    morpho_emb = morphological_analysis(text)
    
    # Combine
    return char_emb + subword_emb + morpho_emb

# SentencePiece ile Türkçe özel tokenization
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='turkish_lyrics.txt',
    model_prefix='turkish_vocab',
    vocab_size=32000,  # Vocabulary size
    character_coverage=0.9995,
    model_type='bpe'  # Byte Pair Encoding
)
```

### 4. Hiperparametre Optimizasyonu

```python
# Grid Search veya Bayesian Optimization
from sklearn.model_selection import GridSearchCV

param_grid = {
    'num_heads': [4, 6, 8, 12],
    'num_layers': [2, 3, 4, 6],
    'dim_feedforward': [256, 512, 1024, 2048],
    'dropout': [0.1, 0.3, 0.5],
    'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
    'warmup_steps': [100, 500, 1000],
    'weight_decay': [0.0, 0.01, 0.1]
}

# Optuna ile Bayesian Optimization
import optuna

def objective(trial):
    num_heads = trial.suggest_int('num_heads', 4, 12)
    num_layers = trial.suggest_int('num_layers', 2, 6)
    dim_ff = trial.suggest_int('dim_feedforward', 256, 2048)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('learning_rate', 1e-5, 5e-4)
    
    model = TransformerClassifier(num_heads, num_layers, dim_ff, dropout)
    # Train and return validation accuracy
    return train_and_evaluate(model, lr)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### 5. Advanced Training Techniques

```python
# 1. Learning Rate Scheduling
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=500,
    num_training_steps=10000
)

# 2. Gradient Accumulation
for i, batch in enumerate(train_loader):
    outputs = model(batch)
    loss = criterion(outputs, batch['labels']) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. Mixed Precision Training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 4. Label Smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# 5. Focal Loss (imbalanced classes için)
import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

---

## 🌟 YAKLAŞIM 2: PRE-TRAINED MODELLER KULLANMA

### LLaMA 3.1 Kullanımı

**Avantajlar:**
- Çok güçlü pre-trained model
- Türkçe desteği var (ama yetersiz)
- 8B parametre
- Fine-tuning çok etkili

**Nasıl Kullanılır:**

```python
from transformers import LlamaForSequenceClassification, LlamaTokenizer, Trainer, TrainingArguments

# 1. Model ve tokenizer yükle
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6,  # 6 genre
    problem_type="single_label_classification"
)

# 2. Tokenization
def tokenize_function(examples):
    return tokenizer(
        examples['lyrics'],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

# 3. Dataset hazırla
from datasets import Dataset
train_dataset = Dataset.from_pandas(train_df)
train_dataset = train_dataset.map(tokenize_function, batched=True)

# 4. Training
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=1,  # GPU memory için küçük
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=5e-5,
    fp16=True,  # Mixed precision
    gradient_accumulation_steps=8,  # Effective batch size = 8
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# 5. Fine-tuned modeli kaydet
model.save_pretrained('./llama_finetuned')
tokenizer.save_pretrained('./llama_finetuned')
```

**Sorunlar:**
- GPU memory yüksek (32GB+ gerekli)
- Slow inference
- Türkçe desteği yetersiz

### mBERT (Multilingual BERT)

**Avantajlar:**
- Türkçe çok iyi destekler
- Hafif model (110M parametre)
- Hızlı inference

**Nasıl Kullanılır:**

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Model yükle
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6
)

# 2. Tokenization
def tokenize_lyrics(lyrics):
    return tokenizer(
        lyrics,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors='pt'
    )

# 3. Training
# Hugging Face Trainer kullan
training_args = TrainingArguments(
    output_dir='./bert_results',
    num_train_epochs=3,
    per_device_train_batch_size=16,  # Daha büyük batch
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Expected accuracy: %85-90+ 🎉
```

### TurkBERT (Türkçe için özel)

```python
# TurkBERT kullan
model_name = "dbmdz/bert-base-turkish-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6
)

# Türkçe için özel optimized!
# Expected accuracy: %88-92+ 🎉
```

---

## 🎯 KARŞILAŞTIRMA TABLOSU

| Model | Parameters | GPU RAM | Accuracy (Expected) | Inference Speed |
|-------|------------|---------|---------------------|-----------------|
| **Current Transformer** | 9.8M | 2GB | 63% | Fast |
| **Improved Transformer** | 9.8M | 2GB | 80% | Fast |
| **mBERT** | 110M | 4GB | 85% | Fast |
| **TurkBERT** | 110M | 4GB | 90% | Fast |
| **LLaMA 3.1** | 8B | 32GB | 88% | Slow |

---

## 💡 EN İYİ YAKLAŞIM

### Option 1: Hızlı ve Verimli (ÖNERİLEN)
```bash
# TurkBERT kullan
pip install transformers datasets accelerate

# Model: TurkBERT
# Accuracy beklenisi: %90+
# Eğitim süresi: 2-3 saat
# GPU: 4GB yeterli
```

### Option 2: En Güçlü (LLaMA)
```bash
# LLaMA 3.1 kullan
# Model: LLaMA-3.1-8B
# Accuracy beklenisi: %88+
# Eğitim süresi: 10-20 saat
# GPU: 32GB+ gerekli
```

### Option 3: Mevcut Transformer'ı İyileştir
```bash
# Preprocessing + Hiperparametre + Veri artır
# Accuracy beklenisi: %75-80
# Eğitim süresi: 8 saat
# GPU: 2GB yeterli
```

---

## 🚀 UYGULAMA ADIMLARI

### Adım 1: Kurulum
```bash
pip install transformers datasets accelerate
pip install sentencepiece  # Türkçe tokenization için
pip install optuna  # Hyperparameter optimization için
```

### Adım 2: TurkBERT ile Eğitim
```python
# scripts/train_turkbert.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }

# Model ve tokenizer
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=6
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./turkbert_results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Expected: %90+ accuracy! 🎉
```

### Adım 3: Sonuçları Değerlendir
```python
# Test setinde evaluate
results = trainer.evaluate(eval_dataset=test_dataset)
print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
print(f"Test F1: {results['eval_f1']:.4f}")

# Model kaydet
model.save_pretrained('./best_turkbert')
tokenizer.save_pretrained('./best_turkbert')
```

---

## 📊 BEKLENİLEN SONUÇLAR

### TurkBERT ile:
- **Accuracy:** %88-92 (CNN'den %7-10 daha iyi!)
- **F1-Score:** %87-91
- **Eğitim süresi:** 2-3 saat
- **GPU kullanımı:** 4GB

### mBERT ile:
- **Accuracy:** %85-88
- **F1-Score:** %84-87
- **Eğitim süresi:** 2-3 saat
- **GPU kullanımı:** 4GB

### LLaMA ile:
- **Accuracy:** %86-90
- **F1-Score:** %85-89
- **Eğitim süresi:** 10-20 saat
- **GPU kullanımı:** 32GB+

---

## 🎯 SONUÇ VE ÖNERİ

**En iyi seçenek: TurkBERT** 🌟

**Neden?**
1. ✅ Türkçe için optimize edilmiş
2. ✅ Hafif ve hızlı
3. ✅ %90+ accuracy bekleniyor
4. ✅ 4GB GPU yeterli
5. ✅ Kolay kullanım

**Nasıl Kullanılır?**
1. `pip install transformers datasets`
2. TurkBERT modelini yükle
3. Fine-tuning yap
4. %90+ accuracy al! 🎉

**Beklenen gelişme:**
- Mevcut CNN: %81.44
- TurkBERT ile: **%88-92** (+6-10 puan artış!)

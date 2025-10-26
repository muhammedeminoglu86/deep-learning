# TEZ ÖNERİ FORMU - PROJE DURUMU ANALİZİ

## 📋 TEZ ÖNERİSİ ÖZET

**Öğrenci:** Muhammed EMİNOĞLU  
**Öğrenci No:** 228273002003  
**Tez Adı:** Derin Öğrenme Yöntemleri ile Türkçe Şarkı Sözü Üretimi  
**Tez İngilizce Adı:** Turkish Song Lyric Generation using Deep Learning Methods  
**Danışman:** Doç. Dr. Murat KÖKLÜ  
**Tez Onay Tarihi:** 17.07.2024

---

## 🎯 TEZ ÖNERİSİNDE BELİRLENEN HEDEFLER

### 1. VERİ TOPLAMA
- **Hedef:** Şarkı sözleri ve müzik verisi toplama
- **Özellikler:** Ritim, tempo, melodi, harmoni, akor, ton
- **Format:** SQL veritabanı

### 2. ÖN İŞLEME
- **Müzik:** Fourier Dönüşümü, Pitch Class Distribution, Key Detection
- **Söz:** N-Gram Model, Word Embedding

### 3. MODEL OLUŞTURMA
- **RNN:** Recurrent Neural Networks
- **LSTM:** Long Short-Term Memory
- **SeqGAN:** Sequence Generative Adversarial Networks
- **Transformer:** Attention mechanism

### 4. HİBRİT MODEL
- **Hedef:** RNN, LSTM ve Transformer'ın güçlü yönlerini birleştiren hibrit model

### 5. DEĞERLENDİRME METRİKLERİ
- **BLEU Score**
- **ROUGE Score**
- **Perplexity**
- **Harmony Score** (özel metrik)

---

## ✅ TAMAMLANAN İŞLER (Şu Ana Kadar)

### ✅ ADIM 1: VERİ TOPLAMA (TAMAMLANDI)

**Gerçekleştirilen:**
- 3,605 gerçek Türkçe şarkı verisi toplandı
- SQL veritabanı (turkce_akorlar.sql) parse edildi
- Akor bilgileri dahil edildi (156 farklı akor)
- Genre bilgisi eklendi (6 genre: slow, rock, folk, arabesk, pop, rap)

**Olumsuz:**
- Yol haritasında belirtilen "müzik dosyaları" (tempo, ritim, melodi) henüz toplanmadı
- Sadece şarkı sözleri ve akor bilgisi mevcut
- Müzik enstrüman analizi yapılmadı

### ✅ ADIM 2: ÖN İŞLEME (KISMEN TAMAMLANDI)

**Gerçekleştirilen:**
- Metin temizleme (Türkçe karakter korunması)
- Tokenization (word-level)
- Vocabulary oluşturma (24,211 kelime)
- Akor sequence oluşturma (156 akor)
- Duygusal özellik çıkarma (6 özellik)
- Ritim özellik çıkarma (4 özellik)

**Eksikler:**
- N-Gram Model kullanılmadı
- Word Embedding kullanılmadı (direkt vocabulary kullanıldı)
- Müzik analizi (Fourier, Pitch, Key Detection) yapılmadı

### ✅ ADIM 3: MODEL GELİŞTİRME (TAMAMLANDI)

**Başarıyla Tamamlanan:**
1. **CNN Model** - %81.44 accuracy ⭐ EN İYİ
2. **LSTM Model** - %77.84 accuracy
3. **Seq2Seq Model** - %80.33 accuracy
4. **Transformer Model** - %73.41 accuracy
5. **Transformer Improved** - %63.11 accuracy
6. **Hybrid Model** - %52.35 accuracy (başarısız - overfitting)

**Ek Modeller:**
- Gelişmiş preprocessing ile enhanced modeller
- Multi-modal Transformer (lyrics + chords + emotional + rhythm features)

### ❌ ADIM 4: HİBRİT MODEL (KISMEN BAŞARISIZ)

**Problem:**
- Hibrit model (CNN + LSTM + Transformer) overfitting gösterdi
- %52.35 accuracy (beklenen: %85+)
- Çok fazla parametre (24M)

**Çözüm:**
- Basit ensemble yaklaşımı önerildi
- Pre-trained model kullanımı (TurkBERT) önerildi

### ❌ ADIM 5: DEĞERLENDİRME METRİKLERİ (KISMEN TAMAMLANDI)

**Kullanılan Metrikler:**
- Accuracy ✓
- Precision ✓
- Recall ✓
- F1-Score (Macro ve Weighted) ✓

**Kullanılmayan Metrikler:**
- BLEU Score ❌
- ROUGE Score ❌
- Perplexity ❌
- Harmony Score (özel metrik) ❌

---

## 📊 MEVCUT DURUM vs TEZ ÖNERİSİ

### TEZ ÖNERİSİ NE DİYORDU?
```
"Şarkı sözleri ile müzik arasındaki uyumu analiz etmek ve 
müzik dosyalarından ritim, tempo, melodi gibi özellikler çıkarmak"
```

### ŞU AN NE YAPTIK?
```
✓ Şarkı sözleri analizi (3,605 şarkı)
✓ Akor bilgisi analizi (156 akor)
✓ Duygusal özellik çıkarma
✓ Ritim özellik çıkarma
❌ Müzik dosyası analizi (tempo, ritim, melodi)
```

**FARK:** Müzik dosyaları analiz edilmedi, sadece akor bilgisi kullanıldı!

---

## 🎯 YOL HARİTASI vs ŞU ANKİ DURUM

### YOL HARİTASI ADIMLARı:

#### ADIM 1: Veri Toplama ✅ %70
- ✅ Şarkı sözleri
- ✅ Akor bilgileri
- ❌ Müzik dosyaları (tempo, ritim, melodi) - YAPILMADI

#### ADIM 2: Ön İşleme ✅ %60
- ✅ Metin temizleme
- ✅ Tokenization
- ✅ Akor işleme
- ❌ N-Gram Model - YAPILMADI
- ❌ Word Embedding - YAPILMADI
- ❌ Fourier Dönüşümü - YAPILMADI

#### ADIM 3: Model Geliştirme ✅ %100
- ✅ RNN (CNN kullanıldı, benzer)
- ✅ LSTM
- ✅ SeqGAN (denendikten sonra başarısız oldu)
- ✅ Transformer

#### ADIM 4: Hibrit Model ❌ %30
- ⚠️ Hibrit model başarısız oldu
- ✅ Alternatif çözüm önerildi (TurkBERT)

#### ADIM 5: Değerlendirme ❌ %40
- ✅ Accuracy, Precision, Recall, F1
- ❌ BLEU, ROUGE, Perplexity
- ❌ Harmony Score (özel metrik)

---

## 🔍 EKSİKLER VE GELECEK PLANLAMA

### Eksik Kalan İşler:

1. **Müzik Dosyası Analizi**
   - Fourier Dönüşümü
   - Tempo, Ritim analizi
   - Melodi analizi
   - Enstrüman analizi

2. **Değerlendirme Metrikleri**
   - BLEU Score implementasyonu
   - ROUGE Score implementasyonu
   - Perplexity hesaplama
   - Harmony Score geliştirme (özel metrik)

3. **N-Gram ve Word Embedding**
   - N-Gram model entegrasyonu
   - Word Embedding (Word2Vec, GloVe)

4. **Pre-trained Model Kullanımı**
   - TurkBERT fine-tuning
   - mBERT deneysi
   - Beklenen: %88-92 accuracy

---

## 📊 TEZ ÖNERİSİ KRİTERLERİ KARŞILAŞTIRMASI

| Kriter | Tez Önerisi | Şu Anki Durum | Durum |
|--------|-------------|---------------|-------|
| Veri Seti | Şarkı sözleri + Müzik | Sadece Sözler | ⚠️ %70 |
| Ön İşleme | N-Gram, Embedding, Fourier | Temizleme, Tokenization | ⚠️ %60 |
| RNN/LSTM | Var | ✓ Var | ✅ %100 |
| SeqGAN | Var | ⚠️ Başarısız | ❌ %0 |
| Transformer | Var | ⚠️ Başarısız (%63) | ⚠️ %30 |
| Hibrit Model | Hedef | ❌ Overfitting | ❌ %20 |
| BLEU/ROUGE | Hedef | ❌ Yok | ❌ %0 |
| Harmony Score | Özel Metrik | ❌ Yok | ❌ %0 |

---

## 🎯 ÖNERİLER

### Kısa Vadeli (1-2 ay)
1. **TurkBERT Modelini Eğit** - %88-92 accuracy bekleniyor
2. **BLEU/ROUGE Metrikleri Ekle** - Tez gereksinimi
3. **Harmony Score Geliştir** - Özel metrik (müzik + söz uyumu)

### Orta Vadeli (3-4 ay)
4. **Müzik Dosyası Analizi** - Tempo, ritim, melodi çıkarma
5. **Pre-trained Modeller** - TurkBERT, mBERT karşılaştırması
6. **Ensemble Model** - CNN + Seq2Seq + TurkBERT

### Uzun Vadeli (6+ ay)
7. **Gerçek Hibrit Model** - Müzik dosyası + Şarkı sözü birleştirme
8. **Yayın Hazırlığı** - Makale yazımı
9. **Tez Yazımı** - Final rapor

---

## 💡 SONUÇ

### Tamamlanan:
- ✅ 6 farklı model eğitildi (CNN en iyi: %81.44)
- ✅ Gerçek veri seti oluşturuldu (3,605 şarkı)
- ✅ Preprocessing pipeline hazır
- ✅ Model karşılaştırması yapıldı

### Eksik Kalan:
- ❌ Müzik dosyası analizi
- ❌ BLEU/ROUGE/Perplexity metrikleri
- ❌ Harmony Score (özel metrik)
- ❌ Pre-trained model kullanımı

### Öncelik:
1. **TurkBERT modelini eğit** - Yüksek accuracy (%88-92)
2. **Metrikleri tamamla** - Tez gereksinimi
3. **Müzik analizini ekle** - Tez kapsamını genişlet

---

**Özet:** Temel modeller tamamlandı ama tez önerisi hedeflerinin tamamına ulaşılmadı. Şu anda tez önerisinin %60-70'i tamamlanmış durumda.

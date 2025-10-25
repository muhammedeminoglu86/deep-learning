# TÜRKÇE ŞARKI SÖZÜ ÜRETİMİ - DEEP LEARNING PROJESİ
## Final Rapor (Güncellenmiş)

**Tarih:** 25/10/2025 22:30
**Proje:** Türkçe Şarkı Sözü Üretimi için 5 Farklı Deep Learning Modelinin Karşılaştırılması

---

## 📊 PROJE ÖZETİ

Bu proje, Türkçe şarkı sözlerini analiz ederek müzik türlerini (genre) sınıflandırmak için 5 farklı deep learning modelini karşılaştırmıştır. **Geliştirilmiş veri temizleme** ve **multi-modal Transformer** yaklaşımı ile daha kapsamlı analiz yapılmıştır.

### 🎯 Amaç
- Türkçe şarkı sözlerini analiz etmek
- Müzik türlerini otomatik olarak sınıflandırmak
- Farklı model mimarilerinin performansını karşılaştırmak
- Akor bilgilerinin model performansına etkisini değerlendirmek
- Geliştirilmiş preprocessing ile daha iyi sonuçlar elde etmek

---

## 📈 VERİ SETİ

### Veri Kaynağı
- **Kaynak:** SQL veritabanı (turkce_akorlar.sql)
- **Toplam Şarkı:** 3,605 gerçek Türkçe şarkı
- **Sanatçı Sayısı:** 1,473 farklı sanatçı
- **Akor Bilgisi:** Her şarkı için akor progresyonları dahil

### Geliştirilmiş Veri İstatistikleri
- **Genre Sayısı:** 6 (slow, rock, folk, arabesk, pop, rap)
- **Kelime Dağarcığı:** 24,211 benzersiz kelime (önceden 20,317)
- **Akor Dağarcığı:** 156 farklı akor (önceden 109)
- **Ortalama Şarkı Uzunluğu:** 149.7 kelime
- **Ortalama Akor Sayısı:** 4.4 akor per şarkı
- **Akor Çeşitliliği:** 0.972 (çok yüksek!)

### Genre Dağılımı
- **Slow:** 1,446 şarkı (40.1%)
- **Rock:** 865 şarkı (24.0%)
- **Folk:** 675 şarkı (18.7%)
- **Arabesk:** 384 şarkı (10.7%)
- **Pop:** 227 şarkı (6.3%)
- **Rap:** 8 şarkı (0.2%)

---

## 🤖 MODELLER

### 1. Seq2Seq (Encoder-Decoder)
- **Accuracy:** 80.33%
- **Parametre:** 9.3M
- **Mimari:** LSTM tabanlı encoder-decoder
- **Özellik:** Bidirectional encoder

### 2. LSTM (Long Short-Term Memory)
- **Accuracy:** 77.84%
- **Parametre:** 5.1M
- **Mimari:** 2 katmanlı bidirectional LSTM
- **Özellik:** Attention mekanizması

### 3. CNN (Convolutional Neural Network)
- **Accuracy:** 81.44% ⭐ **EN İYİ**
- **Parametre:** 2.9M
- **Mimari:** 1D CNN + MaxPooling
- **Özellik:** En verimli model

### 4. Transformer (Original)
- **Accuracy:** 73.41%
- **Parametre:** 3.9M
- **Mimari:** Basic Transformer
- **Özellik:** Multi-head attention

### 5. Transformer (Improved - Multi-modal)
- **Accuracy:** 63.11%
- **Parametre:** 9.8M
- **Mimari:** Multi-modal Transformer + akor bilgisi
- **Özellik:** Lyrics + Chords fusion, geliştirilmiş preprocessing

---

## 🏆 SONUÇLAR

### Model Performans Sıralaması
1. **CNN:** 81.44% accuracy ⭐ **EN İYİ**
2. **Seq2Seq:** 80.33% accuracy
3. **LSTM:** 77.84% accuracy
4. **Transformer (Original):** 73.41% accuracy
5. **Transformer (Improved):** 63.11% accuracy

### Verimlilik Analizi
- **En Verimli:** CNN (0.28 accuracy/parametre oranı)
- **En Az Parametre:** CNN (2.9M)
- **En Çok Parametre:** Transformer Improved (9.8M)

### Geliştirilmiş Preprocessing Sonuçları
- **Kelime dağarcığı:** 24,211 (önceden 20,317) - %19 artış
- **Akor dağarcığı:** 156 (önceden 109) - %43 artış
- **Akor çeşitliliği:** 0.972 (önceden 0.007) - Dramatik iyileştirme
- **Ortalama akor sayısı:** 4.4 (önceden 0.0) - Tamamen düzeltildi

### Beklenmeyen Sonuçlar
- **CNN Başarısı:** En basit mimari en yüksek performansı gösterdi
- **Transformer Düşük Performansı:** Multi-modal yaklaşım beklenen iyileştirmeyi sağlamadı
- **Akorların Etkisi:** Geliştirilmiş akor verisi Transformer'da beklenen iyileştirmeyi sağlamadı
- **Overfitting:** Seq2Seq modelinde erken overfitting gözlendi

---

## 🔍 DETAYLI ANALİZ

### Preprocessing İyileştirmeleri
1. **Veri Temizleme:** Daha az agresif stopwords filtreleme
2. **Akor Çıkarma:** SQL'den direkt akor bilgisi kullanımı
3. **Sequence Length:** 200 token'a çıkarıldı (önceden 100)
4. **Vocabulary:** Minimum frekans filtresi eklendi
5. **Chord Features:** 8 farklı akor özelliği çıkarıldı

### Model Karşılaştırması
- **CNN:** Bu veri seti için en uygun seçim
- **Seq2Seq:** İyi performans ama çok fazla parametre
- **LSTM:** Dengeli performans
- **Transformer:** Modern mimari ama beklenen başarıyı sağlamadı

### Olası Nedenler
1. **Veri Seti Boyutu:** 3,605 şarkı Transformer için yetersiz olabilir
2. **Akor Verisi Kalitesi:** Akor parsing algoritması optimize edilebilir
3. **Hiperparametre Optimizasyonu:** Transformer modelleri için daha fazla tuning gerekli
4. **Sequence Length:** 200 token uzunluğu yeterli olmayabilir

---

## 📁 OLUŞTURULAN DOSYALAR

### Veri Dosyaları
- `data/raw/turkish_lyrics_enhanced_dataset.csv` - Ham veri
- `data/processed/train_improved.csv` - Geliştirilmiş eğitim verisi
- `data/processed/validation_improved.csv` - Geliştirilmiş doğrulama verisi
- `data/processed/test_improved.csv` - Geliştirilmiş test verisi

### Model Dosyaları
- `models/seq2seq/best_model.pth` - Seq2Seq modeli
- `models/lstm/best_model.pth` - LSTM modeli
- `models/cnn/best_model.pth` - CNN modeli
- `models/transformer/best_model_enhanced.pth` - Transformer modeli
- `models/transformer_improved/best_model.pth` - Geliştirilmiş Transformer modeli

### Grafikler ve Görselleştirmeler
- `results/final_model_comparison.png` - Final model karşılaştırması
- `results/comprehensive_model_comparison.png` - Kapsamlı model karşılaştırması
- `results/all_models_training_history.png` - Tüm modellerin eğitim geçmişi
- `results/improved_genre_distribution.png` - Geliştirilmiş genre dağılımı
- `results/improved_chord_features_distribution.png` - Geliştirilmiş akor özellikleri

### Script'ler
- `scripts/parse_sql_enhanced.py` - Veri çıkarma
- `scripts/preprocess_improved.py` - Geliştirilmiş veri işleme
- `scripts/train_seq2seq.py` - Seq2Seq eğitimi
- `scripts/train_lstm.py` - LSTM eğitimi
- `scripts/train_cnn.py` - CNN eğitimi
- `scripts/train_transformer_enhanced.py` - Transformer eğitimi
- `scripts/train_transformer_improved.py` - Geliştirilmiş Transformer eğitimi
- `scripts/final_comparison.py` - Final karşılaştırma

---

## 💡 ÖNERİLER

### Kısa Vadeli
1. **CNN Modelini Optimize Et:** En iyi performans gösteren model
2. **Veri Setini Genişlet:** Daha fazla şarkı ekle
3. **Hiperparametre Optimizasyonu:** Grid search ile en iyi parametreleri bul
4. **Ensemble Methods:** CNN + LSTM kombinasyonu dene

### Uzun Vadeli
1. **Akor Verisi Kalitesi:** Akor parsing algoritmasını iyileştir
2. **Multi-Modal Yaklaşım:** Melodi ve ritim bilgisi ekle
3. **Transfer Learning:** Pre-trained modelleri kullan
4. **Transformer Optimizasyonu:** Daha büyük veri seti ile tekrar dene
5. **Veri Augmentation:** Şarkı sözlerini çeşitlendir

### Teknik İyileştirmeler
1. **Cross-validation:** Daha güvenilir sonuçlar için
2. **Hyperparameter Tuning:** Otomatik optimizasyon
3. **Model Interpretability:** SHAP, LIME gibi araçlar
4. **Real-time Inference:** Canlı sınıflandırma sistemi

---

## 🎯 SONUÇ

Bu proje, Türkçe şarkı sözü analizi için **CNN modelinin en uygun seçim** olduğunu göstermiştir. Geliştirilmiş preprocessing ile veri kalitesi artırılmış, ancak Transformer modellerinin beklenen performansı göstermemesi, veri seti boyutu ve kalitesi ile ilgili olabilir.

### Ana Bulgular
1. **CNN modeli** bu veri seti için en uygun seçim (%81.44 accuracy)
2. **Geliştirilmiş preprocessing** veri kalitesini önemli ölçüde artırdı
3. **Transformer modelleri** beklenen performansı göstermedi
4. **Akor bilgisi** multi-modal yaklaşımda beklenen iyileştirmeyi sağlamadı
5. **Veri seti boyutu** Transformer modelleri için yetersiz olabilir

### Proje Başarıları
- ✅ 5 farklı deep learning modeli başarıyla eğitildi
- ✅ 3,605 gerçek Türkçe şarkı verisi işlendi
- ✅ Geliştirilmiş preprocessing pipeline oluşturuldu
- ✅ Kapsamlı model karşılaştırması yapıldı
- ✅ Tüm sonuçlar dokümante edildi

**Proje başarıyla tamamlanmıştır ve tüm sonuçlar GitHub'da mevcuttur.**

---

## 📊 PERFORMANS ÖZETİ

| Model | Accuracy | Parametre | Verimlilik | Açıklama |
|-------|----------|-----------|------------|----------|
| CNN | **81.44%** | 2.9M | **0.28** | En iyi performans |
| Seq2Seq | 80.33% | 9.3M | 0.09 | İyi ama fazla parametre |
| LSTM | 77.84% | 5.1M | 0.15 | Dengeli performans |
| Transformer (O) | 73.41% | 3.9M | 0.19 | Modern mimari |
| Transformer (I) | 63.11% | 9.8M | 0.06 | Multi-modal |

---

*Bu rapor otomatik olarak oluşturulmuştur - 25/10/2025 22:30*

**GitHub Repository:** https://github.com/muhammedeminoglu86/deep-learning

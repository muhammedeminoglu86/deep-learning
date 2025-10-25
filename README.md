# Türkçe Şarkı Sözü Üretimi - Tez Projesi

Bu proje, müzik türüne göre Türkçe şarkı sözü üretimi için 4 farklı deep learning modelini karşılaştırmaktadır.

## Modeller
- Seq2Seq (Encoder-Decoder)
- LSTM (Long Short-Term Memory)
- CNN (Convolutional Neural Network)
- Transformer

## Kurulum

### Gereksinimler
- Python 3.12+
- PyTorch 2.9+
- CUDA 12.1+ (opsiyonel)

### Environment Kurulumu
```bash
conda create -n thesis python=3.12
conda activate thesis
pip install torch transformers pandas numpy scikit-learn beautifulsoup4 requests nltk rouge-score matplotlib seaborn
```

## Proje Yapısı
```
thesis_project/
├── data/
│   ├── raw/           # Ham veri
│   └── processed/     # İşlenmiş veri
├── models/
│   ├── seq2seq/       # Seq2Seq model
│   ├── lstm/          # LSTM model
│   ├── cnn/           # CNN model
│   └── transformer/   # Transformer model
├── notebooks/          # Jupyter notebook'lar
├── results/           # Sonuçlar ve grafikler
└── utils/             # Yardımcı fonksiyonlar
```

## Kullanım
```bash
# Veri toplama
python scripts/scrape_lyrics.py

# Model eğitimi
python scripts/train_seq2seq.py
python scripts/train_lstm.py
python scripts/train_cnn.py
python scripts/train_transformer.py

# Değerlendirme
python scripts/evaluate_models.py
```

## Lisans
MIT License

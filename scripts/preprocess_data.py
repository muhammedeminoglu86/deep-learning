import pandas as pd
import numpy as np
import re
import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

class TurkishLyricsPreprocessor:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.vocab = set()
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.genre_encoder = LabelEncoder()
        
        # Türkçe stopwords (basit liste)
        self.turkish_stopwords = {
            've', 'bir', 'bu', 'da', 'de', 'ile', 'için', 'olan', 'olan', 'gibi',
            'kadar', 'daha', 'çok', 'en', 'hiç', 'bile', 'sadece', 'hem', 'ya',
            'ya da', 'ama', 'fakat', 'ancak', 'çünkü', 'eğer', 'ise', 'ki', 'mi',
            'mı', 'mu', 'mü', 'ne', 'nasıl', 'neden', 'niçin', 'kim', 'kime',
            'kimin', 'kimde', 'kimden', 'kimle', 'kimse', 'her', 'herkes', 'hiçbir',
            'bazı', 'birkaç', 'birçok', 'tüm', 'bütün', 'hepsi', 'hiçbiri'
        }
        
    def load_data(self, file_path):
        """Veriyi yükle"""
        print(f"Veri yükleniyor: {file_path}")
        
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            self.data = pd.DataFrame(data_list)
        
        print(f"Yüklenen veri: {len(self.data)} şarkı")
        print(f"Kolonlar: {list(self.data.columns)}")
        return self.data
    
    def clean_text(self, text):
        """Metni temizle"""
        if pd.isna(text):
            return ""
        
        # Küçük harfe çevir
        text = text.lower()
        
        # Özel karakterleri temizle (Türkçe karakterleri koru)
        text = re.sub(r'[^\w\sçğıöşüÇĞIİÖŞÜ]', ' ', text)
        
        # Çoklu boşlukları tek boşluğa çevir
        text = re.sub(r'\s+', ' ', text)
        
        # Başta ve sonda boşlukları temizle
        text = text.strip()
        
        return text
    
    def tokenize_text(self, text):
        """Metni tokenize et"""
        if not text:
            return []
        
        # Basit tokenization (kelime bazında)
        tokens = text.split()
        
        # Stopwords'leri çıkar
        tokens = [token for token in tokens if token not in self.turkish_stopwords]
        
        # Çok kısa tokenları çıkar (1-2 karakter)
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def build_vocabulary(self, texts):
        """Kelime dağarcığı oluştur"""
        print("Kelime dağarcığı oluşturuluyor...")
        
        all_tokens = []
        for text in texts:
            tokens = self.tokenize_text(text)
            all_tokens.extend(tokens)
        
        # Kelime frekanslarını hesapla
        word_counts = Counter(all_tokens)
        
        # En az 2 kez geçen kelimeleri al
        min_freq = 2
        frequent_words = [word for word, count in word_counts.items() if count >= min_freq]
        
        # Özel tokenlar ekle
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        self.vocab = special_tokens + frequent_words
        
        # Kelime-indeks mapping'leri oluştur
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Kelime dağarcığı boyutu: {len(self.vocab)}")
        print(f"En sık kullanılan 10 kelime: {word_counts.most_common(10)}")
        
        return self.vocab
    
    def text_to_sequence(self, text, max_length=100):
        """Metni sayı dizisine çevir"""
        tokens = self.tokenize_text(text)
        
        # Kelimeleri indekslere çevir
        sequence = []
        for token in tokens:
            if token in self.word_to_idx:
                sequence.append(self.word_to_idx[token])
            else:
                sequence.append(self.word_to_idx['<UNK>'])
        
        # Padding veya truncation
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence.extend([self.word_to_idx['<PAD>']] * (max_length - len(sequence)))
        
        return sequence
    
    def preprocess_dataset(self, max_length=100):
        """Tüm dataset'i preprocess et"""
        print("Dataset preprocessing başlıyor...")
        
        if self.data is None:
            raise ValueError("Önce veriyi yükleyin!")
        
        # Şarkı sözlerini temizle
        print("Metinler temizleniyor...")
        self.data['cleaned_lyrics'] = self.data['lyrics'].apply(self.clean_text)
        
        # Çok kısa şarkıları filtrele
        min_length = 20
        self.data = self.data[self.data['cleaned_lyrics'].str.len() >= min_length]
        print(f"Temizleme sonrası: {len(self.data)} şarkı")
        
        # Kelime dağarcığı oluştur
        self.build_vocabulary(self.data['cleaned_lyrics'])
        
        # Metinleri sayı dizilerine çevir
        print("Metinler sayı dizilerine çevriliyor...")
        self.data['sequence'] = self.data['cleaned_lyrics'].apply(
            lambda x: self.text_to_sequence(x, max_length)
        )
        
        # Genre'leri encode et
        print("Genre'ler encode ediliyor...")
        self.data['genre_encoded'] = self.genre_encoder.fit_transform(self.data['genre'])
        
        # İstatistikler
        self.print_statistics()
        
        return self.data
    
    def print_statistics(self):
        """Dataset istatistiklerini yazdır"""
        print("\n" + "="*50)
        print("DATASET İSTATİSTİKLERİ")
        print("="*50)
        
        print(f"Toplam şarkı sayısı: {len(self.data)}")
        print(f"Toplam kelime sayısı: {len(self.vocab)}")
        
        # Genre dağılımı
        print(f"\nGenre dağılımı:")
        genre_counts = self.data['genre'].value_counts()
        for genre, count in genre_counts.items():
            print(f"  {genre}: {count} şarkı")
        
        # Şarkı uzunlukları
        lyrics_lengths = self.data['cleaned_lyrics'].str.len()
        print(f"\nŞarkı uzunlukları:")
        print(f"  Ortalama: {lyrics_lengths.mean():.1f} karakter")
        print(f"  Medyan: {lyrics_lengths.median():.1f} karakter")
        print(f"  Min: {lyrics_lengths.min()} karakter")
        print(f"  Max: {lyrics_lengths.max()} karakter")
        
        # Kelime sayıları
        word_counts = self.data['cleaned_lyrics'].apply(lambda x: len(x.split()))
        print(f"\nKelime sayıları:")
        print(f"  Ortalama: {word_counts.mean():.1f} kelime")
        print(f"  Medyan: {word_counts.median():.1f} kelime")
        print(f"  Min: {word_counts.min()} kelime")
        print(f"  Max: {word_counts.max()} kelime")
    
    def split_data(self, test_size=0.2, val_size=0.1):
        """Veriyi train/validation/test olarak böl"""
        print("Veri bölünüyor...")
        
        # Önce train/test split
        train_data, test_data = train_test_split(
            self.data, test_size=test_size, random_state=42, stratify=self.data['genre']
        )
        
        # Sonra train'i train/val olarak böl
        train_data, val_data = train_test_split(
            train_data, test_size=val_size/(1-test_size), random_state=42, stratify=train_data['genre']
        )
        
        print(f"Train: {len(train_data)} şarkı")
        print(f"Validation: {len(val_data)} şarkı")
        print(f"Test: {len(test_data)} şarkı")
        
        return train_data, val_data, test_data
    
    def save_processed_data(self, train_data, val_data, test_data, output_dir="data/processed"):
        """İşlenmiş veriyi kaydet"""
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV formatında kaydet
        train_data.to_csv(f"{output_dir}/train.csv", index=False, encoding='utf-8')
        val_data.to_csv(f"{output_dir}/validation.csv", index=False, encoding='utf-8')
        test_data.to_csv(f"{output_dir}/test.csv", index=False, encoding='utf-8')
        
        # JSON formatında da kaydet
        train_data.to_json(f"{output_dir}/train.json", orient='records', force_ascii=False, indent=2)
        val_data.to_json(f"{output_dir}/validation.json", orient='records', force_ascii=False, indent=2)
        test_data.to_json(f"{output_dir}/test.json", orient='records', force_ascii=False, indent=2)
        
        # Kelime dağarcığını kaydet
        vocab_data = {
            'vocab': self.vocab,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'genre_classes': self.genre_encoder.classes_.tolist()
        }
        
        with open(f"{output_dir}/vocabulary.json", 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nİşlenmiş veri kaydedildi:")
        print(f"  Train: {output_dir}/train.csv")
        print(f"  Validation: {output_dir}/validation.csv")
        print(f"  Test: {output_dir}/test.csv")
        print(f"  Vocabulary: {output_dir}/vocabulary.json")
    
    def create_visualizations(self, output_dir="results"):
        """Veri görselleştirmeleri oluştur"""
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('default')
        
        # Genre dağılımı
        plt.figure(figsize=(10, 6))
        genre_counts = self.data['genre'].value_counts()
        plt.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%')
        plt.title('Genre Dağılımı')
        plt.savefig(f"{output_dir}/genre_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Şarkı uzunlukları
        plt.figure(figsize=(12, 6))
        lyrics_lengths = self.data['cleaned_lyrics'].str.len()
        plt.hist(lyrics_lengths, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Şarkı Uzunluğu (karakter)')
        plt.ylabel('Frekans')
        plt.title('Şarkı Uzunluk Dağılımı')
        plt.savefig(f"{output_dir}/lyrics_length_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Genre'lere göre uzunluk dağılımı
        plt.figure(figsize=(12, 8))
        genres = self.data['genre'].unique()
        for i, genre in enumerate(genres):
            genre_data = self.data[self.data['genre'] == genre]['cleaned_lyrics'].str.len()
            plt.subplot(2, 3, i+1)
            plt.hist(genre_data, bins=15, alpha=0.7, edgecolor='black')
            plt.title(f'{genre.capitalize()} - Uzunluk Dağılımı')
            plt.xlabel('Karakter Sayısı')
            plt.ylabel('Frekans')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/genre_length_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Görselleştirmeler kaydedildi: {output_dir}/")

def main():
    """Ana preprocessing fonksiyonu"""
    print("Türkçe Şarkı Sözü Preprocessing Başlıyor...")
    print("="*60)
    
    # Preprocessor oluştur
    preprocessor = TurkishLyricsPreprocessor()
    
    # Veriyi yükle
    data_path = "data/raw/turkish_lyrics_real_dataset.csv"
    preprocessor.load_data(data_path)
    
    # Preprocessing yap
    processed_data = preprocessor.preprocess_dataset(max_length=100)
    
    # Veriyi böl
    train_data, val_data, test_data = preprocessor.split_data(test_size=0.2, val_size=0.1)
    
    # Kaydet
    preprocessor.save_processed_data(train_data, val_data, test_data)
    
    # Görselleştirmeler oluştur
    preprocessor.create_visualizations()
    
    print("\n" + "="*60)
    print("Preprocessing tamamlandı!")
    print("Artık model eğitimine geçebiliriz.")

if __name__ == "__main__":
    main()

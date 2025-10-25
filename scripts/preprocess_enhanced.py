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
import warnings
warnings.filterwarnings('ignore')

class EnhancedTurkishLyricsPreprocessor:
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.vocab = set()
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.chord_vocab = set()
        self.chord_to_idx = {}
        self.idx_to_chord = {}
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
        
        # Akor kategorileri
        self.chord_categories = {
            'major': ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C#', 'D#', 'F#', 'G#', 'A#'],
            'minor': ['CM', 'DM', 'EM', 'FM', 'GM', 'AM', 'BM', 'C#M', 'D#M', 'F#M', 'G#M', 'A#M'],
            'seventh': ['C7', 'D7', 'E7', 'F7', 'G7', 'A7', 'B7'],
            'complex': ['SUS', 'ADD', 'DIM', 'AUG', 'MAJ']
        }
        
    def load_data(self, file_path):
        """Veriyi yükle"""
        print(f"Gelişmiş veri yükleniyor: {file_path}")
        
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
    
    def build_chord_vocabulary(self, chord_lists):
        """Akor dağarcığı oluştur"""
        print("Akor dağarcığı oluşturuluyor...")
        
        all_chords = []
        for chords in chord_lists:
            if isinstance(chords, list):
                all_chords.extend(chords)
            elif isinstance(chords, str):
                # String olarak kaydedilmiş listeyi parse et
                try:
                    chord_list = eval(chords)
                    all_chords.extend(chord_list)
                except:
                    pass
        
        # Akor frekanslarını hesapla
        chord_counts = Counter(all_chords)
        
        # En az 2 kez geçen akorları al
        min_freq = 2
        frequent_chords = [chord for chord, count in chord_counts.items() if count >= min_freq]
        
        # Özel tokenlar ekle
        special_chords = ['<PAD>', '<UNK>', '<START>', '<END>']
        self.chord_vocab = special_chords + frequent_chords
        
        # Akor-indeks mapping'leri oluştur
        self.chord_to_idx = {chord: idx for idx, chord in enumerate(self.chord_vocab)}
        self.idx_to_chord = {idx: chord for chord, idx in self.chord_to_idx.items()}
        
        print(f"Akor dağarcığı boyutu: {len(self.chord_vocab)}")
        print(f"En sık kullanılan 10 akor: {chord_counts.most_common(10)}")
        
        return self.chord_vocab
    
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
    
    def chords_to_sequence(self, chords, max_length=20):
        """Akorları sayı dizisine çevir"""
        if isinstance(chords, str):
            try:
                chord_list = eval(chords)
            except:
                chord_list = []
        elif isinstance(chords, list):
            chord_list = chords
        else:
            chord_list = []
        
        # Akorları indekslere çevir
        sequence = []
        for chord in chord_list:
            if chord in self.chord_to_idx:
                sequence.append(self.chord_to_idx[chord])
            else:
                sequence.append(self.chord_to_idx['<UNK>'])
        
        # Padding veya truncation
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence.extend([self.chord_to_idx['<PAD>']] * (max_length - len(sequence)))
        
        return sequence
    
    def extract_chord_features(self, chords):
        """Akor özelliklerini çıkar"""
        if isinstance(chords, str):
            try:
                chord_list = eval(chords)
            except:
                chord_list = []
        elif isinstance(chords, list):
            chord_list = chords
        else:
            chord_list = []
        
        features = {
            'total_chords': len(chord_list),
            'unique_chords': len(set(chord_list)),
            'major_count': 0,
            'minor_count': 0,
            'seventh_count': 0,
            'complex_count': 0,
            'chord_diversity': 0
        }
        
        if chord_list:
            for chord in chord_list:
                chord_upper = chord.upper()
                if 'M' in chord_upper and 'MAJ' not in chord_upper:
                    features['minor_count'] += 1
                elif '7' in chord_upper:
                    features['seventh_count'] += 1
                elif any(x in chord_upper for x in ['SUS', 'ADD', 'DIM', 'AUG']):
                    features['complex_count'] += 1
                else:
                    features['major_count'] += 1
            
            features['chord_diversity'] = features['unique_chords'] / features['total_chords']
        
        return features
    
    def preprocess_dataset(self, max_length=100, chord_max_length=20):
        """Tüm dataset'i preprocess et"""
        print("Gelişmiş dataset preprocessing başlıyor...")
        
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
        
        # Akor dağarcığı oluştur
        self.build_chord_vocabulary(self.data['chords'])
        
        # Metinleri sayı dizilerine çevir
        print("Metinler sayı dizilerine çevriliyor...")
        self.data['sequence'] = self.data['cleaned_lyrics'].apply(
            lambda x: self.text_to_sequence(x, max_length)
        )
        
        # Akorları sayı dizilerine çevir
        print("Akorlar sayı dizilerine çevriliyor...")
        self.data['chord_sequence'] = self.data['chords'].apply(
            lambda x: self.chords_to_sequence(x, chord_max_length)
        )
        
        # Akor özelliklerini çıkar
        print("Akor özellikleri çıkarılıyor...")
        chord_features = self.data['chords'].apply(self.extract_chord_features)
        
        # Akor özelliklerini ayrı kolonlar olarak ekle
        for feature in ['total_chords', 'unique_chords', 'major_count', 'minor_count', 'seventh_count', 'complex_count', 'chord_diversity']:
            self.data[f'chord_{feature}'] = chord_features.apply(lambda x: x[feature])
        
        # Genre'leri encode et
        print("Genre'ler encode ediliyor...")
        self.data['genre_encoded'] = self.genre_encoder.fit_transform(self.data['genre'])
        
        # İstatistikler
        self.print_statistics()
        
        return self.data
    
    def print_statistics(self):
        """Dataset istatistiklerini yazdır"""
        print("\n" + "="*60)
        print("GELİŞMİŞ DATASET İSTATİSTİKLERİ")
        print("="*60)
        
        print(f"Toplam şarkı sayısı: {len(self.data)}")
        print(f"Toplam kelime sayısı: {len(self.vocab)}")
        print(f"Toplam akor sayısı: {len(self.chord_vocab)}")
        
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
        
        # Akor istatistikleri
        print(f"\nAkor istatistikleri:")
        print(f"  Ortalama akor sayısı: {self.data['chord_total_chords'].mean():.1f}")
        print(f"  Ortalama benzersiz akor: {self.data['chord_unique_chords'].mean():.1f}")
        print(f"  Ortalama akor çeşitliliği: {self.data['chord_chord_diversity'].mean():.3f}")
    
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
        train_data.to_csv(f"{output_dir}/train_enhanced.csv", index=False, encoding='utf-8')
        val_data.to_csv(f"{output_dir}/validation_enhanced.csv", index=False, encoding='utf-8')
        test_data.to_csv(f"{output_dir}/test_enhanced.csv", index=False, encoding='utf-8')
        
        # JSON formatında da kaydet
        train_data.to_json(f"{output_dir}/train_enhanced.json", orient='records', force_ascii=False, indent=2)
        val_data.to_json(f"{output_dir}/validation_enhanced.json", orient='records', force_ascii=False, indent=2)
        test_data.to_json(f"{output_dir}/test_enhanced.json", orient='records', force_ascii=False, indent=2)
        
        # Kelime ve akor dağarcığını kaydet
        vocab_data = {
            'vocab': self.vocab,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'chord_vocab': self.chord_vocab,
            'chord_to_idx': self.chord_to_idx,
            'idx_to_chord': self.idx_to_chord,
            'genre_classes': self.genre_encoder.classes_.tolist()
        }
        
        with open(f"{output_dir}/vocabulary_enhanced.json", 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nİşlenmiş veri kaydedildi:")
        print(f"  Train: {output_dir}/train_enhanced.csv")
        print(f"  Validation: {output_dir}/validation_enhanced.csv")
        print(f"  Test: {output_dir}/test_enhanced.csv")
        print(f"  Vocabulary: {output_dir}/vocabulary_enhanced.json")
    
    def create_visualizations(self, output_dir="results"):
        """Veri görselleştirmeleri oluştur"""
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('default')
        
        # Genre dağılımı
        plt.figure(figsize=(12, 8))
        genre_counts = self.data['genre'].value_counts()
        plt.pie(genre_counts.values, labels=genre_counts.index, autopct='%1.1f%%')
        plt.title('Genre Dağılımı (Akorlar Dahil)')
        plt.savefig(f"{output_dir}/genre_distribution_enhanced.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Akor dağılımı
        plt.figure(figsize=(15, 8))
        all_chords = []
        for chords in self.data['chords']:
            if isinstance(chords, str):
                try:
                    chord_list = eval(chords)
                    all_chords.extend(chord_list)
                except:
                    pass
            elif isinstance(chords, list):
                all_chords.extend(chords)
        
        chord_counts = Counter(all_chords)
        top_chords = chord_counts.most_common(15)
        
        chords, counts = zip(*top_chords)
        plt.bar(range(len(chords)), counts)
        plt.xticks(range(len(chords)), chords, rotation=45)
        plt.title('En Sık Kullanılan Akorlar')
        plt.xlabel('Akorlar')
        plt.ylabel('Kullanım Sayısı')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/chord_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Akor özellikleri dağılımı
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].hist(self.data['chord_total_chords'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Toplam Akor Sayısı Dağılımı')
        axes[0, 0].set_xlabel('Akor Sayısı')
        axes[0, 0].set_ylabel('Frekans')
        
        axes[0, 1].hist(self.data['chord_unique_chords'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Benzersiz Akor Sayısı Dağılımı')
        axes[0, 1].set_xlabel('Benzersiz Akor Sayısı')
        axes[0, 1].set_ylabel('Frekans')
        
        axes[1, 0].hist(self.data['chord_chord_diversity'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Akor Çeşitliliği Dağılımı')
        axes[1, 0].set_xlabel('Çeşitlilik Oranı')
        axes[1, 0].set_ylabel('Frekans')
        
        # Genre'lere göre akor çeşitliliği
        genre_diversity = self.data.groupby('genre')['chord_chord_diversity'].mean()
        axes[1, 1].bar(range(len(genre_diversity)), genre_diversity.values)
        axes[1, 1].set_xticks(range(len(genre_diversity)))
        axes[1, 1].set_xticklabels(genre_diversity.index, rotation=45)
        axes[1, 1].set_title('Genre\'lere Göre Ortalama Akor Çeşitliliği')
        axes[1, 1].set_ylabel('Ortalama Çeşitlilik')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/chord_features_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Görselleştirmeler kaydedildi: {output_dir}/")

def main():
    """Ana preprocessing fonksiyonu"""
    print("Gelişmiş Türkçe Şarkı Sözü Preprocessing Başlıyor...")
    print("="*70)
    
    # Preprocessor oluştur
    preprocessor = EnhancedTurkishLyricsPreprocessor()
    
    # Veriyi yükle
    data_path = "data/raw/turkish_lyrics_enhanced_dataset.csv"
    preprocessor.load_data(data_path)
    
    # Preprocessing yap
    processed_data = preprocessor.preprocess_dataset(max_length=100, chord_max_length=20)
    
    # Veriyi böl
    train_data, val_data, test_data = preprocessor.split_data(test_size=0.2, val_size=0.1)
    
    # Kaydet
    preprocessor.save_processed_data(train_data, val_data, test_data)
    
    # Görselleştirmeler oluştur
    preprocessor.create_visualizations()
    
    print("\n" + "="*70)
    print("Gelişmiş preprocessing tamamlandı!")
    print("Artık akorları da içeren model eğitimine geçebiliriz.")

if __name__ == "__main__":
    main()


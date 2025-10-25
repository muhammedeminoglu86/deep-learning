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

class ImprovedTurkishLyricsPreprocessor:
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
        
        # Çok daha az stopwords - sadece gerçekten gereksiz olanlar
        self.turkish_stopwords = {
            've', 'bir', 'bu', 'da', 'de', 'ile', 'için', 'olan', 'gibi',
            'kadar', 'daha', 'çok', 'en', 'hiç', 'bile', 'sadece', 'hem',
            'ama', 'fakat', 'ancak', 'çünkü', 'eğer', 'ise', 'ki', 'mi',
            'mı', 'mu', 'mü', 'ne', 'nasıl', 'neden', 'niçin', 'kim',
            'her', 'herkes', 'hiçbir', 'bazı', 'birkaç', 'birçok', 'tüm',
            'bütün', 'hepsi', 'hiçbiri'
        }
        
        # Akor kategorileri - daha kapsamlı
        self.chord_categories = {
            'major': ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C#', 'D#', 'F#', 'G#', 'A#', 'Cb', 'Db', 'Eb', 'Gb', 'Ab', 'Bb'],
            'minor': ['Cm', 'Dm', 'Em', 'Fm', 'Gm', 'Am', 'Bm', 'C#m', 'D#m', 'F#m', 'G#m', 'A#m', 'Cbm', 'Dbm', 'Ebm', 'Gbm', 'Abm', 'Bbm'],
            'seventh': ['C7', 'D7', 'E7', 'F7', 'G7', 'A7', 'B7', 'C#7', 'D#7', 'F#7', 'G#7', 'A#7'],
            'complex': ['sus', 'add', 'dim', 'aug', 'maj', 'min', 'm7', 'maj7', 'dim7', 'aug7']
        }
        
    def load_data(self, file_path):
        """Veriyi yükle"""
        print(f"Geliştirilmiş veri yükleniyor: {file_path}")
        
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            self.data = pd.DataFrame(data_list)
        
        print(f"Yüklenen veri: {len(self.data)} şarkı")
        print(f"Kolonlar: {list(self.data.columns)}")
        return self.data
    
    def clean_text_improved(self, text):
        """Geliştirilmiş metin temizleme"""
        if pd.isna(text):
            return ""
        
        # Küçük harfe çevir
        text = text.lower()
        
        # Sadece gerçekten gereksiz karakterleri temizle
        # Türkçe karakterleri ve noktalama işaretlerini koru
        text = re.sub(r'[^\w\sçğıöşüÇĞIİÖŞÜ.,!?;:\-\(\)\[\]\"\']', ' ', text)
        
        # Çoklu boşlukları tek boşluğa çevir
        text = re.sub(r'\s+', ' ', text)
        
        # Başta ve sonda boşlukları temizle
        text = text.strip()
        
        return text
    
    def tokenize_text_improved(self, text):
        """Geliştirilmiş tokenization"""
        if not text:
            return []
        
        # Kelime bazında tokenization
        tokens = text.split()
        
        # Çok daha az stopwords filtreleme
        tokens = [token for token in tokens if token not in self.turkish_stopwords]
        
        # Sadece çok kısa tokenları çıkar (1 karakter)
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def extract_chords_improved(self, text):
        """Geliştirilmiş akor çıkarma"""
        if pd.isna(text):
            return []
        
        # Daha kapsamlı akor pattern'i
        chord_patterns = [
            r'\b[A-G][b#]?m?(?:7|9|11|13|6|sus|add|dim|aug|maj|min)?\b',  # Temel akorlar
            r'\b[A-G][b#]?m?(?:7|9|11|13|6|sus|add|dim|aug|maj|min)?/[A-G][b#]?\b',  # Slash chords
        ]
        
        chords = []
        for pattern in chord_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            chords.extend(matches)
        
        # Temizle ve normalize et
        chords = [chord.upper().strip() for chord in chords if chord.strip()]
        
        # Benzersiz akorları al ve sırala
        unique_chords = list(dict.fromkeys(chords))  # Sırayı koruyarak benzersiz yap
        
        return unique_chords
    
    def categorize_chords(self, chords):
        """Akorları kategorilere ayır"""
        categories = {
            'major': 0,
            'minor': 0,
            'seventh': 0,
            'complex': 0
        }
        
        for chord in chords:
            chord_upper = chord.upper()
            categorized = False
            
            for category, patterns in self.chord_categories.items():
                for pattern in patterns:
                    if pattern.upper() in chord_upper:
                        categories[category] += 1
                        categorized = True
                        break
                if categorized:
                    break
            
            # Eğer hiçbir kategoriye girmediyse, major olarak say
            if not categorized:
                categories['major'] += 1
        
        return categories
    
    def build_vocabulary_improved(self, texts, min_freq=2):
        """Geliştirilmiş kelime dağarcığı oluşturma"""
        print("Geliştirilmiş kelime dağarcığı oluşturuluyor...")
        
        all_tokens = []
        for text in texts:
            tokens = self.tokenize_text_improved(text)
            all_tokens.extend(tokens)
        
        # Kelime frekansları
        word_counts = Counter(all_tokens)
        
        # Minimum frekans filtresi
        filtered_words = {word: count for word, count in word_counts.items() if count >= min_freq}
        
        # Vocabulary oluştur
        self.word_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }
        
        idx = 4
        for word in sorted(filtered_words.keys()):
            self.word_to_idx[word] = idx
            idx += 1
        
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Kelime dağarcığı boyutu: {len(self.word_to_idx)}")
        print(f"En sık kullanılan 10 kelime: {word_counts.most_common(10)}")
        
        return self.word_to_idx
    
    def build_chord_vocabulary_improved(self, chord_lists):
        """Geliştirilmiş akor dağarcığı oluşturma"""
        print("Geliştirilmiş akor dağarcığı oluşturuluyor...")
        
        all_chords = []
        for chords in chord_lists:
            all_chords.extend(chords)
        
        chord_counts = Counter(all_chords)
        
        # Akor vocabulary oluştur
        self.chord_to_idx = {
            '<PAD>': 0,
            '<UNK>': 1
        }
        
        idx = 2
        for chord in sorted(chord_counts.keys()):
            self.chord_to_idx[chord] = idx
            idx += 1
        
        self.idx_to_chord = {idx: chord for chord, idx in self.chord_to_idx.items()}
        
        print(f"Akor dağarcığı boyutu: {len(self.chord_to_idx)}")
        print(f"En sık kullanılan 10 akor: {chord_counts.most_common(10)}")
        
        return self.chord_to_idx
    
    def text_to_sequence_improved(self, text, max_length=200):
        """Geliştirilmiş metin-to-sequence dönüşümü"""
        tokens = self.tokenize_text_improved(text)
        
        # Sequence oluştur
        sequence = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
        
        # Truncate veya pad
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence.extend([self.word_to_idx['<PAD>']] * (max_length - len(sequence)))
        
        return sequence
    
    def chords_to_sequence_improved(self, chords, max_length=30):
        """Geliştirilmiş akor-to-sequence dönüşümü"""
        sequence = [self.chord_to_idx.get(chord, self.chord_to_idx['<UNK>']) for chord in chords]
        
        # Truncate veya pad
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence.extend([self.chord_to_idx['<PAD>']] * (max_length - len(sequence)))
        
        return sequence
    
    def extract_chord_features_improved(self, chords):
        """Geliştirilmiş akor özellik çıkarma"""
        if not chords:
            return {
                'total_chords': 0,
                'unique_chords': 0,
                'major_count': 0,
                'minor_count': 0,
                'seventh_count': 0,
                'complex_count': 0,
                'chord_diversity': 0.0,
                'chord_progression_length': 0,
                'chord_complexity': 0.0
            }
        
        total_chords = len(chords)
        unique_chords = len(set(chords))
        diversity = unique_chords / total_chords if total_chords > 0 else 0.0
        
        # Kategori sayıları
        categories = self.categorize_chords(chords)
        
        # Akor karmaşıklığı (uzunluk bazında)
        complexity = sum(len(chord) for chord in chords) / total_chords if total_chords > 0 else 0.0
        
        return {
            'total_chords': total_chords,
            'unique_chords': unique_chords,
            'major_count': categories['major'],
            'minor_count': categories['minor'],
            'seventh_count': categories['seventh'],
            'complex_count': categories['complex'],
            'chord_diversity': diversity,
            'chord_progression_length': total_chords,
            'chord_complexity': complexity
        }
    
    def preprocess_dataset_improved(self, max_length=200, chord_max_length=30):
        """Geliştirilmiş dataset preprocessing"""
        print("Geliştirilmiş dataset preprocessing başlıyor...")
        
        # Metin temizleme
        print("Metinler temizleniyor...")
        self.data['cleaned_lyrics'] = self.data['lyrics'].apply(self.clean_text_improved)
        
        # Çok kısa metinleri filtrele
        self.data = self.data[self.data['cleaned_lyrics'].apply(lambda x: len(x.split()) > 5)]
        print(f"Temizleme sonrası: {len(self.data)} şarkı")
        
        # Akorları çıkar - chords kolonundan direkt al
        print("Akorlar çıkarılıyor...")
        # chords kolonu string olarak kaydedilmiş, eval ile parse et
        self.data['extracted_chords'] = self.data['chords'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
        # Kelime dağarcığı oluştur
        self.build_vocabulary_improved(self.data['cleaned_lyrics'])
        
        # Akor dağarcığı oluştur
        self.build_chord_vocabulary_improved(self.data['extracted_chords'])
        
        # Metinleri sequence'e çevir
        print("Metinler sequence'e çevriliyor...")
        self.data['sequence'] = self.data['cleaned_lyrics'].apply(
            lambda x: self.text_to_sequence_improved(x, max_length)
        )
        
        # Akorları sequence'e çevir
        print("Akorlar sequence'e çevriliyor...")
        self.data['chord_sequence'] = self.data['extracted_chords'].apply(
            lambda x: self.chords_to_sequence_improved(x, chord_max_length)
        )
        
        # Akor özelliklerini çıkar
        print("Akor özellikleri çıkarılıyor...")
        chord_features = self.data['extracted_chords'].apply(self.extract_chord_features_improved)
        
        # Akor özelliklerini ayrı kolonlara ayır
        for feature in chord_features.iloc[0].keys():
            self.data[f'chord_{feature}'] = chord_features.apply(lambda x: x[feature])
        
        # Genre'leri encode et
        print("Genre'ler encode ediliyor...")
        self.data['genre_encoded'] = self.genre_encoder.fit_transform(self.data['genre'])
        
        self._print_dataset_stats_improved()
        return self.data
    
    def _print_dataset_stats_improved(self):
        """Geliştirilmiş dataset istatistikleri"""
        print("\n" + "="*60)
        print("GELİŞTİRİLMİŞ DATASET İSTATİSTİKLERİ")
        print("="*60)
        print(f"Toplam şarkı sayısı: {len(self.data)}")
        print(f"Toplam kelime sayısı: {len(self.word_to_idx)}")
        print(f"Toplam akor sayısı: {len(self.chord_to_idx)}")
        
        print("\nGenre dağılımı:")
        genre_counts = self.data['genre'].value_counts()
        for genre, count in genre_counts.items():
            print(f"  {genre}: {count} şarkı")
        
        print("\nŞarkı uzunlukları (kelime):")
        word_counts = self.data['cleaned_lyrics'].apply(lambda x: len(x.split()))
        print(f"  Ortalama: {word_counts.mean():.1f} kelime")
        print(f"  Medyan: {word_counts.median():.1f} kelime")
        print(f"  Min: {word_counts.min()} kelime")
        print(f"  Max: {word_counts.max()} kelime")
        
        print("\nAkor istatistikleri:")
        print(f"  Ortalama akor sayısı: {self.data['chord_total_chords'].mean():.1f}")
        print(f"  Ortalama benzersiz akor: {self.data['chord_unique_chords'].mean():.1f}")
        print(f"  Ortalama akor çeşitliliği: {self.data['chord_chord_diversity'].mean():.3f}")
        print(f"  Ortalama akor karmaşıklığı: {self.data['chord_chord_complexity'].mean():.2f}")
        
        self._create_visualizations_improved()
    
    def _create_visualizations_improved(self):
        """Geliştirilmiş görselleştirmeler"""
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Genre dağılımı
        plt.figure(figsize=(12, 8))
        sns.countplot(y='genre', data=self.data, order=self.data['genre'].value_counts().index, palette='viridis')
        plt.title('Geliştirilmiş Genre Dağılımı', fontsize=16, fontweight='bold')
        plt.xlabel('Şarkı Sayısı', fontsize=12)
        plt.ylabel('Genre', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'improved_genre_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Kelime sayısı dağılımı
        plt.figure(figsize=(12, 8))
        word_counts = self.data['cleaned_lyrics'].apply(lambda x: len(x.split()))
        sns.histplot(word_counts, bins=50, kde=True, color='skyblue')
        plt.title('Geliştirilmiş Kelime Sayısı Dağılımı', fontsize=16, fontweight='bold')
        plt.xlabel('Kelime Sayısı', fontsize=12)
        plt.ylabel('Şarkı Sayısı', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'improved_word_count_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Akor özellikleri
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        sns.histplot(self.data['chord_total_chords'], bins=30, kde=True, color='lightcoral')
        plt.title('Toplam Akor Sayısı')
        
        plt.subplot(2, 3, 2)
        sns.histplot(self.data['chord_unique_chords'], bins=30, kde=True, color='lightgreen')
        plt.title('Benzersiz Akor Sayısı')
        
        plt.subplot(2, 3, 3)
        sns.histplot(self.data['chord_chord_diversity'], bins=30, kde=True, color='lightblue')
        plt.title('Akor Çeşitliliği')
        
        plt.subplot(2, 3, 4)
        sns.histplot(self.data['chord_major_count'], bins=20, kde=True, color='gold')
        plt.title('Major Akor Sayısı')
        
        plt.subplot(2, 3, 5)
        sns.histplot(self.data['chord_minor_count'], bins=20, kde=True, color='purple')
        plt.title('Minor Akor Sayısı')
        
        plt.subplot(2, 3, 6)
        sns.histplot(self.data['chord_chord_complexity'], bins=20, kde=True, color='orange')
        plt.title('Akor Karmaşıklığı')
        
        plt.suptitle('Geliştirilmiş Akor Özellikleri Dağılımı', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'improved_chord_features_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Geliştirilmiş görselleştirmeler kaydedildi: {results_dir}/")
    
    def split_data(self, test_size=0.2, val_size=0.1):
        """Veriyi train, validation ve test setlerine böl"""
        train_val_df, test_df = train_test_split(
            self.data, test_size=test_size, random_state=42, stratify=self.data['genre']
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), random_state=42, stratify=train_val_df['genre']
        )
        
        return train_df, val_df, test_df
    
    def save_processed_data_improved(self, train_df, val_df, test_df, output_dir="data/processed"):
        """Geliştirilmiş işlenmiş veriyi kaydet"""
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV dosyalarını kaydet
        train_df.to_csv(os.path.join(output_dir, "train_improved.csv"), index=False, encoding='utf-8')
        val_df.to_csv(os.path.join(output_dir, "validation_improved.csv"), index=False, encoding='utf-8')
        test_df.to_csv(os.path.join(output_dir, "test_improved.csv"), index=False, encoding='utf-8')
        
        # Vocabulary dosyalarını kaydet
        vocab_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'chord_to_idx': self.chord_to_idx,
            'idx_to_chord': self.idx_to_chord,
            'genre_classes': self.genre_encoder.classes_.tolist()
        }
        
        with open(os.path.join(output_dir, "vocabulary_improved.json"), 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nGeliştirilmiş işlenmiş veri kaydedildi:")
        print(f"  Train: {os.path.join(output_dir, 'train_improved.csv')}")
        print(f"  Validation: {os.path.join(output_dir, 'validation_improved.csv')}")
        print(f"  Test: {os.path.join(output_dir, 'test_improved.csv')}")
        print(f"  Vocabulary: {os.path.join(output_dir, 'vocabulary_improved.json')}")

def main():
    print("GELİŞTİRİLMİŞ TÜRKÇE ŞARKI SÖZÜ PREPROCESSING")
    print("="*70)
    
    preprocessor = ImprovedTurkishLyricsPreprocessor()
    
    # Veriyi yükle
    data_path = "data/raw/turkish_lyrics_enhanced_dataset.csv"
    preprocessor.load_data(data_path)
    
    # Preprocessing yap
    processed_data = preprocessor.preprocess_dataset_improved(max_length=200, chord_max_length=30)
    
    # Veriyi böl
    train_df, val_df, test_df = preprocessor.split_data()
    
    # Kaydet
    preprocessor.save_processed_data_improved(train_df, val_df, test_df)
    
    print("\n" + "="*70)
    print("GELİŞTİRİLMİŞ PREPROCESSING TAMAMLANDI!")
    print("Artık Transformer modeli daha iyi performans gösterebilir.")

if __name__ == "__main__":
    main()

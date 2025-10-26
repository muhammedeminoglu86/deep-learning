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
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class AdvancedTurkishLyricsPreprocessor:
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
        
        # Gelişmiş akor kategorileri
        self.chord_categories = {
            'major': ['C', 'D', 'E', 'F', 'G', 'A', 'B', 'C#', 'D#', 'F#', 'G#', 'A#', 'Cb', 'Db', 'Eb', 'Gb', 'Ab', 'Bb'],
            'minor': ['Cm', 'Dm', 'Em', 'Fm', 'Gm', 'Am', 'Bm', 'C#m', 'D#m', 'F#m', 'G#m', 'A#m', 'Cbm', 'Dbm', 'Ebm', 'Gbm', 'Abm', 'Bbm'],
            'seventh': ['C7', 'D7', 'E7', 'F7', 'G7', 'A7', 'B7', 'C#7', 'D#7', 'F#7', 'G#7', 'A#7'],
            'complex': ['sus', 'add', 'dim', 'aug', 'maj', 'min', 'm7', 'maj7', 'dim7', 'aug7']
        }
        
        # Müzik terimleri ve duygusal kelimeler
        self.music_terms = {
            'aşk', 'sevgi', 'kalp', 'gönül', 'hüzün', 'mutluluk', 'acı', 'gözyaşı',
            'gece', 'gündüz', 'yıldız', 'ay', 'güneş', 'deniz', 'dağ', 'orman',
            'kuş', 'çiçek', 'gül', 'bülbül', 'rüzgar', 'yağmur', 'kar', 'mevsim'
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
    
    def clean_text_advanced(self, text):
        """Gelişmiş metin temizleme"""
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
    
    def tokenize_text_advanced(self, text):
        """Gelişmiş tokenization"""
        if not text:
            return []
        
        # Kelime bazında tokenization
        tokens = text.split()
        
        # Çok daha az stopwords filtreleme
        tokens = [token for token in tokens if token not in self.turkish_stopwords]
        
        # Sadece çok kısa tokenları çıkar (1 karakter)
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def extract_chords_advanced(self, text):
        """Gelişmiş akor çıkarma"""
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
    
    def categorize_chords_advanced(self, chords):
        """Gelişmiş akor kategorilere ayırma"""
        categories = {
            'major': 0,
            'minor': 0,
            'seventh': 0,
            'complex': 0,
            'diminished': 0,
            'augmented': 0,
            'suspended': 0
        }
        
        for chord in chords:
            chord_upper = chord.upper()
            categorized = False
            
            # Diminished chords
            if 'dim' in chord_upper or 'o' in chord_upper:
                categories['diminished'] += 1
                categorized = True
            
            # Augmented chords
            elif 'aug' in chord_upper or '+' in chord_upper:
                categories['augmented'] += 1
                categorized = True
            
            # Suspended chords
            elif 'sus' in chord_upper:
                categories['suspended'] += 1
                categorized = True
            
            # Other categories
            else:
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
    
    def extract_emotional_features(self, text):
        """Duygusal özellikler çıkar"""
        if not text:
            return {
                'positive_words': 0,
                'negative_words': 0,
                'love_words': 0,
                'sadness_words': 0,
                'music_words': 0,
                'emotional_intensity': 0.0
            }
        
        tokens = self.tokenize_text_advanced(text)
        
        # Duygusal kelime listeleri
        positive_words = ['mutlu', 'sevinç', 'gülümseme', 'neşe', 'keyif', 'huzur', 'barış', 'umut']
        negative_words = ['üzüntü', 'keder', 'acı', 'hüzün', 'korku', 'endişe', 'kızgın', 'öfke']
        love_words = ['aşk', 'sevgi', 'kalp', 'gönül', 'sevgili', 'yâr', 'dost', 'arkadaş']
        sadness_words = ['gözyaşı', 'ağlamak', 'hüzün', 'keder', 'acı', 'yas', 'matem']
        music_words = ['şarkı', 'müzik', 'melodi', 'ritim', 'ses', 'nota', 'enstrüman']
        
        # Sayıları hesapla
        positive_count = sum(1 for token in tokens if any(word in token for word in positive_words))
        negative_count = sum(1 for token in tokens if any(word in token for word in negative_words))
        love_count = sum(1 for token in tokens if any(word in token for word in love_words))
        sadness_count = sum(1 for token in tokens if any(word in token for word in sadness_words))
        music_count = sum(1 for token in tokens if any(word in token for word in music_words))
        
        # Duygusal yoğunluk
        emotional_intensity = (positive_count + negative_count + love_count + sadness_count) / len(tokens) if tokens else 0.0
        
        return {
            'positive_words': positive_count,
            'negative_words': negative_count,
            'love_words': love_count,
            'sadness_words': sadness_count,
            'music_words': music_count,
            'emotional_intensity': emotional_intensity
        }
    
    def extract_rhythm_features(self, text):
        """Ritim özellikleri çıkar"""
        if not text:
            return {
                'avg_line_length': 0.0,
                'line_count': 0,
                'rhythm_variation': 0.0,
                'repetition_ratio': 0.0
            }
        
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            return {
                'avg_line_length': 0.0,
                'line_count': 0,
                'rhythm_variation': 0.0,
                'repetition_ratio': 0.0
            }
        
        line_lengths = [len(line.split()) for line in lines]
        avg_line_length = np.mean(line_lengths)
        line_count = len(lines)
        
        # Ritim varyasyonu (satır uzunluklarının standart sapması)
        rhythm_variation = np.std(line_lengths) if len(line_lengths) > 1 else 0.0
        
        # Tekrar oranı (aynı satırların oranı)
        unique_lines = len(set(lines))
        repetition_ratio = 1.0 - (unique_lines / line_count) if line_count > 0 else 0.0
        
        return {
            'avg_line_length': avg_line_length,
            'line_count': line_count,
            'rhythm_variation': rhythm_variation,
            'repetition_ratio': repetition_ratio
        }
    
    def build_vocabulary_advanced(self, texts, min_freq=2):
        """Gelişmiş kelime dağarcığı oluşturma"""
        print("Gelişmiş kelime dağarcığı oluşturuluyor...")
        
        all_tokens = []
        for text in texts:
            tokens = self.tokenize_text_advanced(text)
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
    
    def build_chord_vocabulary_advanced(self, chord_lists):
        """Gelişmiş akor dağarcığı oluşturma"""
        print("Gelişmiş akor dağarcığı oluşturuluyor...")
        
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
    
    def text_to_sequence_advanced(self, text, max_length=300):
        """Gelişmiş metin-to-sequence dönüşümü"""
        tokens = self.tokenize_text_advanced(text)
        
        # Sequence oluştur
        sequence = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
        
        # Truncate veya pad
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence.extend([self.word_to_idx['<PAD>']] * (max_length - len(sequence)))
        
        return sequence
    
    def chords_to_sequence_advanced(self, chords, max_length=50):
        """Gelişmiş akor-to-sequence dönüşümü"""
        sequence = [self.chord_to_idx.get(chord, self.chord_to_idx['<UNK>']) for chord in chords]
        
        # Truncate veya pad
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        else:
            sequence.extend([self.chord_to_idx['<PAD>']] * (max_length - len(sequence)))
        
        return sequence
    
    def extract_chord_features_advanced(self, chords):
        """Gelişmiş akor özellik çıkarma"""
        if not chords:
            return {
                'total_chords': 0,
                'unique_chords': 0,
                'major_count': 0,
                'minor_count': 0,
                'seventh_count': 0,
                'complex_count': 0,
                'diminished_count': 0,
                'augmented_count': 0,
                'suspended_count': 0,
                'chord_diversity': 0.0,
                'chord_progression_length': 0,
                'chord_complexity': 0.0,
                'chord_transitions': 0,
                'key_signature': 'unknown'
            }
        
        total_chords = len(chords)
        unique_chords = len(set(chords))
        diversity = unique_chords / total_chords if total_chords > 0 else 0.0
        
        # Kategori sayıları
        categories = self.categorize_chords_advanced(chords)
        
        # Akor karmaşıklığı (uzunluk bazında)
        complexity = sum(len(chord) for chord in chords) / total_chords if total_chords > 0 else 0.0
        
        # Akor geçişleri (farklı akorlar arasındaki geçiş sayısı)
        transitions = sum(1 for i in range(len(chords)-1) if chords[i] != chords[i+1])
        
        # Anahtar imza tahmini (en sık kullanılan akor)
        key_signature = Counter(chords).most_common(1)[0][0] if chords else 'unknown'
        
        return {
            'total_chords': total_chords,
            'unique_chords': unique_chords,
            'major_count': categories['major'],
            'minor_count': categories['minor'],
            'seventh_count': categories['seventh'],
            'complex_count': categories['complex'],
            'diminished_count': categories['diminished'],
            'augmented_count': categories['augmented'],
            'suspended_count': categories['suspended'],
            'chord_diversity': diversity,
            'chord_progression_length': total_chords,
            'chord_complexity': complexity,
            'chord_transitions': transitions,
            'key_signature': key_signature
        }
    
    def preprocess_dataset_advanced(self, max_length=300, chord_max_length=50):
        """Gelişmiş dataset preprocessing"""
        print("Gelişmiş dataset preprocessing başlıyor...")
        
        # Metin temizleme
        print("Metinler temizleniyor...")
        self.data['cleaned_lyrics'] = self.data['lyrics'].apply(self.clean_text_advanced)
        
        # Çok kısa metinleri filtrele
        self.data = self.data[self.data['cleaned_lyrics'].apply(lambda x: len(x.split()) > 5)]
        print(f"Temizleme sonrası: {len(self.data)} şarkı")
        
        # Akorları çıkar - chords kolonundan direkt al
        print("Akorlar çıkarılıyor...")
        self.data['extracted_chords'] = self.data['chords'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
        # Kelime dağarcığı oluştur
        self.build_vocabulary_advanced(self.data['cleaned_lyrics'])
        
        # Akor dağarcığı oluştur
        self.build_chord_vocabulary_advanced(self.data['extracted_chords'])
        
        # Metinleri sequence'e çevir
        print("Metinler sequence'e çevriliyor...")
        self.data['sequence'] = self.data['cleaned_lyrics'].apply(
            lambda x: self.text_to_sequence_advanced(x, max_length)
        )
        
        # Akorları sequence'e çevir
        print("Akorlar sequence'e çevriliyor...")
        self.data['chord_sequence'] = self.data['extracted_chords'].apply(
            lambda x: self.chords_to_sequence_advanced(x, chord_max_length)
        )
        
        # Akor özelliklerini çıkar
        print("Akor özellikleri çıkarılıyor...")
        chord_features = self.data['extracted_chords'].apply(self.extract_chord_features_advanced)
        
        # Akor özelliklerini ayrı kolonlara ayır
        for feature in chord_features.iloc[0].keys():
            if feature != 'key_signature':  # String olan key_signature'ı ayrı işle
                self.data[f'chord_{feature}'] = chord_features.apply(lambda x: x[feature])
        
        # Key signature'ı encode et
        key_signatures = chord_features.apply(lambda x: x['key_signature'])
        key_encoder = LabelEncoder()
        self.data['chord_key_signature_encoded'] = key_encoder.fit_transform(key_signatures)
        
        # Duygusal özellikleri çıkar
        print("Duygusal özellikler çıkarılıyor...")
        emotional_features = self.data['cleaned_lyrics'].apply(self.extract_emotional_features)
        
        for feature in emotional_features.iloc[0].keys():
            self.data[f'emotional_{feature}'] = emotional_features.apply(lambda x: x[feature])
        
        # Ritim özelliklerini çıkar
        print("Ritim özellikleri çıkarılıyor...")
        rhythm_features = self.data['cleaned_lyrics'].apply(self.extract_rhythm_features)
        
        for feature in rhythm_features.iloc[0].keys():
            self.data[f'rhythm_{feature}'] = rhythm_features.apply(lambda x: x[feature])
        
        # Genre'leri encode et
        print("Genre'ler encode ediliyor...")
        self.data['genre_encoded'] = self.genre_encoder.fit_transform(self.data['genre'])
        
        self._print_dataset_stats_advanced()
        return self.data
    
    def _print_dataset_stats_advanced(self):
        """Gelişmiş dataset istatistikleri"""
        print("\n" + "="*70)
        print("GELİŞMİŞ DATASET İSTATİSTİKLERİ")
        print("="*70)
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
        print(f"  Ortalama akor geçişi: {self.data['chord_chord_transitions'].mean():.1f}")
        
        print("\nDuygusal özellikler:")
        print(f"  Ortalama pozitif kelime: {self.data['emotional_positive_words'].mean():.1f}")
        print(f"  Ortalama negatif kelime: {self.data['emotional_negative_words'].mean():.1f}")
        print(f"  Ortalama aşk kelimesi: {self.data['emotional_love_words'].mean():.1f}")
        print(f"  Ortalama duygusal yoğunluk: {self.data['emotional_emotional_intensity'].mean():.3f}")
        
        print("\nRitim özellikleri:")
        print(f"  Ortalama satır uzunluğu: {self.data['rhythm_avg_line_length'].mean():.1f}")
        print(f"  Ortalama satır sayısı: {self.data['rhythm_line_count'].mean():.1f}")
        print(f"  Ortalama ritim varyasyonu: {self.data['rhythm_rhythm_variation'].mean():.2f}")
        print(f"  Ortalama tekrar oranı: {self.data['rhythm_repetition_ratio'].mean():.3f}")
        
        self._create_visualizations_advanced()
    
    def _create_visualizations_advanced(self):
        """Gelişmiş görselleştirmeler"""
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Genre dağılımı
        plt.figure(figsize=(12, 8))
        sns.countplot(y='genre', data=self.data, order=self.data['genre'].value_counts().index, palette='viridis')
        plt.title('Gelişmiş Genre Dağılımı', fontsize=16, fontweight='bold')
        plt.xlabel('Şarkı Sayısı', fontsize=12)
        plt.ylabel('Genre', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'advanced_genre_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Duygusal özellikler
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        sns.histplot(self.data['emotional_positive_words'], bins=20, kde=True, color='lightgreen')
        plt.title('Pozitif Kelime Dağılımı')
        
        plt.subplot(2, 3, 2)
        sns.histplot(self.data['emotional_negative_words'], bins=20, kde=True, color='lightcoral')
        plt.title('Negatif Kelime Dağılımı')
        
        plt.subplot(2, 3, 3)
        sns.histplot(self.data['emotional_love_words'], bins=20, kde=True, color='pink')
        plt.title('Aşk Kelimesi Dağılımı')
        
        plt.subplot(2, 3, 4)
        sns.histplot(self.data['emotional_emotional_intensity'], bins=20, kde=True, color='purple')
        plt.title('Duygusal Yoğunluk')
        
        plt.subplot(2, 3, 5)
        sns.histplot(self.data['rhythm_avg_line_length'], bins=20, kde=True, color='orange')
        plt.title('Ortalama Satır Uzunluğu')
        
        plt.subplot(2, 3, 6)
        sns.histplot(self.data['rhythm_repetition_ratio'], bins=20, kde=True, color='brown')
        plt.title('Tekrar Oranı')
        
        plt.suptitle('Gelişmiş Özellik Dağılımları', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'advanced_features_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gelişmiş görselleştirmeler kaydedildi: {results_dir}/")
    
    def split_data(self, test_size=0.2, val_size=0.1):
        """Veriyi train, validation ve test setlerine böl"""
        train_val_df, test_df = train_test_split(
            self.data, test_size=test_size, random_state=42, stratify=self.data['genre']
        )
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size/(1-test_size), random_state=42, stratify=train_val_df['genre']
        )
        
        return train_df, val_df, test_df
    
    def save_processed_data_advanced(self, train_df, val_df, test_df, output_dir="data/processed"):
        """Gelişmiş işlenmiş veriyi kaydet"""
        os.makedirs(output_dir, exist_ok=True)
        
        # CSV dosyalarını kaydet
        train_df.to_csv(os.path.join(output_dir, "train_advanced.csv"), index=False, encoding='utf-8')
        val_df.to_csv(os.path.join(output_dir, "validation_advanced.csv"), index=False, encoding='utf-8')
        test_df.to_csv(os.path.join(output_dir, "test_advanced.csv"), index=False, encoding='utf-8')
        
        # Vocabulary dosyalarını kaydet
        vocab_data = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'chord_to_idx': self.chord_to_idx,
            'idx_to_chord': self.idx_to_chord,
            'genre_classes': self.genre_encoder.classes_.tolist()
        }
        
        with open(os.path.join(output_dir, "vocabulary_advanced.json"), 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nGelişmiş işlenmiş veri kaydedildi:")
        print(f"  Train: {os.path.join(output_dir, 'train_advanced.csv')}")
        print(f"  Validation: {os.path.join(output_dir, 'validation_advanced.csv')}")
        print(f"  Test: {os.path.join(output_dir, 'test_advanced.csv')}")
        print(f"  Vocabulary: {os.path.join(output_dir, 'vocabulary_advanced.json')}")

def main():
    print("GELİŞMİŞ TÜRKÇE ŞARKI SÖZÜ PREPROCESSING")
    print("="*70)
    
    preprocessor = AdvancedTurkishLyricsPreprocessor()
    
    # Veriyi yükle
    data_path = "data/raw/turkish_lyrics_enhanced_dataset.csv"
    preprocessor.load_data(data_path)
    
    # Preprocessing yap
    processed_data = preprocessor.preprocess_dataset_advanced(max_length=300, chord_max_length=50)
    
    # Veriyi böl
    train_df, val_df, test_df = preprocessor.split_data()
    
    # Kaydet
    preprocessor.save_processed_data_advanced(train_df, val_df, test_df)
    
    print("\n" + "="*70)
    print("GELİŞMİŞ PREPROCESSING TAMAMLANDI!")
    print("Artık daha yüksek performanslı modeller eğitilebilir.")

if __name__ == "__main__":
    main()

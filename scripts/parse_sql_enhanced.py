import re
import json
import pandas as pd
import os
from datetime import datetime
import numpy as np
from collections import Counter

def parse_sql_file_with_chords(sql_file_path):
    """SQL dosyasını parse edip şarkı verilerini ve akorları çıkar"""
    print(f"SQL dosyası parse ediliyor (akorlar dahil): {sql_file_path}")
    
    with open(sql_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Sanatçıları parse et
    print("Sanatçılar parse ediliyor...")
    artists = {}
    
    # Sanatçı INSERT statement'ını bul
    artists_start = content.find("INSERT INTO `sanatcilar`")
    if artists_start != -1:
        artists_end = content.find(";", artists_start)
        artists_section = content[artists_start:artists_end]
        
        # VALUES kısmını bul
        values_start = artists_section.find("VALUES")
        if values_start != -1:
            values_section = artists_section[values_start + 6:]  # "VALUES" kelimesini atla
            
            # Her INSERT satırını parse et
            lines = values_section.split('\n')
            for line in lines:
                line = line.strip()
                if line and '(' in line and ')' in line:
                    # (id, 'name') formatını parse et
                    match = re.search(r'\((\d+),\s*\'([^\']+)\'\s*\)', line)
                    if match:
                        artist_id = int(match.group(1))
                        artist_name = match.group(2).strip()
                        artists[artist_id] = artist_name
    
    print(f"{len(artists)} sanatçı bulundu")
    
    # Şarkıları parse et
    print("Şarkılar parse ediliyor...")
    songs = []
    
    # Şarkı INSERT statement'larını bul
    songs_start = content.find("INSERT INTO `sarkilar`")
    if songs_start != -1:
        # Tüm INSERT statement'larını bul
        insert_pattern = r"INSERT INTO `sarkilar`.*?VALUES\s*(.*?);"
        insert_matches = re.findall(insert_pattern, content[songs_start:], re.DOTALL)
        
        for insert_match in insert_matches:
            # Her INSERT statement'ındaki satırları parse et
            lines = insert_match.split('\n')
            for line in lines:
                line = line.strip()
                if line and '(' in line and ')' in line:
                    # Çok satırlı şarkı sözleri için daha esnek parsing
                    match = re.search(r'\((\d+),\s*(\d+),\s*\'([^\']+)\',\s*\'(.*?)\'\s*\)', line, re.DOTALL)
                    if match:
                        song_id = int(match.group(1))
                        artist_id = int(match.group(2))
                        song_title = match.group(3).strip()
                        song_content = match.group(4).strip()
                        
                        # Artist adını al
                        artist_name = artists.get(artist_id, f"Unknown_{artist_id}")
                        
                        # Şarkı sözlerini ve akorları ayır
                        lyrics, chords = extract_lyrics_and_chords(song_content)
                        
                        if len(lyrics) > 20:  # Çok kısa şarkıları atla
                            songs.append({
                                'song_id': song_id,
                                'title': song_title,
                                'artist': artist_name,
                                'lyrics': lyrics,
                                'chords': chords,
                                'chord_sequence': extract_chord_sequence(chords),
                                'chord_progression': analyze_chord_progression(chords),
                                'genre': determine_genre_with_chords(song_title, artist_name, lyrics, chords),
                                'year': 'Unknown',
                                'source': 'turkce_akorlar.sql',
                                'scraped_at': datetime.now().isoformat()
                            })
    
    print(f"{len(songs)} şarkı parse edildi")
    return songs

def extract_lyrics_and_chords(content):
    """Şarkı içeriğinden sözleri ve akorları ayır"""
    # Akor pattern'leri
    chord_patterns = [
        r'\b[A-G][#b]?m?[0-9]*\b',  # Temel akorlar (Am, C, Dm, F, G, Em, B7, etc.)
        r'\b[A-G][#b]?maj?\b',      # Major akorlar
        r'\b[A-G][#b]?dim\b',       # Diminished akorlar
        r'\b[A-G][#b]?aug\b',       # Augmented akorlar
        r'\b[A-G][#b]?sus[24]\b',   # Suspended akorlar
        r'\b[A-G][#b]?add[0-9]\b',  # Added note akorlar
    ]
    
    # Akorları bul
    chords = []
    for pattern in chord_patterns:
        found_chords = re.findall(pattern, content, re.IGNORECASE)
        chords.extend(found_chords)
    
    # Akorları temizle ve normalize et
    chords = [chord.upper() for chord in chords]
    chords = list(set(chords))  # Tekrarları kaldır
    
    # Şarkı sözlerini temizle
    lyrics = content
    
    # Akorları çıkar
    for pattern in chord_patterns:
        lyrics = re.sub(pattern, '', lyrics, flags=re.IGNORECASE)
    
    # Teknik notları temizle
    lyrics = re.sub(r'\([^)]*\)', '', lyrics)  # Parantez içi
    lyrics = re.sub(r'\[[^\]]*\]', '', lyrics)  # Köşeli parantez içi
    
    # Tab notasyonlarını temizle
    lyrics = re.sub(r'e-+\|.*?\|', '', lyrics, flags=re.DOTALL)
    lyrics = re.sub(r'h-+\|.*?\|', '', lyrics, flags=re.DOTALL)
    lyrics = re.sub(r'g-+\|.*?\|', '', lyrics, flags=re.DOTALL)
    lyrics = re.sub(r'D-+\|.*?\|', '', lyrics, flags=re.DOTALL)
    lyrics = re.sub(r'A-+\|.*?\|', '', lyrics, flags=re.DOTALL)
    lyrics = re.sub(r'E-+\|.*?\|', '', lyrics, flags=re.DOTALL)
    
    # Tekrarları temizle
    lyrics = re.sub(r'\)\s*\d+\s*[Tt]ekrar', '', lyrics)
    lyrics = re.sub(r'\)\s*\d+\s*[Kk]ere', '', lyrics)
    lyrics = re.sub(r'\)\s*\d+', '', lyrics)
    
    # Fazla boşlukları temizle
    lyrics = re.sub(r'\s+', ' ', lyrics)
    lyrics = re.sub(r'\n\s*\n', '\n', lyrics)
    
    # Başta ve sonda temizle
    lyrics = lyrics.strip()
    
    return lyrics, chords

def extract_chord_sequence(chords):
    """Akor dizisini çıkar"""
    if not chords:
        return []
    
    # Akorları sırala (müzik teorisi açısından)
    chord_order = ['C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab', 'A', 'A#', 'Bb', 'B']
    
    def chord_sort_key(chord):
        # Akorun root notunu bul
        root = chord[0]
        if len(chord) > 1 and chord[1] in ['#', 'b']:
            root = chord[:2]
        
        try:
            return chord_order.index(root)
        except ValueError:
            return 999  # Bilinmeyen akorlar sona
    
    return sorted(chords, key=chord_sort_key)

def analyze_chord_progression(chords):
    """Akor progresyonunu analiz et"""
    if not chords:
        return {}
    
    analysis = {
        'total_chords': len(chords),
        'unique_chords': len(set(chords)),
        'major_chords': len([c for c in chords if 'm' not in c and 'dim' not in c and 'aug' not in c]),
        'minor_chords': len([c for c in chords if 'm' in c and 'maj' not in c]),
        'seventh_chords': len([c for c in chords if '7' in c]),
        'complex_chords': len([c for c in chords if any(x in c for x in ['sus', 'add', 'dim', 'aug'])])
    }
    
    # En sık kullanılan akorlar
    chord_counts = Counter(chords)
    analysis['most_common_chords'] = chord_counts.most_common(5)
    
    return analysis

def determine_genre_with_chords(title, artist, lyrics, chords):
    """Akorları da dahil ederek genre belirleme"""
    text = (title + " " + artist + " " + lyrics).lower()
    
    # Akor analizi
    chord_analysis = analyze_chord_progression(chords)
    
    genre_keywords = {
        'arabesk': {
            'text': ['arabesk', 'acı', 'hüzün', 'aşk', 'müslüm', 'gürses', 'ferdi', 'tayfur', 'ibrahim', 'tatlıses'],
            'chords': ['Am', 'Dm', 'Em', 'F', 'G'],  # Minör tonlar
            'weight': 1.0
        },
        'pop': {
            'text': ['pop', 'şarkı', 'hit', 'melodi', 'tarkan', 'sezen', 'ajda', 'deniz', 'seki', 'sertab'],
            'chords': ['C', 'F', 'G', 'Am'],  # Basit akorlar
            'weight': 1.0
        },
        'rock': {
            'text': ['rock', 'gitar', 'grup', 'metal', 'barış', 'manço', 'mfo', 'cem', 'karaca', 'duman'],
            'chords': ['E', 'A', 'D', 'G', 'B'],  # Rock akorları
            'weight': 1.0
        },
        'rap': {
            'text': ['rap', 'hip hop', 'mc', 'beat', 'ceza', 'sagopa', 'kolera', 'ezhel', 'patron'],
            'chords': ['F', 'Bb', 'Eb', 'Ab'],  # Hip-hop akorları
            'weight': 0.8  # Rap'te akorlar daha az önemli
        },
        'folk': {
            'text': ['folk', 'halk', 'türkü', 'aşık', 'veysel', 'mahzuni', 'neşet', 'ertaş', 'anonim'],
            'chords': ['D', 'G', 'A', 'Em'],  # Folk akorları
            'weight': 1.2  # Folk'ta akorlar çok önemli
        },
        'slow': {
            'text': ['slow', 'ballad', 'romantik', 'aşk', 'tatlıses', 'ibrahim', 'rafet', 'elmas'],
            'chords': ['Am', 'F', 'C', 'G'],  # Ballad akorları
            'weight': 1.1
        }
    }
    
    genre_scores = {}
    
    for genre, data in genre_keywords.items():
        # Metin skoru
        text_score = sum(1 for keyword in data['text'] if keyword in text)
        
        # Akor skoru
        chord_score = 0
        if chords:
            chord_score = sum(1 for chord in chords if chord in data['chords'])
            chord_score = chord_score / len(chords) * 10  # Normalize et
        
        # Toplam skor
        total_score = (text_score + chord_score) * data['weight']
        genre_scores[genre] = total_score
    
    if genre_scores:
        best = max(genre_scores, key=genre_scores.get)
        if genre_scores[best] > 0:
            return best
    
    # Default olarak arabesk (Türk müziğinde yaygın)
    return 'arabesk'

def save_enhanced_data(songs, output_dir="data/raw"):
    """Gelişmiş veriyi kaydet"""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON formatında kaydet
    json_path = f"{output_dir}/turkish_lyrics_enhanced_dataset.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(songs, f, ensure_ascii=False, indent=2)
    
    # CSV formatında kaydet
    df = pd.DataFrame(songs)
    csv_path = f"{output_dir}/turkish_lyrics_enhanced_dataset.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"\nGelişmiş veri kaydedildi:")
    print(f"   JSON: {json_path}")
    print(f"   CSV: {csv_path}")
    print(f"   Toplam şarkı: {len(songs)}")
    
    # Genre dağılımı
    if songs:
        genre_counts = df['genre'].value_counts()
        print(f"   Genre dağılımı:")
        for genre, count in genre_counts.items():
            print(f"      {genre}: {count} şarkı")
    
    # Akor istatistikleri
    all_chords = []
    for song in songs:
        all_chords.extend(song['chords'])
    
    chord_counts = Counter(all_chords)
    print(f"   En sık kullanılan akorlar:")
    for chord, count in chord_counts.most_common(10):
        print(f"      {chord}: {count} kez")
    
    return songs

def main():
    """Ana fonksiyon"""
    print("Gelişmiş Türkçe Şarkı Verisi Parse Ediliyor (Akorlar Dahil)...")
    print("="*70)
    
    # SQL dosyasını parse et
    sql_file = "data/raw/turkce_akorlar.sql"
    songs = parse_sql_file_with_chords(sql_file)
    
    # Veriyi kaydet
    save_enhanced_data(songs)
    
    print("\n" + "="*70)
    print("Gelişmiş veri parse işlemi tamamlandı!")
    print("Artık akorları da içeren preprocessing'e geçebiliriz.")

if __name__ == "__main__":
    main()


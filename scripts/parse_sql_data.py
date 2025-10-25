import re
import json
import pandas as pd
import os
from datetime import datetime

def parse_sql_file(sql_file_path):
    """SQL dosyasını parse edip şarkı verilerini çıkar"""
    print(f"SQL dosyası parse ediliyor: {sql_file_path}")
    
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
                    # Önce temel pattern'i dene
                    match = re.search(r'\((\d+),\s*(\d+),\s*\'([^\']+)\',\s*\'(.*?)\'\s*\)', line, re.DOTALL)
                    if match:
                        song_id = int(match.group(1))
                        artist_id = int(match.group(2))
                        song_title = match.group(3).strip()
                        song_lyrics = match.group(4).strip()
                        
                        # Artist adını al
                        artist_name = artists.get(artist_id, f"Unknown_{artist_id}")
                        
                        # Şarkı sözlerini temizle
                        cleaned_lyrics = clean_lyrics(song_lyrics)
                        
                        if len(cleaned_lyrics) > 20:  # Çok kısa şarkıları atla
                            songs.append({
                                'song_id': song_id,
                                'title': song_title,
                                'artist': artist_name,
                                'lyrics': cleaned_lyrics,
                                'genre': determine_genre(song_title, artist_name, cleaned_lyrics),
                                'year': 'Unknown',
                                'source': 'turkce_akorlar.sql',
                                'scraped_at': datetime.now().isoformat()
                            })
    
    print(f"{len(songs)} şarkı parse edildi")
    return songs

def clean_lyrics(lyrics):
    """Şarkı sözlerini temizle"""
    # Akorları ve teknik notları temizle
    # Akor pattern'leri (Am, C, Dm, F, G, Em, B7, etc.)
    lyrics = re.sub(r'\b[A-G][#b]?m?[0-9]*\b', '', lyrics)
    
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
    
    return lyrics

def determine_genre(title, artist, lyrics):
    """Genre belirleme"""
    text = (title + " " + artist + " " + lyrics).lower()
    
    genre_keywords = {
        'arabesk': ['arabesk', 'acı', 'hüzün', 'aşk', 'müslüm', 'gürses', 'ferdi', 'tayfur', 'ibrahim', 'tatlıses'],
        'pop': ['pop', 'şarkı', 'hit', 'melodi', 'tarkan', 'sezen', 'ajda', 'deniz', 'seki', 'sertab'],
        'rock': ['rock', 'gitar', 'grup', 'metal', 'barış', 'manço', 'mfo', 'cem', 'karaca', 'duman'],
        'rap': ['rap', 'hip hop', 'mc', 'beat', 'ceza', 'sagopa', 'kolera', 'ezhel', 'patron'],
        'folk': ['folk', 'halk', 'türkü', 'aşık', 'veysel', 'mahzuni', 'neşet', 'ertaş', 'anonim'],
        'slow': ['slow', 'ballad', 'romantik', 'aşk', 'tatlıses', 'ibrahim', 'rafet', 'elmas']
    }
    
    genre_scores = {}
    for genre, keywords in genre_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text)
        genre_scores[genre] = score
    
    if genre_scores:
        best = max(genre_scores, key=genre_scores.get)
        if genre_scores[best] > 0:
            return best
    
    # Default olarak arabesk (Türk müziğinde yaygın)
    return 'arabesk'

def save_parsed_data(songs, output_dir="data/raw"):
    """Parse edilen veriyi kaydet"""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON formatında kaydet
    json_path = f"{output_dir}/turkish_lyrics_real_dataset.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(songs, f, ensure_ascii=False, indent=2)
    
    # CSV formatında kaydet
    df = pd.DataFrame(songs)
    csv_path = f"{output_dir}/turkish_lyrics_real_dataset.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"\nGerçek veri kaydedildi:")
    print(f"   JSON: {json_path}")
    print(f"   CSV: {csv_path}")
    print(f"   Toplam şarkı: {len(songs)}")
    
    # Genre dağılımı
    if songs:
        genre_counts = df['genre'].value_counts()
        print(f"   Genre dağılımı:")
        for genre, count in genre_counts.items():
            print(f"      {genre}: {count} şarkı")
    
    return songs

def clean_mock_data():
    """Mock data dosyalarını temizle"""
    print("Mock data dosyaları temizleniyor...")
    
    mock_files = [
        "data/raw/turkish_lyrics_dataset.json",
        "data/raw/turkish_lyrics_dataset.csv",
        "data/raw/bbs_simple_backup_50.json",
        "data/raw/bbs_backup_100.json",
        "data/raw/bbs_backup_75.json",
        "data/raw/bbs_backup_50.json",
        "data/raw/bbs_backup_25.json"
    ]
    
    for file_path in mock_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Silindi: {file_path}")

def main():
    """Ana fonksiyon"""
    print("Gerçek Türkçe Şarkı Verisi Parse Ediliyor...")
    print("="*60)
    
    # Mock dataları temizle
    clean_mock_data()
    
    # SQL dosyasını parse et
    sql_file = "data/raw/turkce_akorlar.sql"
    songs = parse_sql_file(sql_file)
    
    # Veriyi kaydet
    save_parsed_data(songs)
    
    print("\n" + "="*60)
    print("Gerçek veri parse işlemi tamamlandı!")
    print("Artık preprocessing'e geçebiliriz.")

if __name__ == "__main__":
    main()

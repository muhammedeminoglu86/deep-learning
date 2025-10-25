import requests
from bs4 import BeautifulSoup
import json
import time
import random
import os
from datetime import datetime
import pandas as pd

def scrape_bbs_tr_simple(max_songs=500):
    """Basit ve etkili sarkisozleri.bbs.tr scraper"""
    print(f"sarkisozleri.bbs.tr'den {max_songs} şarkı topluyorum...")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    songs = []
    song_count = 0
    
    # Ana sayfa ve farklı sayfa numaraları dene
    base_urls = [
        "https://www.sarkisozleri.bbs.tr/",
        "https://www.sarkisozleri.bbs.tr/index.php",
        "https://www.sarkisozleri.bbs.tr/index.html",
        "https://www.sarkisozleri.bbs.tr/page/1",
        "https://www.sarkisozleri.bbs.tr/page/2",
        "https://www.sarkisozleri.bbs.tr/page/3",
        "https://www.sarkisozleri.bbs.tr/page/4",
        "https://www.sarkisozleri.bbs.tr/page/5"
    ]
    
    for base_url in base_urls:
        if song_count >= max_songs:
            break
            
        print(f"Sayfa işleniyor: {base_url}")
        
        try:
            response = session.get(base_url, timeout=10)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                print(f"Sayfa yüklenemedi: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Tüm linkleri al
            all_links = soup.find_all('a', href=True)
            print(f"Bu sayfada {len(all_links)} link bulundu")
            
            # Her linki şarkı olarak dene
            for link in all_links[:50]:  # İlk 50 linki dene
                if song_count >= max_songs:
                    break
                    
                href = link.get('href')
                if not href:
                    continue
                    
                # URL'yi tamamla
                if href.startswith('/'):
                    href = 'https://www.sarkisozleri.bbs.tr' + href
                elif not href.startswith('http'):
                    continue
                    
                if 'sarkisozleri.bbs.tr' not in href:
                    continue
                
                # Bu linki şarkı olarak dene
                song_data = try_scrape_as_song(session, href)
                if song_data:
                    songs.append(song_data)
                    song_count += 1
                    print(f"[{song_count}] {song_data['title']} - {song_data['artist']}")
                    
                    if song_count % 50 == 0:
                        save_backup(songs, f"bbs_simple_backup_{song_count}.json")
                
                # Rate limiting
                time.sleep(random.uniform(0.3, 0.8))
                
        except Exception as e:
            print(f"Sayfa hatası ({base_url}): {e}")
            continue
    
    # Final kaydet
    save_final_data(songs)
    print(f"Toplam {len(songs)} şarkı toplandı!")
    return songs

def try_scrape_as_song(session, url):
    """URL'yi şarkı olarak scrape etmeyi dene"""
    try:
        response = session.get(url, timeout=8)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Başlık bul
        title_elem = soup.find('h1') or soup.find('title')
        if not title_elem:
            return None
            
        title_text = title_elem.get_text(strip=True)
        
        # Çok kısa başlıkları atla
        if len(title_text) < 3:
            return None
        
        # Sanatçı ve şarkı adını ayır
        artist = "Unknown"
        song_title = title_text
        
        if " - " in title_text:
            parts = title_text.split(" - ")
            if len(parts) >= 2:
                artist = parts[0].strip()
                song_title = " - ".join(parts[1:]).strip()
        
        # Şarkı sözleri bul - tüm metin bloklarını dene
        lyrics_candidates = []
        
        # Farklı element türlerini dene
        for tag in ['div', 'p', 'article', 'section', 'span']:
            for elem in soup.find_all(tag):
                text = elem.get_text(strip=True)
                if len(text) > 50:  # Uzun metin blokları
                    lyrics_candidates.append((elem, len(text)))
        
        if not lyrics_candidates:
            return None
            
        # En uzun metin bloğunu al
        longest_block = max(lyrics_candidates, key=lambda x: x[1])[0]
        lyrics = longest_block.get_text(strip=True)
        
        # Çok kısa şarkıları atla
        if len(lyrics) < 30:
            return None
            
        # Genre belirleme
        genre = determine_genre(song_title, artist, lyrics)
        
        return {
            'title': song_title,
            'artist': artist,
            'lyrics': lyrics,
            'genre': genre,
            'year': 'Unknown',
            'source': 'sarkisozleri.bbs.tr',
            'url': url,
            'scraped_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        return None

def determine_genre(title, artist, lyrics):
    """Genre belirleme"""
    text = (title + " " + artist + " " + lyrics).lower()
    
    genre_keywords = {
        'pop': ['pop', 'şarkı', 'hit', 'melodi', 'tarkan', 'sezen', 'ajda'],
        'rock': ['rock', 'gitar', 'grup', 'metal', 'barış', 'manço', 'mfo'],
        'arabesk': ['arabesk', 'acı', 'hüzün', 'aşk', 'müslüm', 'gürses', 'ferdi'],
        'rap': ['rap', 'hip hop', 'mc', 'beat', 'ceza', 'sagopa', 'kolera'],
        'slow': ['slow', 'ballad', 'romantik', 'aşk', 'tatlıses', 'ibrahim']
    }
    
    genre_scores = {}
    for genre, keywords in genre_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text)
        genre_scores[genre] = score
        
    if genre_scores:
        return max(genre_scores, key=genre_scores.get)
    return 'pop'

def save_backup(songs, filename):
    """Backup kaydet"""
    os.makedirs("data/raw", exist_ok=True)
    backup_path = f"data/raw/{filename}"
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(songs, f, ensure_ascii=False, indent=2)
    
    print(f"Backup kaydedildi: {backup_path}")

def save_final_data(songs):
    """Final veriyi kaydet"""
    os.makedirs("data/raw", exist_ok=True)
    
    # JSON
    json_path = "data/raw/turkish_lyrics_dataset.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(songs, f, ensure_ascii=False, indent=2)
    
    # CSV
    df = pd.DataFrame(songs)
    csv_path = "data/raw/turkish_lyrics_dataset.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"Final veri kaydedildi:")
    print(f"   JSON: {json_path}")
    print(f"   CSV: {csv_path}")
    print(f"   Toplam şarkı: {len(songs)}")
    
    # Genre dağılımı
    if songs:
        genre_counts = df['genre'].value_counts()
        print(f"   Genre dağılımı:")
        for genre, count in genre_counts.items():
            print(f"      {genre}: {count} şarkı")

if __name__ == "__main__":
    import sys
    
    max_songs = 500
    if len(sys.argv) > 1:
        try:
            max_songs = int(sys.argv[1])
        except:
            pass
    
    print(f"Basit Türkçe Şarkı Sözü Scraper - Hedef: {max_songs} şarkı")
    print("=" * 50)
    
    songs = scrape_bbs_tr_simple(max_songs)


import requests
from bs4 import BeautifulSoup
import json
import time
import random
import os
from datetime import datetime
import pandas as pd

def scrape_bbs_tr_fast(max_songs=500):
    """Hızlı sarkisozleri.bbs.tr scraper"""
    print(f"sarkisozleri.bbs.tr'den {max_songs} şarkı topluyorum...")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    songs = []
    visited = set()
    
    # Ana sayfa ve sayfa numaraları
    base_urls = [
        "https://www.sarkisozleri.bbs.tr/",
        "https://www.sarkisozleri.bbs.tr/?page=1",
        "https://www.sarkisozleri.bbs.tr/?page=2", 
        "https://www.sarkisozleri.bbs.tr/?page=3",
        "https://www.sarkisozleri.bbs.tr/?page=4",
        "https://www.sarkisozleri.bbs.tr/?page=5",
        "https://www.sarkisozleri.bbs.tr/?page=6",
        "https://www.sarkisozleri.bbs.tr/?page=7",
        "https://www.sarkisozleri.bbs.tr/?page=8",
        "https://www.sarkisozleri.bbs.tr/?page=9",
        "https://www.sarkisozleri.bbs.tr/?page=10"
    ]
    
    song_count = 0
    
    for page_url in base_urls:
        if song_count >= max_songs:
            break
            
        print(f"Sayfa işleniyor: {page_url}")
        
        try:
            response = session.get(page_url, timeout=10)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                print(f"Sayfa yüklenemedi: {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Şarkı linklerini bul
            links = soup.find_all('a', href=True)
            song_links = []
            
            for link in links:
                href = link.get('href')
                if href and ('sarki' in href.lower() or 'song' in href.lower()):
                    if href.startswith('/'):
                        href = 'https://www.sarkisozleri.bbs.tr' + href
                    elif not href.startswith('http'):
                        continue
                    
                    if href not in visited and 'sarkisozleri.bbs.tr' in href:
                        song_links.append(href)
                        visited.add(href)
            
            print(f"Bu sayfada {len(song_links)} şarkı linki bulundu")
            
            # Her şarkı linkini ziyaret et
            for song_url in song_links[:20]:  # Sayfa başına max 20 şarkı
                if song_count >= max_songs:
                    break
                    
                try:
                    song_data = scrape_single_song(session, song_url)
                    if song_data:
                        songs.append(song_data)
                        song_count += 1
                        print(f"[{song_count}] {song_data['title']} - {song_data['artist']}")
                        
                        # Her 50 şarkıda backup
                        if song_count % 50 == 0:
                            save_backup(songs, f"bbs_fast_backup_{song_count}.json")
                            
                except Exception as e:
                    print(f"Şarkı hatası ({song_url}): {e}")
                    continue
                    
                # Rate limiting
                time.sleep(random.uniform(0.5, 1.5))
                
        except Exception as e:
            print(f"Sayfa hatası ({page_url}): {e}")
            continue
    
    # Final kaydet
    save_final_data(songs)
    print(f"Toplam {len(songs)} şarkı toplandı!")
    return songs

def scrape_single_song(session, url):
    """Tek şarkı sayfasını scrape et"""
    try:
        response = session.get(url, timeout=10)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Başlık
        title_elem = soup.find('h1') or soup.find('title')
        if not title_elem:
            return None
            
        title_text = title_elem.get_text(strip=True)
        
        # Sanatçı ve şarkı adını ayır
        artist = "Unknown"
        song_title = title_text
        
        if " - " in title_text:
            parts = title_text.split(" - ")
            if len(parts) >= 2:
                artist = parts[0].strip()
                song_title = " - ".join(parts[1:]).strip()
        
        # Şarkı sözleri - en uzun metin bloğunu bul
        text_blocks = []
        for elem in soup.find_all(['div', 'p', 'article']):
            text = elem.get_text(strip=True)
            if len(text) > 100:  # Uzun metin blokları
                text_blocks.append((elem, len(text)))
        
        if not text_blocks:
            return None
            
        # En uzun bloğu al
        longest_block = max(text_blocks, key=lambda x: x[1])[0]
        lyrics = longest_block.get_text(strip=True)
        
        # Çok kısa şarkıları atla
        if len(lyrics) < 50:
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
        print(f"Hata ({url}): {e}")
        return None

def determine_genre(title, artist, lyrics):
    """Genre belirleme"""
    text = (title + " " + artist + " " + lyrics).lower()
    
    genre_keywords = {
        'pop': ['pop', 'şarkı', 'hit', 'melodi', 'tarkan', 'sezen'],
        'rock': ['rock', 'gitar', 'grup', 'metal', 'barış', 'manço'],
        'arabesk': ['arabesk', 'acı', 'hüzün', 'aşk', 'müslüm', 'gürses'],
        'rap': ['rap', 'hip hop', 'mc', 'beat', 'ceza', 'sagopa'],
        'slow': ['slow', 'ballad', 'romantik', 'aşk', 'tatlıses']
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
    
    print(f"Türkçe Şarkı Sözü Scraper - Hedef: {max_songs} şarkı")
    print("=" * 50)
    
    songs = scrape_bbs_tr_fast(max_songs)


import requests
from bs4 import BeautifulSoup
import json
import time
import random
import os
from datetime import datetime
import pandas as pd
from urllib.parse import urljoin, urlparse

def scrape_bbs_tr_comprehensive(max_songs=500):
    """Kapsamlı sarkisozleri.bbs.tr scraper"""
    print(f"sarkisozleri.bbs.tr'den {max_songs} şarkı topluyorum...")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    songs = []
    visited_urls = set()
    to_visit = ["https://www.sarkisozleri.bbs.tr/"]
    
    song_count = 0
    page_count = 0
    
    while to_visit and song_count < max_songs and page_count < 100:  # Max 100 sayfa
        current_url = to_visit.pop(0)
        
        if current_url in visited_urls:
            continue
            
        visited_urls.add(current_url)
        page_count += 1
        
        print(f"Sayfa {page_count}: {current_url}")
        
        try:
            response = session.get(current_url, timeout=10)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                continue
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Bu sayfada şarkı var mı kontrol et
            page_songs = extract_songs_from_page(soup, current_url)
            for song in page_songs:
                if song_count >= max_songs:
                    break
                    
                songs.append(song)
                song_count += 1
                print(f"[{song_count}] {song['title']} - {song['artist']}")
                
                if song_count % 50 == 0:
                    save_backup(songs, f"bbs_comprehensive_backup_{song_count}.json")
            
            # Yeni sayfa linklerini bul
            new_links = find_new_links(soup, current_url, visited_urls)
            to_visit.extend(new_links[:10])  # Her sayfadan max 10 yeni link
            
            print(f"Bu sayfada {len(page_songs)} şarkı, {len(new_links)} yeni link bulundu")
            
            # Rate limiting
            time.sleep(random.uniform(1, 2))
            
        except Exception as e:
            print(f"Sayfa hatası ({current_url}): {e}")
            continue
    
    # Final kaydet
    save_final_data(songs)
    print(f"Toplam {len(songs)} şarkı toplandı!")
    return songs

def extract_songs_from_page(soup, page_url):
    """Sayfadan şarkıları çıkar"""
    songs = []
    
    # Şarkı listesi formatını dene
    song_links = soup.find_all('a', href=True)
    
    for link in song_links:
        href = link.get('href')
        if not href:
            continue
            
        # Şarkı linki mi kontrol et
        if is_song_link(href):
            song_url = urljoin(page_url, href)
            
            try:
                song_data = scrape_single_song(session, song_url)
                if song_data:
                    songs.append(song_data)
            except:
                continue
    
    # Eğer şarkı linki bulamadıysak, bu sayfa zaten bir şarkı sayfası olabilir
    if not songs:
        song_data = scrape_single_song(session, page_url)
        if song_data:
            songs.append(song_data)
    
    return songs

def is_song_link(href):
    """Link şarkı linki mi kontrol et"""
    href_lower = href.lower()
    return any(keyword in href_lower for keyword in [
        'sarki', 'song', 'lyrics', 'soz', 'şarkı', 'söz'
    ])

def find_new_links(soup, current_url, visited_urls):
    """Yeni sayfa linklerini bul"""
    new_links = []
    domain = urlparse(current_url).netloc
    
    for link in soup.find_all('a', href=True):
        href = link.get('href')
        if not href:
            continue
            
        # Mutlak URL'e çevir
        full_url = urljoin(current_url, href)
        
        # Aynı domain mi kontrol et
        if urlparse(full_url).netloc != domain:
            continue
            
        # Zaten ziyaret edilmiş mi
        if full_url in visited_urls:
            continue
            
        # Statik dosya değil mi kontrol et
        if any(full_url.lower().endswith(ext) for ext in ['.jpg', '.png', '.css', '.js', '.pdf']):
            continue
            
        new_links.append(full_url)
    
    return new_links

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
        for elem in soup.find_all(['div', 'p', 'article', 'section']):
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
    
    print(f"Kapsamlı Türkçe Şarkı Sözü Scraper - Hedef: {max_songs} şarkı")
    print("=" * 50)
    
    songs = scrape_bbs_tr_comprehensive(max_songs)


import requests
from bs4 import BeautifulSoup
import json
import time
import random
import os
from datetime import datetime
import pandas as pd

def scrape_lyricstranslate_tr(max_songs=500):
    """LyricsTranslate.com'dan Türkçe şarkıları topla"""
    print(f"LyricsTranslate'ten {max_songs} gerçek Türkçe şarkı topluyorum...")
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    songs = []
    visited_urls = set()
    song_count = 0
    
    # LyricsTranslate Türkçe şarkı sayfaları
    base_url = "https://lyricstranslate.com/tr/language/turkish-lyrics"
    
    # Sayfa numaralarını dene (1-20 arası)
    for page_num in range(1, 21):
        if song_count >= max_songs:
            break
        
        page_url = f"{base_url}?page={page_num}" if page_num > 1 else base_url
        print(f"\nSayfa {page_num}: {page_url}")
        
        try:
            response = session.get(page_url, timeout=12)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                print(f"Sayfa yüklenemedi: {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Şarkı linklerini bul - "Şarkı" sütunundaki linkler
            song_links = []
            
            # Tablo satırlarını bul
            rows = soup.find_all('tr')
            print(f"Sayfa {page_num}'de {len(rows)} satır bulundu")
            
            for row in rows:
                # Her satırda 3 kolonda: Şarkı, Dil, (Sanatçı/info)
                cols = row.find_all('td')
                
                if len(cols) >= 2:
                    # İlk kolonda şarkı linki
                    song_cell = cols[0]
                    song_link = song_cell.find('a')
                    
                    if song_link and song_link.get('href'):
                        href = song_link.get('href')
                        title = song_link.get_text(strip=True)
                        
                        # Geçerli bir şarkı linki mi kontrol et
                        if href.startswith('/tr/') and title and len(title) > 2:
                            full_url = f"https://lyricstranslate.com{href}"
                            
                            if full_url not in visited_urls:
                                song_links.append((title, full_url))
                                visited_urls.add(full_url)
            
            print(f"Bu sayfada {len(song_links)} gerçek şarkı linki bulundu")
            
            # Her şarkıyı scrape et
            for title, song_url in song_links:
                if song_count >= max_songs:
                    break
                
                try:
                    song_data = scrape_lyricstranslate_song(session, song_url, title)
                    if song_data:
                        songs.append(song_data)
                        song_count += 1
                        print(f"[{song_count}] {song_data['title']} - {song_data['artist']}")
                        
                        if song_count % 50 == 0:
                            save_backup(songs, f"lyrics_translate_backup_{song_count}.json")
                    
                except Exception as e:
                    print(f"Şarkı hatası: {e}")
                    continue
                
                # Rate limiting
                time.sleep(random.uniform(0.5, 1.5))
            
            # Sayfalar arasında bekleme
            time.sleep(random.uniform(1, 3))
            
        except Exception as e:
            print(f"Sayfa hatası ({page_url}): {e}")
            continue
    
    # Final kaydet
    save_final_data(songs)
    print(f"\nToplam {len(songs)} gerçek Türkçe şarkı toplandı!")
    return songs

def scrape_lyricstranslate_song(session, url, title):
    """LyricsTranslate'ten tek şarkıyı scrape et"""
    try:
        response = session.get(url, timeout=12)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Sanatçı adını bul - "Sanatçı:" etiketi altında
        artist = "Unknown"
        artist_elem = soup.find('a', class_='artist')
        if artist_elem:
            artist = artist_elem.get_text(strip=True)
        
        # Şarkı sözlerini bul - en yaygın konteynırlar
        lyrics = ""
        
        # Div.js-lyricbox sınıfını dene
        lyrics_container = soup.find('div', class_='js-lyricbox')
        if not lyrics_container:
            # Alternatif konteynırları dene
            lyrics_container = soup.find('div', class_='lyricbox')
        if not lyrics_container:
            lyrics_container = soup.find('div', {'data-lyrics': True})
        
        if lyrics_container:
            # Tüm metni al, br'leri yeni satır ile değiştir
            for br in lyrics_container.find_all('br'):
                br.replace_with('\n')
            
            lyrics = lyrics_container.get_text(strip=True)
        
        # Çok kısa veya hiç şarkı sözü yoksa atla
        if len(lyrics) < 50:
            return None
        
        # Genre belirleme
        genre = determine_genre(title, artist, lyrics)
        
        return {
            'title': title,
            'artist': artist,
            'lyrics': lyrics,
            'genre': genre,
            'year': 'Unknown',
            'source': 'lyricstranslate.com',
            'url': url,
            'scraped_at': datetime.now().isoformat()
        }
    
    except Exception as e:
        return None

def determine_genre(title, artist, lyrics):
    """Genre belirleme"""
    text = (title + " " + artist + " " + lyrics).lower()
    
    genre_keywords = {
        'pop': ['pop', 'şarkı', 'hit', 'melodi', 'tarkan', 'sezen', 'ajda', 'deniz seki'],
        'rock': ['rock', 'gitar', 'grup', 'metal', 'barış', 'manço', 'mfo', 'cem karaca'],
        'arabesk': ['arabesk', 'acı', 'hüzün', 'aşk', 'müslüm', 'gürses', 'ferdi tayfur'],
        'rap': ['rap', 'hip hop', 'mc', 'beat', 'ceza', 'sagopa', 'kolera', 'ezhel'],
        'slow': ['slow', 'ballad', 'romantik', 'aşk', 'tatlıses', 'ibrahim', 'rafet']
    }
    
    genre_scores = {}
    for genre, keywords in genre_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text)
        genre_scores[genre] = score
    
    if genre_scores:
        best = max(genre_scores, key=genre_scores.get)
        if genre_scores[best] > 0:
            return best
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
    
    print(f"\nFinal veri kaydedildi:")
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
    
    print(f"LyricsTranslate Türkçe Şarkı Scraper - Hedef: {max_songs} şarkı")
    print("=" * 60)
    
    songs = scrape_lyricstranslate_tr(max_songs)


import requests
from bs4 import BeautifulSoup
import json
import time
import random
from urllib.parse import urljoin, urlparse
import os
from datetime import datetime

class TurkishLyricsScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.scraped_songs = []
        self.genres = ['pop', 'rock', 'arabesk', 'rap', 'slow']
        
    def get_page(self, url, max_retries=3):
        """Sayfa içeriğini güvenli şekilde al"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                print(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(2, 5))
                else:
                    return None
    
    def scrape_genius_turkish(self, max_songs=200):
        """Genius.com'dan Türkçe şarkıları scrape et"""
        print("Genius.com Turkish songs scraping...")
        
        # Türkçe şarkılar için arama sayfaları
        search_urls = [
            "https://genius.com/search/songs?q=turkish",
            "https://genius.com/search/songs?q=turkish%20music",
            "https://genius.com/search/songs?q=turkish%20song"
        ]
        
        song_count = 0
        
        for search_url in search_urls:
            if song_count >= max_songs:
                break
                
            print(f"Search: {search_url}")
            response = self.get_page(search_url)
            
            if not response:
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Şarkı linklerini bul
            song_links = soup.find_all('a', href=True)
            song_urls = []
            
            for link in song_links:
                href = link.get('href')
                if href and '/songs/' in href and href not in song_urls:
                    full_url = urljoin('https://genius.com', href)
                    song_urls.append(full_url)
            
            # Her şarkı sayfasını scrape et
            for song_url in song_urls[:20]:  # Her arama için max 20 şarkı
                if song_count >= max_songs:
                    break
                    
                song_data = self.scrape_genius_song(song_url)
                if song_data:
                    self.scraped_songs.append(song_data)
                    song_count += 1
                    print(f"[{song_count}] Song: {song_data['title']} - {song_data['artist']}")
                    
                    # Her 10 şarkıda bir kaydet
                    if song_count % 10 == 0:
                        self.save_backup(f"genius_backup_{song_count}.json")
                
                # Rate limiting
                time.sleep(random.uniform(1, 3))
    
    def scrape_genius_song(self, url):
        """Tek bir Genius şarkı sayfasını scrape et"""
        response = self.get_page(url)
        if not response:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        try:
            # Şarkı başlığı
            title_elem = soup.find('h1', class_='SongHeaderdesktop__Title-sc-1b7qngi-0')
            if not title_elem:
                title_elem = soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else "Unknown"
            
            # Sanatçı
            artist_elem = soup.find('a', class_='SongHeaderdesktop__Artist-sc-1b7qngi-1')
            if not artist_elem:
                artist_elem = soup.find('span', class_='SongHeaderdesktop__Artist-sc-1b7qngi-1')
            artist = artist_elem.get_text(strip=True) if artist_elem else "Unknown"
            
            # Şarkı sözleri
            lyrics_elem = soup.find('div', class_='Lyrics__Container-sc-1ynbvzw-6')
            if not lyrics_elem:
                lyrics_elem = soup.find('div', {'data-testid': 'lyrics-root'})
            if not lyrics_elem:
                lyrics_elem = soup.find('div', class_='lyrics')
                
            lyrics = ""
            if lyrics_elem:
                # Tüm paragrafları birleştir
                paragraphs = lyrics_elem.find_all(['p', 'div', 'br'])
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and text not in lyrics:
                        lyrics += text + "\n"
            
            # Genre belirleme (basit keyword matching)
            genre = self.determine_genre(title, artist, lyrics)
            
            if len(lyrics.strip()) < 50:  # Çok kısa şarkıları atla
                return None
                
            return {
                'title': title,
                'artist': artist,
                'lyrics': lyrics.strip(),
                'genre': genre,
                'year': 'Unknown',
                'source': 'genius.com',
                'url': url,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error ({url}): {e}")
            return None
    
    def determine_genre(self, title, artist, lyrics):
        """Şarkı içeriğine göre genre belirle"""
        text = (title + " " + artist + " " + lyrics).lower()
        
        genre_keywords = {
            'pop': ['pop', 'şarkı', 'hit', 'melodi'],
            'rock': ['rock', 'gitar', 'grup', 'metal'],
            'arabesk': ['arabesk', 'acı', 'hüzün', 'aşk'],
            'rap': ['rap', 'hip hop', 'mc', 'beat'],
            'slow': ['slow', 'ballad', 'romantik', 'aşk']
        }
        
        genre_scores = {}
        for genre, keywords in genre_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            genre_scores[genre] = score
            
        # En yüksek skorlu genre'yi döndür
        if genre_scores:
            return max(genre_scores, key=genre_scores.get)
        return 'pop'  # Default
    
    def scrape_sarki_sozleri(self, max_songs=300):
        """sarki-sozleri.com'dan şarkıları scrape et"""
        print("sarki-sozleri.com songs scraping...")
        
        # Ana sayfa ve kategori sayfaları
        base_urls = [
            "https://www.sarki-sozleri.com/",
            "https://www.sarki-sozleri.com/kategori/pop",
            "https://www.sarki-sozleri.com/kategori/rock", 
            "https://www.sarki-sozleri.com/kategori/arabesk",
            "https://www.sarki-sozleri.com/kategori/rap"
        ]
        
        song_count = len(self.scraped_songs)
        
        for base_url in base_urls:
            if song_count >= max_songs:
                break
                
            print(f"Page: {base_url}")
            response = self.get_page(base_url)
            
            if not response:
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Şarkı linklerini bul
            song_links = soup.find_all('a', href=True)
            song_urls = []
            
            for link in song_links:
                href = link.get('href')
                if href and '/sarki/' in href and href not in song_urls:
                    full_url = urljoin('https://www.sarki-sozleri.com', href)
                    song_urls.append(full_url)
            
            # Her şarkı sayfasını scrape et
            for song_url in song_urls[:30]:  # Her kategori için max 30 şarkı
                if song_count >= max_songs:
                    break
                    
                song_data = self.scrape_sarki_sozleri_song(song_url)
                if song_data:
                    self.scraped_songs.append(song_data)
                    song_count += 1
                    print(f"[{song_count}] Song: {song_data['title']} - {song_data['artist']}")
                    
                    # Her 10 şarkıda bir kaydet
                    if song_count % 10 == 0:
                        self.save_backup(f"sarki_sozleri_backup_{song_count}.json")
                
                # Rate limiting
                time.sleep(random.uniform(1, 2))
    
    def scrape_sarki_sozleri_song(self, url):
        """Tek bir sarki-sozleri.com şarkı sayfasını scrape et"""
        response = self.get_page(url)
        if not response:
            return None
            
        soup = BeautifulSoup(response.content, 'html.parser')
        
        try:
            # Şarkı başlığı
            title_elem = soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else "Unknown"
            
            # Sanatçı (genellikle başlıkta "Sanatçı - Şarkı" formatında)
            if " - " in title:
                parts = title.split(" - ")
                artist = parts[0].strip()
                song_title = parts[1].strip()
            else:
                artist = "Unknown"
                song_title = title
            
            # Şarkı sözleri
            lyrics_elem = soup.find('div', class_='lyrics')
            if not lyrics_elem:
                lyrics_elem = soup.find('div', class_='content')
            if not lyrics_elem:
                lyrics_elem = soup.find('div', {'id': 'lyrics'})
                
            lyrics = ""
            if lyrics_elem:
                lyrics = lyrics_elem.get_text(strip=True)
            
            # Genre belirleme
            genre = self.determine_genre(song_title, artist, lyrics)
            
            if len(lyrics.strip()) < 50:  # Çok kısa şarkıları atla
                return None
                
            return {
                'title': song_title,
                'artist': artist,
                'lyrics': lyrics.strip(),
                'genre': genre,
                'year': 'Unknown',
                'source': 'sarki-sozleri.com',
                'url': url,
                'scraped_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error ({url}): {e}")
            return None
    
    def save_backup(self, filename):
        """Backup dosyası kaydet"""
        backup_path = f"data/raw/{filename}"
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_songs, f, ensure_ascii=False, indent=2)
        
        print(f"Backup saved: {backup_path}")
    
    def save_final_data(self):
        """Final veriyi kaydet"""
        # CSV formatında kaydet
        import pandas as pd
        
        df = pd.DataFrame(self.scraped_songs)
        
        # JSON formatında kaydet
        json_path = "data/raw/turkish_lyrics_dataset.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_songs, f, ensure_ascii=False, indent=2)
        
        # CSV formatında kaydet
        csv_path = "data/raw/turkish_lyrics_dataset.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"Final data saved:")
        print(f"   JSON: {json_path}")
        print(f"   CSV: {csv_path}")
        print(f"   Total songs: {len(self.scraped_songs)}")
        
        # Genre dağılımı
        if len(self.scraped_songs) > 0:
            genre_counts = df['genre'].value_counts()
            print(f"   Genre distribution:")
            for genre, count in genre_counts.items():
                print(f"      {genre}: {count} songs")
        else:
            print("   No songs collected.")

def main():
    """Ana scraping fonksiyonu"""
    print("Turkish Lyrics Scraper Starting...")
    print("=" * 50)
    
    scraper = TurkishLyricsScraper()
    
    try:
        # Genius.com'dan scrape et
        scraper.scrape_genius_turkish(max_songs=200)
        
        # Sarki-sozleri.com'dan scrape et  
        scraper.scrape_sarki_sozleri(max_songs=300)
        
        # Final veriyi kaydet
        scraper.save_final_data()
        
        print("=" * 50)
        print("Scraping completed!")
        
    except KeyboardInterrupt:
        print("\nScraping stopped by user.")
        scraper.save_final_data()
    except Exception as e:
        print(f"Unexpected error: {e}")
        scraper.save_final_data()

if __name__ == "__main__":
    main()

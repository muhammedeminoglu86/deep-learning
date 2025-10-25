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
        self.seen_song_urls = set()
        
    def get_page(self, url, max_retries=3):
        """Sayfa içeriğini güvenli şekilde al"""
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, timeout=12)
                response.raise_for_status()
                # Encoding'i güvenli ayarla (Türkçe sitelerde önemli)
                if response.apparent_encoding:
                    response.encoding = response.apparent_encoding
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
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
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
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
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
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
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
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
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
    
    def scrape_bbs_tr(self, max_songs: int = 200):
        """sarkisozleri.bbs.tr sitesinden şarkıları topla (basit gezgin)."""
        print("sarkisozleri.bbs.tr songs scraping...")
        start_url = "https://www.sarkisozleri.bbs.tr/"
        domain = "www.sarkisozleri.bbs.tr"
        visited_pages = set()
        candidate_pages = [start_url]
        candidate_set = {start_url}
        song_count = len(self.scraped_songs)
        stop_titles = {"Şarkı Sözleri", "Sanatçılar", "Hakkında", "Ara!", "Şarkı Sözü Ekle"}

        def is_same_domain(href: str) -> bool:
            try:
                return urlparse(href).netloc in (domain, "")
            except Exception:
                return False

        def absolutize(href: str) -> str:
            return urljoin(start_url, href)

        while candidate_pages and song_count < max_songs:
            current = candidate_pages.pop(0)
            if current in visited_pages:
                continue
            visited_pages.add(current)

            resp = self.get_page(current)
            if not resp:
                continue

            soup = BeautifulSoup(resp.content, "html.parser")

            # Basit şarkı sayfası tespiti: başlık + uzun metin bloğu
            title_elem = soup.find("h1")
            page_text_blocks = []
            for el in soup.find_all(["div", "article", "section"]):
                try:
                    text_len = len(el.get_text("\n", strip=True))
                except Exception:
                    text_len = 0
                page_text_blocks.append((el, text_len))
            page_text_blocks.sort(key=lambda x: x[1], reverse=True)

            lyrics_text = ""
            links_in_block = 0
            if page_text_blocks:
                largest_block = page_text_blocks[0][0]
                # Navigasyon/altbilgi içermemesi için link yoğun blokları ele
                text_candidate = largest_block.get_text("\n", strip=True)
                links_in_block = len(largest_block.find_all("a"))
                # Çok kısa blokları ele
                if text_candidate and len(text_candidate) > 120:
                    lyrics_text = text_candidate

            # Şarkı sayfası gibi görünüyorsa kaydet
            if title_elem and lyrics_text and song_count < max_songs:
                title_full = title_elem.get_text(strip=True)
                artist = "Unknown"
                song_title = title_full
                if " - " in title_full:
                    parts = title_full.split(" - ")
                    if len(parts) >= 2:
                        artist = parts[0].strip()
                        song_title = " - ".join(parts[1:]).strip()

                # Çok uzun site içeriği gelirse kaba temizlik (üstteki başlık ve aşırı boşlukları sadeleştir)
                cleaned = "\n".join([ln.strip() for ln in lyrics_text.splitlines() if ln.strip()])

                # Filtreler: başlık kara liste, link yoğun bloklar, tekrar URL'ler
                if (len(cleaned) >= 50 and
                    title_full not in stop_titles and
                    links_in_block <= 15 and
                    current not in self.seen_song_urls):
                    genre = self.determine_genre(song_title, artist, cleaned)
                    self.scraped_songs.append({
                        "title": song_title,
                        "artist": artist,
                        "lyrics": cleaned,
                        "genre": genre,
                        "year": "Unknown",
                        "source": "sarkisozleri.bbs.tr",
                        "url": current,
                        "scraped_at": datetime.now().isoformat(),
                    })
                    self.seen_song_urls.add(current)
                    song_count += 1
                    print(f"[{song_count}] Song: {song_title} - {artist}")
                    if song_count % 50 == 0:
                        self.save_backup(f"bbs_backup_{song_count}.json")

            # Yeni aday linkleri sıraya ekle (aynı domain, tekrar yok, sınırlı sayıda)
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if not href:
                    continue
                if href.startswith("javascript:") or href.startswith("mailto:"):
                    continue
                abs_url = absolutize(href)
                if not is_same_domain(abs_url):
                    continue
                # Statik dosyaları atla
                if any(abs_url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", ".pdf", ".zip")):
                    continue
                if "#" in abs_url:
                    abs_url = abs_url.split("#", 1)[0]
                if abs_url in visited_pages or abs_url in candidate_set:
                    continue
                candidate_pages.append(abs_url)
                candidate_set.add(abs_url)

            # Basit ilerleme çıktısı
            if song_count % 50 == 0:
                print(f"Progress: songs={song_count}, visited={len(visited_pages)}, queue={len(candidate_pages)}")
    
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
        # CLI argümanları (site, max)
        import sys
        site_select = os.environ.get("SCRAPE_SITE", "all").lower()
        if len(sys.argv) >= 2 and sys.argv[1]:
            site_select = sys.argv[1].lower()
        try:
            target_max = int(os.environ.get("MAX_SONGS", "100"))
        except Exception:
            target_max = 100
        if len(sys.argv) >= 3 and sys.argv[2].isdigit():
            target_max = int(sys.argv[2])
        if site_select in ("bbs", "bbs_tr"):
            scraper.scrape_bbs_tr(max_songs=target_max)
        else:
            # Genius.com'dan scrape et
            scraper.scrape_genius_turkish(max_songs=target_max)
            
            # Sarki-sozleri.com'dan scrape et  
            scraper.scrape_sarki_sozleri(max_songs=target_max)
        
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

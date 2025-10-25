import json
import pandas as pd
import os
from datetime import datetime

def create_mock_dataset():
    """Mock Türkçe şarkı sözü dataset'i oluştur"""
    print("Creating mock Turkish lyrics dataset...")
    
    # Mock şarkı verileri
    mock_songs = [
        {
            "title": "Sevda",
            "artist": "Tarkan",
            "lyrics": "Sevda nedir bilmezdim\nSeni görünce anladım\nAşkın ateşi yanar\nKalpte derin izler bırakır\n\nSevda nedir bilmezdim\nSeni görünce anladım\nGözlerinde kayboldum\nRüyalarımda sen varsın",
            "genre": "pop",
            "year": "1997",
            "source": "mock_data",
            "scraped_at": datetime.now().isoformat()
        },
        {
            "title": "Gülüm",
            "artist": "Sezen Aksu",
            "lyrics": "Gülüm benim\nSen benim en güzel çiçeğimsin\nBahçemde açan\nEn güzel gülümsün\n\nGülüm benim\nSen benim en güzel çiçeğimsin\nKalpimde açan\nEn güzel gülümsün",
            "genre": "pop",
            "year": "1995",
            "source": "mock_data",
            "scraped_at": datetime.now().isoformat()
        },
        {
            "title": "Hayat",
            "artist": "Barış Manço",
            "lyrics": "Hayat bir oyun\nOyunda sen\nOyunda ben\nHayat bir şarkı\nŞarkıda sen\nŞarkıda ben\n\nHayat bir aşk\nAşkta sen\nAşkta ben\nHayat bir rüya\nRüyada sen\nRüyada ben",
            "genre": "rock",
            "year": "1988",
            "source": "mock_data",
            "scraped_at": datetime.now().isoformat()
        },
        {
            "title": "Acı",
            "artist": "Müslüm Gürses",
            "lyrics": "Acı nedir bilmezdim\nSeni kaybedince anladım\nAşkın acısı yanar\nKalpte derin yaralar açar\n\nAcı nedir bilmezdim\nSeni kaybedince anladım\nGözlerimde yaşlar\nRüyalarımda sen varsın",
            "genre": "arabesk",
            "year": "1990",
            "source": "mock_data",
            "scraped_at": datetime.now().isoformat()
        },
        {
            "title": "Rap",
            "artist": "Ceza",
            "lyrics": "Rap müzik benim hayatım\nMikrofon elimde\nSözlerim kalbimde\nRap müzik benim hayatım\nBeatler kulaklarımda\nRitimler ayaklarımda\n\nRap müzik benim hayatım\nSokaklar benim evim\nMüzik benim nefesim",
            "genre": "rap",
            "year": "2000",
            "source": "mock_data",
            "scraped_at": datetime.now().isoformat()
        },
        {
            "title": "Aşk",
            "artist": "İbrahim Tatlıses",
            "lyrics": "Aşk nedir bilmezdim\nSeni görünce anladım\nAşkın gücü büyük\nKalpte derin izler bırakır\n\nAşk nedir bilmezdim\nSeni görünce anladım\nGözlerinde kayboldum\nRüyalarımda sen varsın",
            "genre": "slow",
            "year": "1992",
            "source": "mock_data",
            "scraped_at": datetime.now().isoformat()
        },
        {
            "title": "Güneş",
            "artist": "Ajda Pekkan",
            "lyrics": "Güneş doğar her sabah\nIşığıyla aydınlatır\nDünyayı güzelleştirir\nHayatı renklendirir\n\nGüneş doğar her sabah\nIşığıyla aydınlatır\nKalplerimizi ısıtır\nRuhlarımızı besler",
            "genre": "pop",
            "year": "1985",
            "source": "mock_data",
            "scraped_at": datetime.now().isoformat()
        },
        {
            "title": "Deniz",
            "artist": "MFÖ",
            "lyrics": "Deniz mavi\nGökyüzü mavi\nSenin gözlerin mavi\nDeniz mavi\nGökyüzü mavi\nSenin gözlerin mavi\n\nDeniz mavi\nGökyüzü mavi\nSenin gözlerin mavi\nDeniz mavi\nGökyüzü mavi\nSenin gözlerin mavi",
            "genre": "rock",
            "year": "1987",
            "source": "mock_data",
            "scraped_at": datetime.now().isoformat()
        },
        {
            "title": "Hüzün",
            "artist": "Zeki Müren",
            "lyrics": "Hüzün nedir bilmezdim\nSeni kaybedince anladım\nHüzünün acısı yanar\nKalpte derin yaralar açar\n\nHüzün nedir bilmezdim\nSeni kaybedince anladım\nGözlerimde yaşlar\nRüyalarımda sen varsın",
            "genre": "arabesk",
            "year": "1975",
            "source": "mock_data",
            "scraped_at": datetime.now().isoformat()
        },
        {
            "title": "Beat",
            "artist": "Sagopa Kajmer",
            "lyrics": "Beat benim hayatım\nMikrofon elimde\nSözlerim kalbimde\nBeat benim hayatım\nRitimler kulaklarımda\nMüzik ayaklarımda\n\nBeat benim hayatım\nSokaklar benim evim\nMüzik benim nefesim",
            "genre": "rap",
            "year": "2005",
            "source": "mock_data",
            "scraped_at": datetime.now().isoformat()
        }
    ]
    
    # Daha fazla mock data ekle (toplam 50 şarkı)
    genres = ['pop', 'rock', 'arabesk', 'rap', 'slow']
    artists = ['Tarkan', 'Sezen Aksu', 'Barış Manço', 'Müslüm Gürses', 'Ceza', 
              'İbrahim Tatlıses', 'Ajda Pekkan', 'MFÖ', 'Zeki Müren', 'Sagopa Kajmer']
    
    for i in range(40):  # 10 zaten var, 40 daha ekle
        genre = genres[i % len(genres)]
        artist = artists[i % len(artists)]
        
        mock_songs.append({
            "title": f"Şarkı {i+11}",
            "artist": artist,
            "lyrics": f"Bu {genre} tarzında bir şarkı\nSanatçı: {artist}\nŞarkı numarası: {i+11}\n\nBu şarkı Türkçe müzik kültürünün\nBir parçası olarak yazılmıştır\n\n{genre} müziğin güzelliği\nTürk halkının kalbinde\nDerin izler bırakır",
            "genre": genre,
            "year": str(1980 + (i % 30)),
            "source": "mock_data",
            "scraped_at": datetime.now().isoformat()
        })
    
    # Klasörleri oluştur
    os.makedirs("data/raw", exist_ok=True)
    
    # JSON formatında kaydet
    json_path = "data/raw/turkish_lyrics_dataset.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mock_songs, f, ensure_ascii=False, indent=2)
    
    # CSV formatında kaydet
    df = pd.DataFrame(mock_songs)
    csv_path = "data/raw/turkish_lyrics_dataset.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    print(f"Mock dataset created:")
    print(f"   JSON: {json_path}")
    print(f"   CSV: {csv_path}")
    print(f"   Total songs: {len(mock_songs)}")
    
    # Genre dağılımı
    genre_counts = df['genre'].value_counts()
    print(f"   Genre distribution:")
    for genre, count in genre_counts.items():
        print(f"      {genre}: {count} songs")
    
    return mock_songs

if __name__ == "__main__":
    create_mock_dataset()

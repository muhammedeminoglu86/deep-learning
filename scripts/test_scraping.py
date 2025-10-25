import requests
from bs4 import BeautifulSoup
import json
import time
import random
import os
from datetime import datetime

def test_scraping():
    """Basit scraping testi"""
    print("Testing web scraping...")
    
    # Test URL'leri
    test_urls = [
        "https://www.sarki-sozleri.com/",
        "https://genius.com/search/songs?q=turkish"
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for url in test_urls:
        print(f"Testing: {url}")
        try:
            response = session.get(url, timeout=10)
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                print(f"Title: {soup.title.string if soup.title else 'No title'}")
            else:
                print(f"Failed to access: {url}")
        except Exception as e:
            print(f"Error: {e}")
        
        time.sleep(2)
    
    print("Test completed!")

if __name__ == "__main__":
    test_scraping()


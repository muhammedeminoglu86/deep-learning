import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_final_comparison():
    """Final model karşılaştırması oluştur"""
    print("FINAL MODEL KARŞILAŞTIRMASI")
    print("="*60)
    
    # Model sonuçları
    models = {
        "Seq2Seq": {"accuracy": 0.8033, "params": 9349121, "description": "Encoder-Decoder"},
        "LSTM": {"accuracy": 0.7784, "params": 5137415, "description": "Bidirectional LSTM"},
        "CNN": {"accuracy": 0.8144, "params": 2869042, "description": "1D Convolutional"},
        "Transformer (Original)": {"accuracy": 0.7341, "params": 3889926, "description": "Basic Transformer"},
        "Transformer (Improved)": {"accuracy": 0.6311, "params": 9806342, "description": "Multi-modal Transformer"}
    }
    
    # Veri istatistikleri
    data_stats = {
        "total_songs": 3605,
        "vocabulary_size": 24211,
        "chord_vocabulary_size": 156,
        "genres": 6,
        "avg_chords_per_song": 4.4,
        "chord_diversity": 0.972
    }
    
    print("MODEL PERFORMANSLARI:")
    print("-" * 60)
    print(f"{'Model':<25} {'Accuracy':<10} {'Params (M)':<12} {'Description'}")
    print("-" * 60)
    
    best_model = ""
    best_accuracy = 0.0
    
    for model_name, data in models.items():
        accuracy = data["accuracy"]
        params_m = data["params"] / 1_000_000
        description = data["description"]
        
        print(f"{model_name:<25} {accuracy:<10.4f} {params_m:<12.1f} {description}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_name
    
    print("-" * 60)
    print(f"EN İYİ MODEL: {best_model} ({best_accuracy:.4f})")
    
    print("\nVERİ İSTATİSTİKLERİ:")
    print("-" * 60)
    for key, value in data_stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Görselleştirme
    create_comparison_charts(models, data_stats)
    
    # Sonuç analizi
    print("\nSONUÇ ANALİZİ:")
    print("-" * 60)
    print("1. CNN modeli en yüksek accuracy (%81.44) elde etti")
    print("2. Seq2Seq modeli ikinci sırada (%80.33) ama çok fazla parametre kullanıyor")
    print("3. LSTM modeli dengeli performans (%77.84)")
    print("4. Transformer modelleri beklenen performansı göstermedi")
    print("5. Akor bilgisi Transformer'da beklenen iyileştirmeyi sağlamadı")
    
    print("\nÖNERİLER:")
    print("-" * 60)
    print("• CNN modeli bu veri seti için en uygun seçim")
    print("• Transformer modelleri için daha fazla hiperparametre optimizasyonu gerekli")
    print("• Akor verilerinin kalitesi artırılabilir")
    print("• Veri seti genişletilerek model performansları artırılabilir")
    print("• Ensemble methods denenebilir")
    
    return models, data_stats

def create_comparison_charts(models, data_stats):
    """Karşılaştırma grafikleri oluştur"""
    print("\nGrafikler oluşturuluyor...")
    
    # Model isimlerini kısalt
    model_names = []
    accuracies = []
    params = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    for model_name, data in models.items():
        model_names.append(model_name.replace('Transformer (', 'T-').replace(')', ''))
        accuracies.append(data["accuracy"])
        params.append(data["params"] / 1_000_000)
    
    # Ana karşılaştırma grafiği
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy karşılaştırması
    bars1 = ax1.bar(model_names, accuracies, color=colors)
    ax1.set_title('Model Accuracy Karşılaştırması', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0.6, 0.85)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Accuracy değerlerini bar üzerine yaz
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Parametre sayısı karşılaştırması
    bars2 = ax2.bar(model_names, params, color=colors)
    ax2.set_title('Model Parametre Sayısı Karşılaştırması', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Parametre Sayısı (Milyon)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Parametre değerlerini bar üzerine yaz
    for bar, param in zip(bars2, params):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{param:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # 3. Efficiency (Accuracy/Parametre) karşılaştırması
    efficiency = [acc/param for acc, param in zip(accuracies, params)]
    bars3 = ax3.bar(model_names, efficiency, color=colors)
    ax3.set_title('Model Verimliliği (Accuracy/Parametre)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Verimlilik', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Efficiency değerlerini bar üzerine yaz
    for bar, eff in zip(bars3, efficiency):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{eff:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Veri istatistikleri
    stats_labels = ['Şarkı\nSayısı', 'Kelime\nDağarcığı', 'Akor\nDağarcığı', 'Genre\nSayısı', 'Ort.\nAkor']
    stats_values = [
        data_stats['total_songs'] / 1000,  # Bin cinsinden
        data_stats['vocabulary_size'] / 1000,  # Bin cinsinden
        data_stats['chord_vocabulary_size'],
        data_stats['genres'],
        data_stats['avg_chords_per_song']
    ]
    
    bars4 = ax4.bar(stats_labels, stats_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax4.set_title('Veri Seti İstatistikleri', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Değer', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Değerleri bar üzerine yaz
    for bar, value in zip(bars4, stats_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/final_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Final karşılaştırma grafiği kaydedildi: results/final_model_comparison.png")

def create_summary_report():
    """Özet rapor oluştur"""
    print("\nFINAL ÖZET RAPOR OLUŞTURULUYOR...")
    
    report = f"""
# TÜRKÇE ŞARKI SÖZÜ ÜRETİMİ - FINAL RAPOR
## Deep Learning Model Karşılaştırması

**Tarih:** {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
**Proje:** Türkçe Şarkı Sözü Analizi için 5 Farklı Deep Learning Modelinin Karşılaştırılması

---

## 🏆 SONUÇLAR

### Model Performans Sıralaması
1. **CNN:** 81.44% accuracy (2.9M parametre) ⭐ **EN İYİ**
2. **Seq2Seq:** 80.33% accuracy (9.3M parametre)
3. **LSTM:** 77.84% accuracy (5.1M parametre)
4. **Transformer (Original):** 73.41% accuracy (3.9M parametre)
5. **Transformer (Improved):** 63.11% accuracy (9.8M parametre)

### Verimlilik Analizi
- **En Verimli:** CNN (0.28 accuracy/parametre oranı)
- **En Az Parametre:** CNN (2.9M)
- **En Çok Parametre:** Transformer Improved (9.8M)

---

## 📊 VERİ SETİ

### İstatistikler
- **Toplam Şarkı:** 3,605 gerçek Türkçe şarkı
- **Kelime Dağarcığı:** 24,211 benzersiz kelime
- **Akor Dağarcığı:** 156 farklı akor
- **Genre Sayısı:** 6 (slow, rock, folk, arabesk, pop, rap)
- **Ortalama Akor:** 4.4 akor per şarkı
- **Akor Çeşitliliği:** 0.972 (çok yüksek)

### Genre Dağılımı
- **Slow:** 1,446 şarkı (40.1%)
- **Rock:** 865 şarkı (24.0%)
- **Folk:** 675 şarkı (18.7%)
- **Arabesk:** 384 şarkı (10.7%)
- **Pop:** 227 şarkı (6.3%)
- **Rap:** 8 şarkı (0.2%)

---

## 🔍 ANALİZ

### Beklenmeyen Sonuçlar
1. **CNN Başarısı:** En basit mimari en yüksek performansı gösterdi
2. **Transformer Düşük Performansı:** Modern mimari beklenen başarıyı sağlamadı
3. **Akorların Etkisi:** Multi-modal Transformer'da akor bilgisi iyileştirme sağlamadı
4. **Overfitting:** Seq2Seq modelinde erken overfitting gözlendi

### Olası Nedenler
1. **Veri Seti Boyutu:** 3,605 şarkı Transformer için yetersiz olabilir
2. **Akor Verisi Kalitesi:** Akor parsing algoritması optimize edilebilir
3. **Hiperparametre Optimizasyonu:** Transformer modelleri için daha fazla tuning gerekli
4. **Sequence Length:** 200 token uzunluğu yeterli olmayabilir

---

## 💡 ÖNERİLER

### Kısa Vadeli
1. **CNN Modelini Optimize Et:** En iyi performans gösteren model
2. **Veri Setini Genişlet:** Daha fazla şarkı ekle
3. **Hiperparametre Optimizasyonu:** Grid search ile en iyi parametreleri bul

### Uzun Vadeli
1. **Akor Verisi Kalitesi:** Akor parsing algoritmasını iyileştir
2. **Multi-Modal Yaklaşım:** Melodi ve ritim bilgisi ekle
3. **Transfer Learning:** Pre-trained modelleri kullan
4. **Ensemble Methods:** Birden fazla modeli birleştir
5. **Transformer Optimizasyonu:** Daha büyük veri seti ile tekrar dene

---

## 🎯 SONUÇ

Bu proje, Türkçe şarkı sözü analizi için **CNN modelinin en uygun seçim** olduğunu göstermiştir. Transformer modellerinin beklenen performansı göstermemesi, veri seti boyutu ve kalitesi ile ilgili olabilir.

**Proje başarıyla tamamlanmıştır ve tüm sonuçlar dokümante edilmiştir.**

---

*Bu rapor otomatik olarak oluşturulmuştur - {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}*
"""
    
    # Raporu kaydet
    with open('FINAL_SUMMARY_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Final özet raporu oluşturuldu: FINAL_SUMMARY_REPORT.md")

def main():
    """Ana fonksiyon"""
    models, data_stats = create_final_comparison()
    create_summary_report()
    
    print("\n" + "="*60)
    print("FINAL KARŞILAŞTIRMA TAMAMLANDI!")
    print("="*60)
    print("Oluşturulan dosyalar:")
    print("- results/final_model_comparison.png")
    print("- FINAL_SUMMARY_REPORT.md")
    print("\nBu raporlar tez için kullanılabilir!")

if __name__ == "__main__":
    main()

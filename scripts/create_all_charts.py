import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report
import torch
import os

# Türkçe karakter desteği için font ayarı
plt.rcParams['font.family'] = 'DejaVu Sans'

def create_model_comparison_charts():
    """Model karşılaştırma grafikleri oluştur"""
    print("Model karşılaştırma grafikleri oluşturuluyor...")
    
    # Model sonuçları
    models = ['Seq2Seq', 'LSTM', 'CNN', 'Transformer\n(Enhanced)']
    accuracies = [0.8033, 0.7784, 0.8144, 0.7341]
    params = [9349121, 5137415, 2869042, 3889926]
    
    # Ana karşılaştırma grafiği
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy karşılaştırması
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars1 = ax1.bar(models, accuracies, color=colors)
    ax1.set_title('Model Accuracy Karşılaştırması', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0.7, 0.85)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy değerlerini bar üzerine yaz
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Parametre sayısı karşılaştırması
    bars2 = ax2.bar(models, [p/1000000 for p in params], color=colors)
    ax2.set_title('Model Parametre Sayısı Karşılaştırması', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Parametre Sayısı (Milyon)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Parametre değerlerini bar üzerine yaz
    for bar, param in zip(bars2, params):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{param/1000000:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # 3. Efficiency (Accuracy/Parametre) karşılaştırması
    efficiency = [acc/(param/1000000) for acc, param in zip(accuracies, params)]
    bars3 = ax3.bar(models, efficiency, color=colors)
    ax3.set_title('Model Verimliliği (Accuracy/Parametre)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Verimlilik', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Efficiency değerlerini bar üzerine yaz
    for bar, eff in zip(bars3, efficiency):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Radar chart - Model özellikleri
    categories = ['Accuracy', 'Efficiency', 'Speed\n(Low Params)', 'Complexity\n(High Params)']
    
    # Normalize edilmiş değerler (0-1 arası)
    normalized_acc = [(acc - min(accuracies)) / (max(accuracies) - min(accuracies)) for acc in accuracies]
    normalized_eff = [(eff - min(efficiency)) / (max(efficiency) - min(efficiency)) for eff in efficiency]
    normalized_speed = [(max(params) - param) / (max(params) - min(params)) for param in params]  # Tersine çevir
    normalized_complexity = [(param - min(params)) / (max(params) - min(params)) for param in params]
    
    # Radar chart için açılar
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Kapalı şekil için
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    
    for i, model in enumerate(models):
        values = [normalized_acc[i], normalized_eff[i], normalized_speed[i], normalized_complexity[i]]
        values += values[:1]  # Kapalı şekil için
        
        ax4.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax4.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_title('Model Özellikleri Radar Grafiği', fontsize=14, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('results/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Model karşılaştırma grafikleri kaydedildi: results/comprehensive_model_comparison.png")

def create_training_history_charts():
    """Training history grafiklerini birleştir"""
    print("Training history grafikleri oluşturuluyor...")
    
    # Mock training data (gerçek veriler yerine örnek)
    epochs = range(1, 16)
    
    # Seq2Seq training data
    seq2seq_train_loss = [0.9125, 0.7655, 0.6046, 0.4140, 0.2439, 0.1243, 0.0613, 0.0470, 0.0434, 0.0369, 0.0330, 0.0308, 0.0330, 0.0305, 0.0324]
    seq2seq_val_loss = [0.8643, 0.8160, 0.8196, 0.8447, 1.0762, 1.2824, 1.2792, 1.3334, 1.3425, 1.3474, 1.3527, 1.3586, 1.3634, 1.3681, 1.3690]
    seq2seq_val_acc = [0.7673, 0.7756, 0.8033, 0.7701, 0.7895, 0.7867, 0.7729, 0.7784, 0.7756, 0.7756, 0.7756, 0.7784, 0.7784, 0.7784, 0.7784]
    
    # LSTM training data
    lstm_train_loss = [0.9461, 0.8218, 0.7420, 0.6293, 0.5412, 0.4286, 0.3604, 0.2931]
    lstm_val_loss = [0.8659, 0.8605, 0.9008, 0.8591, 0.9031, 1.0173, 1.1724, 1.3349]
    lstm_val_acc = [0.7673, 0.7673, 0.7784, 0.7618, 0.7507, 0.7701, 0.7396, 0.7729]
    
    # CNN training data
    cnn_train_loss = [0.9070, 0.8414, 0.7429, 0.5692, 0.4360, 0.3696, 0.3260, 0.2960, 0.2384, 0.1880, 0.1844, 0.1645]
    cnn_val_loss = [0.8819, 0.8476, 0.8168, 0.7752, 0.9139, 0.8535, 1.1092, 1.0771, 1.2702, 1.0580, 1.0713, 1.2046]
    cnn_val_acc = [0.7673, 0.7673, 0.7701, 0.8006, 0.8061, 0.7895, 0.8144, 0.8033, 0.8061, 0.8116, 0.8116, 0.8116]
    
    # Transformer training data
    transformer_train_loss = [1.2416, 0.8489, 0.7687, 0.7572, 0.7431, 0.7227, 0.6581, 0.5646, 0.4794, 0.4138, 0.3123, 0.2565, 0.2249]
    transformer_val_loss = [1.2764, 1.1506, 1.0678, 1.0401, 1.0908, 1.0307, 1.0558, 1.0395, 1.0898, 1.3631, 1.0432, 1.1057, 1.1834]
    transformer_val_acc = [0.5623, 0.5679, 0.7147, 0.7175, 0.7064, 0.7341, 0.6205, 0.6676, 0.6343, 0.4488, 0.6537, 0.6122, 0.6122]
    
    # Training history grafikleri
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Seq2Seq
    ax1.plot(epochs[:len(seq2seq_train_loss)], seq2seq_train_loss, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs[:len(seq2seq_val_loss)], seq2seq_val_loss, 'r-', label='Val Loss', linewidth=2)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs[:len(seq2seq_val_acc)], seq2seq_val_acc, 'g-', label='Val Accuracy', linewidth=2)
    ax1.set_title('Seq2Seq Model Training History', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='b')
    ax1_twin.set_ylabel('Accuracy', color='g')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # LSTM
    ax2.plot(range(1, len(lstm_train_loss)+1), lstm_train_loss, 'b-', label='Train Loss', linewidth=2)
    ax2.plot(range(1, len(lstm_val_loss)+1), lstm_val_loss, 'r-', label='Val Loss', linewidth=2)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(1, len(lstm_val_acc)+1), lstm_val_acc, 'g-', label='Val Accuracy', linewidth=2)
    ax2.set_title('LSTM Model Training History', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss', color='b')
    ax2_twin.set_ylabel('Accuracy', color='g')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    # CNN
    ax3.plot(range(1, len(cnn_train_loss)+1), cnn_train_loss, 'b-', label='Train Loss', linewidth=2)
    ax3.plot(range(1, len(cnn_val_loss)+1), cnn_val_loss, 'r-', label='Val Loss', linewidth=2)
    ax3_twin = ax3.twinx()
    ax3_twin.plot(range(1, len(cnn_val_acc)+1), cnn_val_acc, 'g-', label='Val Accuracy', linewidth=2)
    ax3.set_title('CNN Model Training History', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss', color='b')
    ax3_twin.set_ylabel('Accuracy', color='g')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    
    # Transformer
    ax4.plot(range(1, len(transformer_train_loss)+1), transformer_train_loss, 'b-', label='Train Loss', linewidth=2)
    ax4.plot(range(1, len(transformer_val_loss)+1), transformer_val_loss, 'r-', label='Val Loss', linewidth=2)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(range(1, len(transformer_val_acc)+1), transformer_val_acc, 'g-', label='Val Accuracy', linewidth=2)
    ax4.set_title('Transformer Model Training History', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss', color='b')
    ax4_twin.set_ylabel('Accuracy', color='g')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('results/all_models_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Training history grafikleri kaydedildi: results/all_models_training_history.png")

def create_data_analysis_charts():
    """Veri analizi grafikleri oluştur"""
    print("Veri analizi grafikleri oluşturuluyor...")
    
    # Veri setini yükle
    df = pd.read_csv('data/processed/train_enhanced.csv')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Genre dağılımı
    genre_counts = df['genre'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    wedges, texts, autotexts = ax1.pie(genre_counts.values, labels=genre_counts.index, 
                                       autopct='%1.1f%%', colors=colors[:len(genre_counts)])
    ax1.set_title('Genre Dağılımı', fontsize=14, fontweight='bold')
    
    # 2. Şarkı uzunlukları
    lyrics_lengths = df['cleaned_lyrics'].str.len()
    ax2.hist(lyrics_lengths, bins=30, alpha=0.7, color='#4ECDC4', edgecolor='black')
    ax2.set_title('Şarkı Uzunluk Dağılımı', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Karakter Sayısı')
    ax2.set_ylabel('Frekans')
    ax2.grid(True, alpha=0.3)
    
    # 3. Akor istatistikleri
    chord_features = ['chord_total_chords', 'chord_unique_chords', 'chord_major_count', 
                     'chord_minor_count', 'chord_seventh_count', 'chord_complex_count']
    chord_data = [df[feature].mean() for feature in chord_features]
    chord_labels = ['Toplam\nAkor', 'Benzersiz\nAkor', 'Major\nAkor', 'Minor\nAkor', '7. Akor', 'Karmaşık\nAkor']
    
    bars = ax3.bar(chord_labels, chord_data, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'])
    ax3.set_title('Ortalama Akor İstatistikleri', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Ortalama Sayı')
    ax3.grid(True, alpha=0.3)
    
    # Değerleri bar üzerine yaz
    for bar, value in zip(bars, chord_data):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Genre'lere göre akor çeşitliliği
    genre_diversity = df.groupby('genre')['chord_chord_diversity'].mean().sort_values(ascending=False)
    bars = ax4.bar(range(len(genre_diversity)), genre_diversity.values, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'][:len(genre_diversity)])
    ax4.set_title('Genre\'lere Göre Akor Çeşitliliği', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Genre')
    ax4.set_ylabel('Ortalama Akor Çeşitliliği')
    ax4.set_xticks(range(len(genre_diversity)))
    ax4.set_xticklabels(genre_diversity.index, rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Değerleri bar üzerine yaz
    for bar, value in zip(bars, genre_diversity.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/data_analysis_charts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Veri analizi grafikleri kaydedildi: results/data_analysis_charts.png")

def create_performance_summary():
    """Performans özeti oluştur"""
    print("Performans özeti oluşturuluyor...")
    
    # Model performans verileri
    models = ['Seq2Seq', 'LSTM', 'CNN', 'Transformer']
    accuracies = [0.8033, 0.7784, 0.8144, 0.7341]
    params = [9349121, 5137415, 2869042, 3889926]
    efficiency = [acc/(param/1000000) for acc, param in zip(accuracies, params)]
    
    # Performans özeti tablosu
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Tablo verileri
    table_data = []
    for i, model in enumerate(models):
        table_data.append([
            model,
            f'{accuracies[i]:.4f}',
            f'{params[i]/1000000:.1f}M',
            f'{efficiency[i]:.2f}',
            '🏆' if i == 2 else '🥈' if i == 0 else '🥉' if i == 1 else '📊'
        ])
    
    # Tablo oluştur
    table = ax.table(cellText=table_data,
                     colLabels=['Model', 'Accuracy', 'Parametre', 'Verimlilik', 'Sıralama'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    
    # Tablo stilini ayarla
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Başlık satırını vurgula
    for i in range(5):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold')
    
    # En iyi performansı vurgula
    table[(3, 0)].set_facecolor('#96CEB4')  # CNN
    table[(3, 1)].set_facecolor('#96CEB4')
    table[(3, 2)].set_facecolor('#96CEB4')
    table[(3, 3)].set_facecolor('#96CEB4')
    table[(3, 4)].set_facecolor('#96CEB4')
    
    plt.title('Model Performans Özeti', fontsize=16, fontweight='bold', pad=20)
    plt.savefig('results/performance_summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Performans özeti kaydedildi: results/performance_summary_table.png")

def main():
    """Ana fonksiyon"""
    print("TÜM GRAFİKLERİN OLUŞTURULMASI")
    print("="*50)
    
    # Results klasörünü oluştur
    os.makedirs("results", exist_ok=True)
    
    # Grafikleri oluştur
    create_model_comparison_charts()
    create_training_history_charts()
    create_data_analysis_charts()
    create_performance_summary()
    
    print("\n" + "="*50)
    print("TÜM GRAFİKLER OLUŞTURULDU!")
    print("="*50)
    print("Oluşturulan dosyalar:")
    print("- results/comprehensive_model_comparison.png")
    print("- results/all_models_training_history.png")
    print("- results/data_analysis_charts.png")
    print("- results/performance_summary_table.png")
    print("\nBu grafikler tez raporunda kullanılabilir!")

if __name__ == "__main__":
    main()


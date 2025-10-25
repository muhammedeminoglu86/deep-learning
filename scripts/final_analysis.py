import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Model sonuçları
models = ['Seq2Seq', 'LSTM', 'CNN', 'Transformer\n(Enhanced)']
accuracies = [0.8033, 0.7784, 0.8144, 0.7341]
params = [9349121, 5137415, 2869042, 3889926]

# Grafik oluştur
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy karşılaştırması
bars1 = ax1.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax1.set_title('Model Accuracy Karşılaştırması', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_ylim(0.7, 0.85)
ax1.grid(True, alpha=0.3)

# Accuracy değerlerini bar üzerine yaz
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

# Parametre sayısı karşılaştırması
bars2 = ax2.bar(models, [p/1000000 for p in params], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax2.set_title('Model Parametre Sayısı Karşılaştırması', fontsize=14, fontweight='bold')
ax2.set_ylabel('Parametre Sayısı (Milyon)', fontsize=12)
ax2.grid(True, alpha=0.3)

# Parametre değerlerini bar üzerine yaz
for bar, param in zip(bars2, params):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{param/1000000:.1f}M', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Detaylı analiz tablosu
print("DETAYLI MODEL ANALİZİ")
print("="*50)
print(f"{'Model':<20} {'Accuracy':<10} {'Parametre':<12} {'Efficiency':<12}")
print("-"*50)

efficiency = [acc/(param/1000000) for acc, param in zip(accuracies, params)]
for model, acc, param, eff in zip(models, accuracies, params, efficiency):
    print(f"{model:<20} {acc:<10.4f} {param/1000000:<12.1f}M {eff:<12.2f}")

print("\nSONUÇLAR:")
print("="*50)
print("1. CNN modeli en yüksek accuracy (%81.44) elde etti")
print("2. En verimli model: CNN (accuracy/parametre oranı en yüksek)")
print("3. Transformer modeli akorlar dahil olmasına rağmen daha düşük performans")
print("4. Seq2Seq modeli iyi accuracy ama çok fazla parametre kullanıyor")
print("5. LSTM modeli dengeli performans gösteriyor")

print("\nÖNERİLER:")
print("="*50)
print("- CNN modeli bu veri seti için en uygun seçim")
print("- Akor verilerinin kalitesi artırılabilir")
print("- Transformer modeli için daha fazla hiperparametre optimizasyonu gerekli")
print("- Veri seti genişletilerek model performansları artırılabilir")


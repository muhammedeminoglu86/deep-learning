import pandas as pd
import json

print('TÜM MODELLERİN SONUÇLARI KARŞILAŞTIRMASI')
print('='*60)

# Model sonuçları
models = {
    'Seq2Seq': {'accuracy': 0.8033, 'params': '9,349,121'},
    'LSTM': {'accuracy': 0.7784, 'params': '5,137,415'},
    'CNN': {'accuracy': 0.8144, 'params': '2,869,042'},
    'Transformer (Enhanced)': {'accuracy': 0.7341, 'params': '3,889,926'}
}

print('Model Performansları:')
print('-'*40)
for model, data in models.items():
    print(f'{model:20}: {data["accuracy"]:.4f} ({data["params"]} parametre)')

print('\nEn İyi Performans:')
best_model = max(models.items(), key=lambda x: x[1]['accuracy'])
print(f'{best_model[0]}: {best_model[1]["accuracy"]:.4f}')

print('\nVeri İstatistikleri:')
print('-'*40)
df = pd.read_csv('data/processed/train_enhanced.csv')
print(f'Toplam şarkı: {len(df)}')
print(f'Genre dağılımı:')
genre_counts = df['genre'].value_counts()
for genre, count in genre_counts.items():
    print(f'  {genre}: {count} şarkı')

print(f'\nAkor istatistikleri:')
print(f'  Ortalama akor sayısı: {df["chord_total_chords"].mean():.1f}')
print(f'  Ortalama benzersiz akor: {df["chord_unique_chords"].mean():.1f}')
print(f'  Ortalama akor çeşitliliği: {df["chord_chord_diversity"].mean():.3f}')

print('\nSonuçlar:')
print('-'*40)
print('1. CNN modeli en yüksek accuracy (%81.44) elde etti')
print('2. Seq2Seq modeli ikinci sırada (%80.33)')
print('3. LSTM modeli üçüncü sırada (%77.84)')
print('4. Transformer modeli akorlar dahil olmasına rağmen daha düşük (%73.41)')
print('\nAkorların eklenmesi Transformer modelinde beklenen iyileştirmeyi sağlamadı.')
print('Bu durum akor verilerinin kalitesi veya model mimarisinin optimize edilmesi gerektiğini gösteriyor.')



# TURKCE SARKI SOZU ANALIZI - FINAL SONUCLAR

## EN IYI MODEL: CNN
**Accuracy:** 81.44%
**Parametre:** 2.9M
**Durum:** OK EN IYI PERFORMANS

## TUM MODEL SONUCLARI

- CNN: 81.44% EN IYI
- Seq2Seq: 80.33% 
- LSTM: 77.84% 
- Transformer (Original): 73.41% 
- Transformer (Improved): 63.11% 
- Hybrid: 52.35% 


## ANALIZ

### Basarili Modeller
1. **CNN** - En basit ve en basarili (%81.44)
2. **Seq2Seq** - Iyi performans ama fazla parametre (%80.33)
3. **LSTM** - Dengeli performans (%77.84)

### Basarisiz Modeller
1. **Hibrid** - Overfitting problemi (%52.35)
2. **Transformer (Original)** - Dusuk performans (%73.41)
3. **Transformer (Improved)** - Cok dusuk performans (%63.11)

## ONERILER

1. **CNN modeli kullan** - En iyi performans
2. **Hibrit modelden kacin** - Overfitting problemi
3. **Basit ensemble dene** - CNN + Seq2Seq birlestir
4. **Daha fazla veri** - Model performansini artirir
5. **Regularizasyon** - Overfitting'i onler

## SONUC

Bu projede **CNN modeli** Turkce sarki sozu analizi icin en uygun secim olmustur.
Basit mimarisi ve %81.44 accuracy'si ile en yuksek basariyi saglamistir.

**Proje basariyla tamamlandi!**


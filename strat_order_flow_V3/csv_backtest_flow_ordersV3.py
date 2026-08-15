import pandas as pd
import numpy as np
import csv
import itertools
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numba

LIMIT_TIME_MIN = 30
LIMIT_TIME_MAX = 200
MIN_ASK = 0.32
MAX_ASK = 0.82
PROBA = 0.65
TR_DROP = 0.4
LIMIT_DROP = 0.6
TRAILING_SL = 0.2
RR = 1.25

print("Chargement des données...")
df = pd.read_csv("entrainement_xboost_V3.csv", sep=";", encoding="utf-8-sig")
print(f"Nombre de lignes chargées : {len(df)}")

# ── Conversion et nettoyage ──
# numeric_cols = ['side', 'ask', 'bid', 'restant', 'imbalance', 'pred_PM', 'pred_DD', 'entry_signal']
# for col in numeric_cols:
    # if col in df.columns:
        # df[col] = pd.to_numeric(df[col], errors='coerce')

# df = df.dropna(subset=numeric_cols).reset_index(drop=True)
# print(f"Après nettoyage : {len(df)} lignes")

# ── Convertir en arrays numpy UNE SEULE FOIS ──
# (au lieu de df.iloc[i] à chaque itération)
ask          = df["ask"].values.astype(np.float32)
bid          = df["bid"].values.astype(np.float32)
restant      = df["restant"].values.astype(np.float32)
side         = df["side"].values.astype(np.float32)
imbalance    = df["imbalance"].values.astype(np.float32)
pred_PM      = df["pred_PM"].values.astype(np.float32)
pred_DD      = df["pred_DD"].values.astype(np.float32)
entry_signal = df["entry_signal"].values.astype(np.float32)

N = len(ask)

# ── Pré-calculer les indices de début de nouvelle bougie ──
# Quand restant[i+1] > restant[i] → nouvelle bougie
new_candle_mask = np.zeros(N, dtype=np.bool_)
new_candle_mask[0] = True
new_candle_mask[1:] = restant[1:] > restant[:-1]
candle_ids = np.cumsum(new_candle_mask)   # chaque ligne a un ID de bougie

print(f"Nombre de bougies : {candle_ids[-1]}")


@numba.njit(cache=True)
def _backtest_core(ask, bid, restant, side, imbalance, pred_PM, pred_DD, entry_signal, candle_ids):
    N = len(ask)
    nbr_ligne = 0
    benefice_total = 0.0
    perte_latent = 0.0
    perte_max = 0.0
    i = 0
    
    profits      = np.zeros(N, dtype=np.float64)   # profit de chaque trade

    while i < N:
        ep = ask[i]
        proba_i = entry_signal[i]
        pm = pred_PM[i]
        dd = pred_DD[i]
        rest_i = restant[i]
        imb_i = imbalance[i]
        side_i = int(side[i])  # conversion explicite

        rr_i = abs(pm / dd) if pm > 0 and dd != 0 else 0.0
        
        # ── Filtre d'entrée ──
        if not (rest_i > LIMIT_TIME_MIN
                and rest_i < LIMIT_TIME_MAX
                and ep     > MIN_ASK
                and ep     < MAX_ASK
                and proba_i > PROBA):
            i += 1
            continue

        candle_i = candle_ids[i]
        trailing_sl = 0.0
        j = i + 1
        while j < N:
            
           
            current_price = bid[j]
            drop_proba = entry_signal[i] - entry_signal[j]
            
            if candle_ids[j] != candle_i:
               
                profit = current_price * (1.0 / ep) - 1.0
                benefice_total += profit
                nbr_ligne += 1
                profits[nbr_ligne]      = profit
                
                if profit < 0.0:
                    perte_latent += profit
                    if perte_latent < perte_max:
                       perte_max = perte_latent
                else:
                    perte_latent = 0.0
                
                i = j
                break  # nouvelle bougie

            if int(side[j]) != side_i:
                j += 1
                continue
            
           # print ("JOJO : ",rr_i, "    ", RR, "    ", j, " ", i)

            # Mise à jour trailing
            if (drop_proba > TR_DROP or current_price > ep + TRAILING_SL * 2) and trailing_sl < current_price - TRAILING_SL:
                trailing_sl = current_price - TRAILING_SL

            profit = current_price * (1.0 / ep) - 1.0
            
            print ("infos dd : ", profit, " ", dd)

            # Condition de clôture
            if drop_proba    > LIMIT_DROP or current_price > 0.98 or (trailing_sl  > 0 and trailing_sl > current_price) or (restant[j]   < 30 and current_price > ep):
                
                benefice_total += profit
                nbr_ligne += 1
                profits[nbr_ligne]      = profit
                
                if profit < 0.0:
                    perte_latent += profit
                    if perte_latent < perte_max:
                        perte_max = perte_latent
                else:
                    perte_latent = 0.0
                
                i = j
                break
            j += 1
        i += 1# if j == i + 1 else 0   # avance correctement
        
        print ("conditions : ", i, "    ", j, " ",  j == i + 1)

    return nbr_ligne, benefice_total, perte_max, profits[:nbr_ligne], 


def run_backtest():
    """Wrapper Python autour du cœur Numba"""

    nbr_ligne, benefice_total, perte_max, profits = _backtest_core(
        ask, bid, restant, side, imbalance,
        pred_PM, pred_DD, entry_signal, candle_ids,)

    return {
        'profit_total':   benefice_total,
        'NBR Trades':     nbr_ligne,
        'enfoncement_max': perte_max,
    }


# ── Pré-compilation Numba (première exécution) ──
print("Compilation Numba (première fois seulement)...")
_backtest_core(
    ask[:100], bid[:100], restant[:100], side[:100], imbalance[:100],
    pred_PM[:100], pred_DD[:100], entry_signal[:100], candle_ids[:100],
)
print("✅ Numba compilé")

# ── Lancement ──
with open("backtest_results.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow([
        "benef", "enfoncement_max", "benef moyen", "NBR Trades"
    ])
    
nbr_ligne, benefice_total, perte_max, profits = _backtest_core(
        ask, bid, restant, side, imbalance,
        pred_PM, pred_DD, entry_signal, candle_ids,)
        
        
with open("backtest_results.csv", "a", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow([
        nbr_ligne,
        benefice_total,
        perte_max,
    ])

print ("RESULT : ", benefice_total, " ", nbr_ligne, "   ", profits)
print("✅ Optimisation terminée — résultats dans opti_results.csv")
pd.DataFrame(profits, columns=['profit']).to_csv("backtest_results.csv", index=False)  
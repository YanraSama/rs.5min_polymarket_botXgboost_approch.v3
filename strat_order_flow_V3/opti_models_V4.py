import pandas as pd
import numpy as np
import csv
import itertools
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numba

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
print ("LEN : N : ", N)
# ── Pré-calculer les indices de début de nouvelle bougie ──
# Quand restant[i+1] > restant[i] → nouvelle bougie
new_candle_mask = np.zeros(N, dtype=np.bool_)
new_candle_mask[0] = True
new_candle_mask[1:] = restant[1:] > restant[:-1]
candle_ids = np.cumsum(new_candle_mask)   # chaque ligne a un ID de bougie

print(f"Nombre de bougies : {candle_ids[-1]}")

param_grid = {
    'LIMIT_TIME_MIN': [30],
    'LIMIT_TIME_MAX': [200],
    'PROBA':          [0.65],
    'MIN_ASK':        [0.32],
    'MAX_ASK':        [0.82],
    'TR_DROP':        [0.00, 0.2, 0.4, 0.60],
    'LIMIT_DROP':     [0.00, 0.30, 0.60],
    'TRAILING_SL':    [0.10, 0.20],
    'RR':             [1.25]
}

@numba.njit(cache=True)
def _backtest_core(ask, bid, restant, side, imbalance,
                   pred_PM, pred_DD, entry_signal, candle_ids,
                   LIMIT_TIME_MIN, LIMIT_TIME_MAX, PROBA,
                   MIN_ASK, MAX_ASK, TR_DROP, LIMIT_DROP,
                   TRAILING_SL, RR):
    """
    Boucle de backtest compilée par Numba — ~100x plus rapide que Python pur
    Même logique que l'original mais en C compilé
    """
    N = len(ask)
    nbr_ligne          = 0
    benefice_total     = 0.0
    perte_latent       = 0.0
    perte_max          = 0.0
    profits_sum        = 0.0
    profits_sum_sq     = 0.0

    i = 0
    while i < N:

        ep       = ask[i]
        proba_i  = entry_signal[i]
        pm       = pred_PM[i]
        dd       = pred_DD[i]
        rest_i   = restant[i]
        imb_i    = imbalance[i]
        side_i   = side[i]

        # Risk reward
        # rr_i = abs(pm / dd) if pm > 0 and dd != 0 else 0.0

        # ── Filtre d'entrée ──
        if not (rest_i > LIMIT_TIME_MIN
                and rest_i < LIMIT_TIME_MAX
                and ep     > MIN_ASK
                and ep     < MAX_ASK
                and proba_i > PROBA):
            i += 1
            continue

        # ── Chercher la clôture dans les lignes suivantes ──
        candle_i     = candle_ids[i]
        trailing_sl  = 0.0
        closed        = False

        j = i + 1
        while j < N:
            
            current_price = bid[j]
            drop_proba    = entry_signal[i] - entry_signal[j]

            # Nouvelle bougie → sortie sans clôture
            if candle_ids[j] != candle_i:
               
                profit = current_price * (1.0 / ep) - 1.0
                benefice_total += profit
                nbr_ligne += 1
                
                if profit < 0.0:
                    perte_latent += profit
                    if perte_latent < perte_max:
                       perte_max = perte_latent
                else:
                    perte_latent = 0.0
                
                i = j
                break  # nouvelle bougie

            if side[j] != side_i:
                j += 1
                continue


            # Mise à jour trailing stop
            if ((drop_proba > TR_DROP or current_price > ep + TRAILING_SL * 2)
                    and trailing_sl < current_price - TRAILING_SL):
                trailing_sl = current_price - TRAILING_SL

            profit = current_price * (1.0 / ep) - 1.0

            # Condition de clôture
            if drop_proba    > LIMIT_DROP or current_price > 0.98 or (trailing_sl  > 0 and trailing_sl > current_price) or (restant[j]   < 30 and current_price > ep):

                benefice_total += profit
                nbr_ligne      += 1
                profits_sum    += profit
                profits_sum_sq += profit * profit

                if profit < 0.0:
                    perte_latent += profit
                    if perte_latent < perte_max:
                        perte_max = perte_latent
                else:
                    perte_latent = 0.0

                i = j   # ← avancer i à la clôture
                closed = True
                break

            j += 1

        i += 1

    profit_moyen = profits_sum / nbr_ligne if nbr_ligne > 0 else 0.0

    return nbr_ligne, benefice_total, profit_moyen, perte_max


def run_backtest(params):
    """Wrapper Python autour du cœur Numba"""

    nbr_ligne, benefice_total, profit_moyen, perte_max = _backtest_core(
        ask, bid, restant, side, imbalance,
        pred_PM, pred_DD, entry_signal, candle_ids,
        float(params['LIMIT_TIME_MIN']),
        float(params['LIMIT_TIME_MAX']),
        float(params['PROBA']),
        float(params['MIN_ASK']),
        float(params['MAX_ASK']),
        float(params['TR_DROP']),
        float(params['LIMIT_DROP']),
        float(params['TRAILING_SL']),
        float(params['RR']),
    )

    return {
        'profit_total':   benefice_total,
        'profit_moyen':   profit_moyen,
        'NBR Trades':     nbr_ligne,
        'enfoncement_max': perte_max,
        'TIME MAX':       params['LIMIT_TIME_MAX'],
        'TIME MIN':       params['LIMIT_TIME_MIN'],
        'MAX ASK':        params['MAX_ASK'],
        'MIN ASK':        params['MIN_ASK'],
        'proba':          params['PROBA'],
        'TR SL':          params['TRAILING_SL'],
        'TR DROP':        params['TR_DROP'],
        'TR LIMIT_DROP':  params['LIMIT_DROP'],
        'RR':             params['RR'],
    }


# ── Pré-compilation Numba (première exécution) ──
print("Compilation Numba (première fois seulement)...")
_backtest_core(
    ask[:100], bid[:100], restant[:100], side[:100], imbalance[:100],
    pred_PM[:100], pred_DD[:100], entry_signal[:100], candle_ids[:100],
    30.0, 120.0, 0.65, 0.32, 0.82, 0.0, 0.0, 0.10, 0.5
)
print("✅ Numba compilé")

# ── Génération des combinaisons ──
all_params = [
    dict(zip(param_grid.keys(), combo))
    for combo in itertools.product(*param_grid.values())
]
print(f"Nombre de combinaisons : {len(all_params)}")

# ── Lancement ──
with open("opti_results.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow([
        "benef", "enfoncement_max", "benef moyen", "NBR Trades",
        "TIME MAX", "TIME MIN", "MAX ASK", "MIN ASK",
        "proba", "TR SL", "TR DROP", "TR LIMIT_DROP", "RR"
    ])

for param_dict in tqdm(all_params, desc="Optimisation"):
    result = run_backtest(param_dict)

    with open("opti_results.csv", "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            result['profit_total'],
            result['enfoncement_max'],
            result['profit_moyen'],
            result['NBR Trades'],
            result['TIME MAX'],
            result['TIME MIN'],
            result['MAX ASK'],
            result['MIN ASK'],
            result['proba'],
            result['TR SL'],
            result['TR DROP'],
            result['TR LIMIT_DROP'],
            result['RR'],
        ])

print("✅ Optimisation terminée — résultats dans opti_results.csv")

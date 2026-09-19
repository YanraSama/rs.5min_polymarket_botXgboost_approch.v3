import pandas as pd
import numpy as np
import csv
import itertools
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numba

def optimisation(CSV_OPTI, BACKTEST_FILE, SEUIL):

    print("Chargement des données...")
    df = pd.read_csv(BACKTEST_FILE, sep=";", encoding="utf-8-sig")
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
    # pred_PM      = df["pred_PM"].values.astype(np.float32)
    # pred_DD      = df["pred_DD"].values.astype(np.float32)
    entry_signal = df["entry_signal"].values.astype(np.float32)

    N = len(ask)
    # print ("LEN : N : ", N)
    # ── Pré-calculer les indices de début de nouvelle bougie ──
    # Quand restant[i+1] > restant[i] → nouvelle bougie
    new_candle_mask = np.zeros(N, dtype=np.bool_)
    new_candle_mask[0] = True
    new_candle_mask[1:] = restant[1:] > restant[:-1]
    candle_ids = np.cumsum(new_candle_mask)   # chaque ligne a un ID de bougie

    print(f"Nombre de bougies : {candle_ids[-1]}")

    param_grid = {
        'LIMIT_TIME_MIN': [30, 60],
        'LIMIT_TIME_MAX': [90, 120, 200],
        'PROBA':          [SEUIL],
        'MIN_ASK':        [0.32],
        'MAX_ASK':        [0.75, 0.82],
        'TR_DROP':        [0.20, 0.40],
        'LIMIT_DROP':     [0.20, 0.40],
        'TRAILING_SL':    [0.05, 0.10, 0.20],
        'RR':             [1.25],
        'IMBA':           [0.05, 0.10],
        'STOP_LOSS':      [-0.20, -0.30, -0.40, -0.99]
    }

    @numba.njit(cache=True)
    def _backtest_core(ask, bid, restant, side, imbalance,
                       # pred_PM, pred_DD, 
                       entry_signal, candle_ids,
                       LIMIT_TIME_MIN, LIMIT_TIME_MAX, PROBA,
                       MIN_ASK, MAX_ASK, TR_DROP, LIMIT_DROP,
                       TRAILING_SL, RR, IMBA, STOP_LOSS):
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
        nbr_gagnant        = 0
        proba_gain         = 0
        toggle_side        = False
        side_open          = False 
        side_o             = 0       
        ep_oth             = 0 
        i = 0
        profit_oth         = 0
        maximmum           = 10
        minimmum           = -1
        balance            = 10 
        enfoncement_reel   = 0
        enf_max            = 0        
        
        while i < N:

            ep       = ask[i]
            proba_i  = entry_signal[i]
            # pm       = pred_PM[i]
            # dd       = pred_DD[i]
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
                    and proba_i > PROBA
                    and imb_i > IMBA):
                i += 1
                continue

            # ── Chercher la clôture dans les lignes suivantes ──
            candle_i     = candle_ids[i]
            trailing_sl  = 0.0
            closed       = False
            side_open    = True

            j = i + 1
            while j < N:
                
                current_price = bid[j]
                drop_proba    = entry_signal[i] - entry_signal[j]
                
                if side_open == True:

                    # Nouvelle bougie → sortie sans clôture
                    if candle_ids[j] != candle_i:
                       
                        profit = current_price * (1.0 / ep) - 1.0 if ep != 0 else 0
                        benefice_total += profit
                        nbr_ligne += 1
                        balance        += profit

                        if balance > maximmum:
                            
                            maximmum = balance
                            if minimmum > -1:
                                enfoncement_reel = (maximmum - minimmum) / maximmum
                            
                            minimmum = -1
                                
                        if balance < maximmum and (minimmum == -1 or minimmum > balance):
                            minimmum = balance
                            enfoncement_reel = (maximmum - minimmum) / maximmum
                
                        if enfoncement_reel > enf_max:
                            enf_max = enfoncement_reel

                        if balance > maximmum:
                            
                            maximmum = balance
                            if minimmum > -1:
                                enfoncement_reel = (maximmum - minimmum) / maximmum
                            
                            minimmum = -1
                                
                        if balance < maximmum and (minimmum == -1 or minimmum > balance):
                            minimmum = balance
                            enfoncement_reel = (maximmum - minimmum) / maximmum
                
                        if enfoncement_reel > enf_max:
                            enf_max = enfoncement_reel
                                                
                        if profit < 0.0:
                            perte_latent += profit
                            if perte_latent < perte_max:
                               perte_max = perte_latent
                        else:
                            perte_latent = 0.0
                        side_open   = False
                        i = j
                        
                if toggle_side == True:
                    # Nouvelle bougie → sortie sans clôture
                    if candle_ids[j] != candle_i:
                       
                        profit = current_price * (1.0 / ep_oth) - 1.0 if ep_oth != 0 else 0
                        benefice_total += profit
                        nbr_ligne += 1
                        balance        += profit

                        if balance > maximmum:
                            
                            maximmum = balance
                            if minimmum > -1:
                                enfoncement_reel = (maximmum - minimmum) / maximmum
                            
                            minimmum = -1
                                
                        if balance < maximmum and (minimmum == -1 or minimmum > balance):
                            minimmum = balance
                            enfoncement_reel = (maximmum - minimmum) / maximmum
                
                        if enfoncement_reel > enf_max:
                            enf_max = enfoncement_reel
                        
                        if profit_oth < 0.0:
                            perte_latent += profit_oth
                            if perte_latent < perte_max:
                               perte_max = perte_latent
                        else:
                            perte_latent = 0.0
                        toggle_side   = False
                        i = j

                if candle_ids[j] != candle_i:
                    break
                 

                if side[j] != side_i and toggle_side == False:
                    if not (restant[j] > LIMIT_TIME_MIN
                            and restant[j] < LIMIT_TIME_MAX
                            and ask[j]     > MIN_ASK
                            and ask[j]     < MAX_ASK
                            and entry_signal[j] > PROBA
                            and imbalance[j] > IMBA):
                        
                        j += 1
                        continue
                    else:
                        candle_j     = candle_ids[j]                        
                        ep_oth        = ask[j] 
                        toggle_side   = True  
                        side_o        = side[j]                  
                        trailing_sl_o = 0.0
                        proba_oth     = entry_signal[j]
                        j += 1
                        continue
                
                if side[j] == side_o and toggle_side == True:
                    drop_proba_oth    = proba_oth - entry_signal[j]
                    
                    # Mise à jour trailing stop
                    if ((drop_proba_oth > TR_DROP or current_price > ep_oth + TRAILING_SL * 2)
                            and trailing_sl < current_price - TRAILING_SL):
                        trailing_sl_o = current_price - TRAILING_SL
                        
                    profit_oth = current_price * (1.0 / ep_oth) - 1.0 if ep_oth != 0 else 0 

                    # Condition de clôture
                    if STOP_LOSS > profit_oth or drop_proba_oth > LIMIT_DROP or current_price > 0.98 or (trailing_sl_o  > 0 and trailing_sl_o > current_price) or (restant[j]   < 5 and current_price > ep_oth):

                        benefice_total += profit_oth
                        nbr_ligne      += 1
                        profits_sum    += profit_oth
                        profits_sum_sq += profit_oth * profit_oth
                        balance        += profit

                        if balance > maximmum:
                            
                            maximmum = balance
                            if minimmum > -1:
                                enfoncement_reel = (maximmum - minimmum) / maximmum
                            
                            minimmum = -1
                                
                        if balance < maximmum and (minimmum == -1 or minimmum > balance):
                            minimmum = balance
                            enfoncement_reel = (maximmum - minimmum) / maximmum
                
                        if enfoncement_reel > enf_max:
                            enf_max = enfoncement_reel
                        
                        if profit_oth < 0.0:
                            perte_latent += profit_oth
                            if perte_latent < perte_max:
                                perte_max = perte_latent
                        else:
                            perte_latent = 0.0
                            nbr_gagnant += 1

                        i = j   # ← avancer i à la clôture
                        closed = True
                        toggle_side   = False
                        j += 1 
                        continue
                
                if side[j] != side_i:
                    j += 1
                    continue

                
                if side_open == True:
                
                    # Mise à jour trailing stop
                    if ((drop_proba > TR_DROP or current_price > ep + TRAILING_SL * 2)
                            and trailing_sl < current_price - TRAILING_SL):
                        trailing_sl = current_price - TRAILING_SL

                    profit = current_price * (1.0 / ep) - 1.0 if ep != 0 else 0
                                                
                    # Condition de clôture
                    if STOP_LOSS > profit or drop_proba > LIMIT_DROP or current_price > 0.98 or (trailing_sl  > 0 and trailing_sl > current_price) or (restant[j]   < 5 and current_price > ep):

                        benefice_total += profit
                        nbr_ligne      += 1
                        profits_sum    += profit
                        profits_sum_sq += profit * profit
                        balance        += profit

                        if balance > maximmum:
                            
                            maximmum = balance
                            if minimmum > -1:
                                enfoncement_reel = (maximmum - minimmum) / maximmum
                            
                            minimmum = -1
                                
                        if balance < maximmum and (minimmum == -1 or minimmum > balance):
                            minimmum = balance
                            enfoncement_reel = (maximmum - minimmum) / maximmum
                
                        if enfoncement_reel > enf_max:
                            enf_max = enfoncement_reel

                        if profit < 0.0:
                            perte_latent += profit
                            if perte_latent < perte_max:
                                perte_max = perte_latent
                        else:
                            perte_latent = 0.0
                            nbr_gagnant += 1

                        i = j   # ← avancer i à la clôture
                        closed = True
                        side_open    = False
                    
                    j += 1
                if side_open == False and toggle_side == False:
                    break
                
                j += 1
                
            i += 1
        
        if (nbr_ligne > 0):
            proba_gain = nbr_gagnant/nbr_ligne
        
        profit_moyen = profits_sum / nbr_ligne if nbr_ligne > 0 else 0.0

        return nbr_ligne, benefice_total, profit_moyen, perte_max, proba_gain, enf_max


    def run_backtest(params):
        """Wrapper Python autour du cœur Numba"""

        nbr_ligne, benefice_total, profit_moyen, perte_max, wr, enf_max = _backtest_core(
            ask, bid, restant, side, imbalance,
            # pred_PM, pred_DD,
            entry_signal, candle_ids,
            float(params['LIMIT_TIME_MIN']),
            float(params['LIMIT_TIME_MAX']),
            float(params['PROBA']),
            float(params['MIN_ASK']),
            float(params['MAX_ASK']),
            float(params['TR_DROP']),
            float(params['LIMIT_DROP']),
            float(params['TRAILING_SL']),
            float(params['RR']),
            float(params['IMBA']),
            float(params['STOP_LOSS']),
        )

        return {
            'profit_total':   benefice_total,
            'win_rate':       wr,
            'profit_moyen':   profit_moyen,
            'NBR Trades':     nbr_ligne,
            'enfoncement_max':perte_max,
            'dd_max':         enf_max,
            'TIME MAX':       params['LIMIT_TIME_MAX'],
            'TIME MIN':       params['LIMIT_TIME_MIN'],
            'MAX ASK':        params['MAX_ASK'],
            'MIN ASK':        params['MIN_ASK'],
            'proba':          params['PROBA'],
            'TR SL':          params['TRAILING_SL'],
            'TR DROP':        params['TR_DROP'],
            'TR LIMIT_DROP':  params['LIMIT_DROP'],
            'RR':             params['RR'],
            'IMBA':           params['IMBA'],
            'STOP_LOSS':      params['STOP_LOSS'],
        }


    # ── Pré-compilation Numba (première exécution) ──
    print("Compilation Numba (première fois seulement)...")
    _backtest_core(
        ask[:100], bid[:100], restant[:100], side[:100], imbalance[:100],
        # pred_PM[:100], pred_DD[:100], 
        entry_signal[:100], candle_ids[:100],
        30.0, 120.0, 0.65, 0.32, 0.82, 0.0, 0.0, 0.10, 0.5, 0.1, -0.30
    )
    print("✅ Numba compilé")

    # ── Génération des combinaisons ──
    all_params = [
        dict(zip(param_grid.keys(), combo))
        for combo in itertools.product(*param_grid.values())
    ]
    # print(f"Nombre de combinaisons : {len(all_params)}")

    # ── Lancement ──
    with open(CSV_OPTI, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "benef", "win rate", "enfoncement_max", "dd_max", "benef moyen", "NBR Trades",
            "TIME MAX", "TIME MIN", "MAX ASK", "MIN ASK",
            "proba", "TR SL", "TR DROP", "TR LIMIT_DROP", "RR", "IMBA", "STOP_LOSS"
        ])

    for param_dict in tqdm(all_params, desc="Optimisation"):
        result = run_backtest(param_dict)

        with open(CSV_OPTI, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([
                result['profit_total'],
                result['win_rate'],
                result['enfoncement_max'],
                result['dd_max'],
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
                result['IMBA'],
                result['STOP_LOSS'],
            ])

    print(f"✅ Optimisation terminée — résultats dans {CSV_OPTI}")

    # df = pd.read_csv(CSV_OPTI, sep=";", encoding="utf-8-sig", low_memory=False)
    
    # print ("Max value : ", df.idxmax(), " VV  ", df['benef'].idxmax(0))
    
    # index_features = df['benef'].idxmax(0)
    # enfoncement_max= df["enfoncement_max"].values.astype(np.float32)
    # benef_moyen    = df["benef moyen"].values.astype(np.float32)
    # NBR_trades     = df["NBR Trades"].values.astype(np.float32)
    # benef          = df["benef"].values.astype(np.float32)
    # time_max       = df["TIME MAX"].values.astype(np.float32)
    # time_min       = df["TIME MIN"].values.astype(np.float32)
    # max_ask        = df["MAX ASK"].values.astype(np.float32)
    # min_ask        = df["MIN ASK"].values.astype(np.float32)
    # proba          = df["proba"].values.astype(np.float32)
    # tr_sl          = df["TR SL"].values.astype(np.float32)
    # tr_drop        = df["TR DROP"].values.astype(np.float32)
    # tr_limit_drop  = df["TR LIMIT_DROP"].values.astype(np.float32)
    
    # print ("les features d'entrée : ", index_features, "    ", benef[index_features], " ", time_max[index_features], "  ", time_min[index_features], "  ", max_ask[index_features], "   ", min_ask[index_features], "   ", proba[index_features], "   ", tr_sl[index_features], "   ", tr_drop[index_features], "   ", tr_limit_drop[index_features])
    

# optimisation("opti_results.csv", "backtest_models_V3.csv", 0.7)
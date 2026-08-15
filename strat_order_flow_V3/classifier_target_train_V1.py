import pandas as pd
import numpy as np
import pandas as pd
import itertools
import numpy as np
from tqdm import tqdm   # Barre de progression

def classifier_target_vectorized(CSV_FILE):

    print("Chargement des données...")

    df = pd.read_csv(CSV_FILE, sep=";", encoding="utf-8-sig")

    print(f"Nombre de lignes chargées : {len(df)}")

    # Conversion side en numérique
    # df["side"] = df["side"].map({"BUY": 1, "SELL": 0})

    # Conversion des colonnes numériques
    df["side"] = df["side"].apply(
        lambda x: 1 if str(x).strip() == "BUY"
                  else 0 if str(x).strip() == "SELL"
                  else int(float(x)) if str(x).strip() in ["0", "1", "0.0", "1.0"]
              else np.nan
)
    # Créer les colonnes si elles n'existent pas
    for col in ["target", "benef_max", "drawdown_max"]:
        if col not in df.columns:
            df[col] = np.nan

    # Trouver les lignes où les targets sont vides
    mask_empty = (
        df["target"].isna() #| 
        # df["benef_max"].isna() | 
        # df["drawdown_max"].isna()
    )
    
    empty_indices = df.index[mask_empty].tolist()
    
    if len(empty_indices) == 0:
            print("✅ Toutes les lignes sont déjà remplies. Rien à faire.")
            return df["target"].values, df["benef_max"].values, df["drawdown_max"].values  
    
    """
    Version O(n) — utilise numpy pour trouver
    le prochain event de fin de bougie ou win
    """
    n            = len(df)
    target       = np.zeros(n, dtype=np.int8)
    benef_max = np.zeros(n, dtype=np.float32)
    drawdown_max = np.zeros(n, dtype=np.float32)

    ask       = df["ask"].values.astype(np.float32)
    bid       = df["bid"].values.astype(np.float32)
    restant   = df["restant"].values.astype(np.float32)
    side      = df["side"].values.astype(np.int8)
    imbalance = df["imbalance"].values.astype(np.float32)
    # Par défaut : NaN (pas 0)
    target = np.full(n, np.nan)
    target    = df["target"].values.astype(np.float32)

    # ── Identifier les débuts de nouvelles bougies ──
    # Quand restant[i+1] > restant[i] → nouvelle bougie
    new_candle = np.where(np.diff(restant, prepend=0) > 0)[0]
    
    # Pour chaque ligne, trouver l'index de fin de sa bougie
    candle_end = np.searchsorted(new_candle, np.arange(n), side='right')
    candle_end = np.where(candle_end < len(new_candle), 
                          new_candle[np.minimum(candle_end, len(new_candle)-1)], 
                          n)

    print("Calcul vectorisé des targets... ", empty_indices)
    
    
    for i in tqdm(empty_indices):
        end   = int(candle_end[i])
        s     = side[i]
        ep    = ask[i]

        # Slice numpy de la bougie courante
        future_bid     = bid[i+1:end]
        future_restant = restant[i+1:end]
        future_side    = side[i+1:end]

        if len(future_bid) == 0:
            continue
        
        if ep > 0.82 or ep < 0.32 or restant[i] > 200 or restant[i] < 30:
            continue

        # Masque même side
        same_side = future_side == s

        # Condition win 1 : bid > 0.98
        win1 = same_side & (future_bid > 0.98)

        # Condition win 2 : restant < 30 ET bid > entry_price
        win2 = same_side & (future_restant < 30) & (future_bid > ep)

        win_mask = win1 | win2
        
        if win_mask.any():
            target[i] = 1
        else :
            target[i] = 0
            
        # if len(future_bid) > 0 and same_side.any():
                    # future_bid_same = future_bid[same_side]
                    # if ep > 0 and len(future_bid_same) > 0:
                        # benef_max[i] = np.max(future_bid_same) * (1 / ep) - 1
                        # drawdown_max[i] = np.min(future_bid_same) * (1 / ep) - 1
                    
    # print(f"\n=== RÉSULTATS ===")
    # print(f"Wins (target=1) : {target.sum():6d} ({target.mean()*100:.1f}%)")
    # print(f"Loses (target=0): {(target==0).sum():6d} ({(1-target.mean())*100:.1f}%)")                    
                    
    # ── Ajouter au df et sauvegarder ──
    df["target"]       = target
    # df["benef_max"]    = benef_max
    # df["drawdown_max"] = drawdown_max
    
    
    df.to_csv(CSV_FILE, 
              sep=";", index=False, encoding="utf-8-sig")
    print(f"✅ Sauvegardé : {CSV_FILE}") 
    
    return target, benef_max, drawdown_max

# classifier_target_vectorized("learning_live_v3.csv")

# ── Calcul target ──
# target, benef_max, drawdown_max = classifier_target_vectorized(df)

# print(f"\n=== RÉSULTATS ===")
# print(f"Wins (target=1) : {target.sum():6d} ({target.mean()*100:.1f}%)")
# print(f"Loses (target=0): {(target==0).sum():6d} ({(1-target.mean())*100:.1f}%)")

# ── Ajouter au df et sauvegarder ──
# df["target"]       = target
# df["benef_max"]    = benef_max
# df["drawdown_max"] = drawdown_max
# df.to_csv("data_model_classifier_V1_with_target.csv", 
          # sep=";", index=False, encoding="utf-8-sig")
# print("✅ Sauvegardé : data_model_classifier_V1_with_target.csv")

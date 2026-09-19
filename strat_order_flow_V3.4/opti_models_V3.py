import pandas as pd
import numpy as np
import csv
import pandas as pd
import itertools
import numpy as np
from tqdm import tqdm   # Barre de progression

print("Chargement des données...")

df = pd.read_csv("backtest_infos.csv", sep=";", encoding="utf-8-sig")

print(f"Nombre de lignes chargées : {len(df)}")

# Conversion des colonnes numériques
numeric_cols = ['side', 'ask', 'bid', 'restant', 'imbalance', 'pred_PM', 'pred_DD', 'entry_signal']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
param_grid = {
    'LIMIT_TIME_MIN' : [30],
    'LIMIT_TIME_MAX' : [90, 120, 200],
    'PROBA': [0.65, 0.75, 0.85],
    'MIN_ASK': [0.32, 0.42],
    'MAX_ASK': [0.72, 0.82],
    'TR_DROP': [0.00, 0.60],
    'LIMIT_DROP': [0.00, 0.60],
    'TRAILING_SL': [0.10, 0.20],
    'RR': [0.50, 1, 1.5, 2]
}

# Paramètres
LIMIT_TIME_MIN = 30
LIMIT_TIME_MAX = 120
MIN_ASK = 0.32
MAX_ASK = 0.82
PROBA = 0.85
TR_DROP = 0.6
LIMIT_DROP = 0.6
TRAILING_SL = 0.2
RR = 2

def run_backtest(params):

    nbr_ligne = 0
    benefice_total_reel = 0
    trades = []
    perte_latent = 0
    perte_max    = 0

    for i in range(len(df)):
       
        row = df.iloc[i]
        entry_price = float(row['ask'])
        proba = float(row['entry_signal'])
        pred_PM = float(row['pred_PM'])
        pred_DD = float(row['pred_DD'])
        
        risk_reward = abs(pred_PM / pred_DD) if pred_PM > 0 else 0
    
        if risk_reward > params['RR'] and float(row['restant']) > params['LIMIT_TIME_MIN'] and float(row['restant']) < params['LIMIT_TIME_MAX'] and entry_price > params['MIN_ASK'] and entry_price < params['MAX_ASK'] and float(row['imbalance']) > 0.00 and proba > params['PROBA'] :
           
           side  = row['side'] 
           
           future_rows = df.iloc[i+1:]  # limite à 100 lignes suivantes pour vitesse
           
           trailling_sl = 0
           
           for j, future in future_rows.iterrows():
                              
               if float(future['side']) != side:
                   continue
               
               current_price = float(future['bid'])
               drop_proba = float(row['entry_signal']) - float(future['entry_signal'])
               
               trailling_sl = current_price - params['TRAILING_SL'] if (drop_proba > params['TR_DROP'] or current_price > entry_price + params['TRAILING_SL']*2) and trailling_sl < current_price - params['TRAILING_SL'] else trailling_sl
               
               if drop_proba > params['LIMIT_DROP'] or current_price > 0.98 or trailling_sl > current_price or (float(future['restant']) < 30 and current_price > entry_price):
                   
                   benefice_total_reel += current_price * (1/entry_price) - 1
                   nbr_ligne = nbr_ligne + 1
                   profit = current_price * (1/entry_price) - 1
                   trades.append(profit)
                   perte_latent = perte_latent + profit if profit < 0.00 else 0
                   perte_max = perte_latent if perte_latent < perte_max else perte_max
                   i = j
                   break
                   
    print(f"\n=== RÉSULTATS BACKTEST ===")
    print(f"Nombre de trades exécutés : {nbr_ligne}")
    print(f"Profit total : {benefice_total_reel:.4f}")
    print(f"Profit moyen par trade : {np.mean(trades):.4f}" if trades else "Aucun trade") 
    print(f"enfoncement max : ", perte_max) 

    return {
            'params': params,
            'trades': trades,
            'profit_total': benefice_total_reel,
            'profit_moyen': benefice_total_reel / nbr_ligne if nbr_ligne > 0 else 0,
            'NBR Trades' : nbr_ligne,
            'enfoncement_max' : perte_max, 
            'TIME MAX': params['LIMIT_TIME_MAX'], 
            'TIME MIN' : params['LIMIT_TIME_MIN'],
            'MAX ASK' :  params['MAX_ASK'],
            'MIN ASK' :  params['MIN_ASK'],
            'proba' : params['PROBA'],
            'TR SL' : params['TRAILING_SL'],
            'TR DROP' : params['TR_DROP'],
            'TR LIMIT_DROP' : params['LIMIT_DROP'],
        } 

with open("opti_results.csv", "a", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")  # ← point-virgule
    writer.writerow([
        "benef" , "enfoncement_max", "benef moyen", "NBR Trades", "TIME MAX", "TIME MIN", "MAX ASK", "MIN ASK", "proba", "TR SL", "TR DROP", "TR LIMIT_DROP"
    ])        
  
for params in tqdm(list(itertools.product(*param_grid.values())), desc="Optimisation"):
    param_dict = dict(zip(param_grid.keys(), params))
    result = run_backtest(param_dict)
    
    with open("opti_results.csv", "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")  # ← point-virgule
        writer.writerow([
            result['profit_total'], result['enfoncement_max'], result['profit_moyen'], result['NBR Trades'], result['TIME MAX'], result['TIME MIN'], result['MAX ASK'], result['MIN ASK'], result['proba'], result['TR SL'], result['TR DROP'], result['TR LIMIT_DROP']
        ])
  
# Sauvegarde des résultats
#pd.DataFrame(trades, columns=['profit']).to_csv("backtest_results.csv", index=False)               
           
    
    
    
    
    import pandas as pd
import numpy as np
import csv
import pandas as pd
import itertools
import numpy as np
from tqdm import tqdm   # Barre de progression

print("Chargement des données...")

df = pd.read_csv("backtest_infos.csv", sep=";", encoding="utf-8-sig")

print(f"Nombre de lignes chargées : {len(df)}")

# Conversion des colonnes numériques
numeric_cols = ['side', 'ask', 'bid', 'restant', 'imbalance', 'pred_PM', 'pred_DD', 'entry_signal']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
param_grid = {
    'LIMIT_TIME_MIN' : [30],
    'LIMIT_TIME_MAX' : [90, 120, 200],
    'PROBA': [0.65, 0.75, 0.85],
    'MIN_ASK': [0.32, 0.42],
    'MAX_ASK': [0.72, 0.82],
    'TR_DROP': [0.00, 0.60],
    'LIMIT_DROP': [0.00, 0.60],
    'TRAILING_SL': [0.10, 0.20],
    'RR': [0.50, 1, 1.5, 2]
}

# Paramètres
LIMIT_TIME_MIN = 30
LIMIT_TIME_MAX = 120
MIN_ASK = 0.32
MAX_ASK = 0.82
PROBA = 0.85
TR_DROP = 0.6
LIMIT_DROP = 0.6
TRAILING_SL = 0.2
RR = 2

def run_backtest(params):

    nbr_ligne = 0
    benefice_total_reel = 0
    trades = []
    perte_latent = 0
    perte_max    = 0

    for i in range(len(df)):
       
        row = df.iloc[i]
        entry_price = float(row['ask'])
        proba = float(row['entry_signal'])
        pred_PM = float(row['pred_PM'])
        pred_DD = float(row['pred_DD'])
        
        risk_reward = abs(pred_PM / pred_DD) if pred_PM > 0 else 0
    
        if risk_reward > params['RR'] and float(row['restant']) > params['LIMIT_TIME_MIN'] and float(row['restant']) < params['LIMIT_TIME_MAX'] and entry_price > params['MIN_ASK'] and entry_price < params['MAX_ASK'] and float(row['imbalance']) > 0.00 and proba > params['PROBA'] :
           
           side  = row['side'] 
           
           future_rows = df.iloc[i+1:]  # limite à 100 lignes suivantes pour vitesse
           
           trailling_sl = 0
           
           for j, future in future_rows.iterrows():
                              
               if float(future['side']) != side:
                   continue
               
               current_price = float(future['bid'])
               drop_proba = float(row['entry_signal']) - float(future['entry_signal'])
               
               trailling_sl = current_price - params['TRAILING_SL'] if (drop_proba > params['TR_DROP'] or current_price > entry_price + params['TRAILING_SL']*2) and trailling_sl < current_price - params['TRAILING_SL'] else trailling_sl
               
               if drop_proba > params['LIMIT_DROP'] or current_price > 0.98 or trailling_sl > current_price or (float(future['restant']) < 30 and current_price > entry_price):
                   
                   benefice_total_reel += current_price * (1/entry_price) - 1
                   nbr_ligne = nbr_ligne + 1
                   profit = current_price * (1/entry_price) - 1
                   trades.append(profit)
                   perte_latent = perte_latent + profit if profit < 0.00 else 0
                   perte_max = perte_latent if perte_latent < perte_max else perte_max
                   i = j
                   break
                   
    print(f"\n=== RÉSULTATS BACKTEST ===")
    print(f"Nombre de trades exécutés : {nbr_ligne}")
    print(f"Profit total : {benefice_total_reel:.4f}")
    print(f"Profit moyen par trade : {np.mean(trades):.4f}" if trades else "Aucun trade") 
    print(f"enfoncement max : ", perte_max) 

    return {
            'params': params,
            'trades': trades,
            'profit_total': benefice_total_reel,
            'profit_moyen': benefice_total_reel / nbr_ligne if nbr_ligne > 0 else 0,
            'NBR Trades' : nbr_ligne,
            'enfoncement_max' : perte_max, 
            'TIME MAX': params['LIMIT_TIME_MAX'], 
            'TIME MIN' : params['LIMIT_TIME_MIN'],
            'MAX ASK' :  params['MAX_ASK'],
            'MIN ASK' :  params['MIN_ASK'],
            'proba' : params['PROBA'],
            'TR SL' : params['TRAILING_SL'],
            'TR DROP' : params['TR_DROP'],
            'TR LIMIT_DROP' : params['LIMIT_DROP'],
        } 

with open("opti_results.csv", "a", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")  # ← point-virgule
    writer.writerow([
        "benef" , "enfoncement_max", "benef moyen", "NBR Trades", "TIME MAX", "TIME MIN", "MAX ASK", "MIN ASK", "proba", "TR SL", "TR DROP", "TR LIMIT_DROP"
    ])        
  
for params in tqdm(list(itertools.product(*param_grid.values())), desc="Optimisation"):
    param_dict = dict(zip(param_grid.keys(), params))
    result = run_backtest(param_dict)
    
    with open("opti_results.csv", "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")  # ← point-virgule
        writer.writerow([
            result['profit_total'], result['enfoncement_max'], result['profit_moyen'], result['NBR Trades'], result['TIME MAX'], result['TIME MIN'], result['MAX ASK'], result['MIN ASK'], result['proba'], result['TR SL'], result['TR DROP'], result['TR LIMIT_DROP']
        ])
  
# Sauvegarde des résultats
#pd.DataFrame(trades, columns=['profit']).to_csv("backtest_results.csv", index=False)               
           
    
    
    
    
    
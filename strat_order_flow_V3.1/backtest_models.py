import pandas as pd
import joblib
import numpy as np
"""
# Charger le CSV
df = pd.read_csv("backtest_infos.csv", sep=";")

# Mélange aléatoire puis sélection de 40 000 lignes
df_sample = df.sample(n=40000, random_state=42)

# Sauvegarde
df_sample.to_csv("backtest_infos_sample.csv",
                 sep=";",
                 index=False)

print("Nombre de lignes :", len(df_sample
"""

"""
import pandas as pd
import joblib


"""
# ================= Chargement =================

df = pd.read_csv("entrainement_xboost_V3.csv",
                 sep=";",
                 encoding="utf-8-sig")

feature_cols = [
    'side', 'restant', 'variation', 'spread',
    'imbalance', 'delta_imbalance', 'ratio_flow',
    'moyenne_ratio', 'slope', 'tendance', 'whale',
    'imbalance_flow', 'flow_momentum', 
    'price_momentum', 'time_pressure', 'stake_var',
    # ── Features enrichies ──
    'imb_x_tend', 'flow_strength',
    'momentum_score', 'rr_theorique',
    'flow_acceleration', 'imbalance_acceleration',
]

# df['imbalance_flow'] = df['imbalance'] * df['ratio_flow']
# df['flow_momentum'] = df['delta_imbalance'] * df['restant']
# df['ask_distance'] = abs(df['ask'] - 0.5)
# df['price_momentum'] = df['variation'] * df['imbalance']
# df['time_pressure'] = df['restant'] / 300
# df['stake_var'] = (df['price'] - df['stake']) / df['stake']

# df["imb_x_tend"]     = df["imbalance"]  * df["tendance"]
# df["flow_strength"]  = df["ratio_flow"].abs() * df["moyenne_ratio"].abs()
# df["momentum_score"] = df["ratio_flow"] * df["moyenne_ratio"] * df["tendance"]
# df["rr_theorique"]   = (1.0 - df["ask"]) / df["ask"].clip(0.01)
# ✅ Fix — np.where opère ligne par ligne
# df["flow_acceleration"] = np.where(
   # df["restant"].shift(2) > df["restant"],          # condition
   # df["ratio_flow"] - df["ratio_flow"].shift(2),    # si vrai
   # 0.00                                              # si faux (nouvelle bougie)
# )

# df["imbalance_acceleration"] = np.where(
   # df["restant"].shift(2) > df["restant"],
   # df["imbalance"] - df["imbalance"].shift(2),
   # 0.00
# )
# ================= Conversion =================

# df["side"] = df["side"].map({
    # "BUY":1,
    # "SELL":0
# })


                 
X = df[feature_cols]

# ================= Chargement modèles =================

#reg_profit = joblib.load("xgboost_regressor_dyna_benef.pkl")
# reg_PM = joblib.load("xgboost_regressor_benef_max.pkl")
# reg_DD = joblib.load("xgboost_regressor_dd.pkl")
#reg_TTP = joblib.load("xgboost_regressor_ttp.pkl")
#reg_TTPDD = joblib.load("xgboost_regressor_ttp_DD.pkl")
#reg_EV = joblib.load("xgboost_regressor_ev.pkl")
regressor_proba = joblib.load("xgboost_model_v3_target.pkl")

# ================= Prédictions =================

#df["pred_profit"] = reg_profit.predict(X)

# df["pred_PM"] = reg_PM.predict(X)

# df["pred_DD"] = reg_DD.predict(X)

#df["pred_TTP"] = reg_TTP.predict(X)

#df["pred_TTP_DD"] = reg_TTPDD.predict(X)

#df["pred_EV"] = reg_EV.predict(X)

df["entry_signal"] =regressor_proba.predict_proba(X)[:, 1]

# ================= Sauvegarde =================

df.to_csv(
    "entrainement_xboost_V3.csv",
    sep=";",
    index=False,
    encoding="utf-8-sig"
)

print("Prédictions terminées.")

import asyncio
import ccxt
from binance import BinanceSocketManager, AsyncClient
import threading
import websocket
import time
import json  # Pour convertir la string clobTokenIds en vraie liste
import requests
from py_clob_client_v2.client import ClobClient
from py_clob_client_v2 import ClobClient
from py_clob_client_v2.clob_types import ApiCreds
from py_clob_client_v2.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
from py_clob_client_v2.order_builder.constants import BUY, SELL
from web3 import Web3
from eth_account import Account
import requests as req
from dotenv import load_dotenv
import requests, time, hmac, hashlib
import sqlite3
import websockets
import pandas as pd
import shutil

from py_clob_client_v2.clob_types import MarketOrderArgs

from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType

import time
from datetime import datetime

import csv
import os

import socket
import pprint
import traceback

import sys
import numpy as np

import time as time_module

from collections import deque
import copy  # ← Ajoute cet import en haut

import ssl
import httpx

from xgboost_learning_machine_V3 import learning_data_classifier
from xgboost_learning_machine_regressor_V3 import learning_data_regressor
from classifier_target_train_V1 import classifier_target_vectorized
from backtest_models import add_signal
from opti_models_V4 import optimisation

#--------------------------------------------------------------------------------------------------------------------------------------------

from py_clob_client_v2 import (
    ApiCreds,
    AssetType,
    BalanceAllowanceParams,
    ClobClient,
    OrderArgs,
    OrderType,
    PartialCreateOrderOptions,
    Side,
    SignatureTypeV2,
)

import joblib

regressor_proba = joblib.load("xgboost_model_v3_target.pkl")
calibrator = joblib.load("xgboost_model_v3_target_calibrator.pkl")

# regressor_maxbenef = joblib.load("xgboost_regressor_benef_max.pkl")
# regressor_maxdd = joblib.load("xgboost_regressor_dd.pkl")

#--------------------------------------------------------------------------------------------------------------------------------------------
# Augmenter les timeouts TCP au niveau OS

TICK_VAR            = 1
IMBALANCE_LIMIT     = 0.08
DELTA_IMB_ENTRY     = 0.06
DELTA_IMB_EXIT      = -0.1
SPREAD_LIMIT        = 0.045
PROFIT_MIN          =  0.20

CSV_FILE            = "order_flow_v3.csv"
CSV_LIVE            = "learning_live_v3.csv"
CSV_BACKTEST        = "backtest_models_V3.csv"
CSV_OPTI            = "opti_results.csv"
CSV_LIVE_TRAIN      = "learning_live_train.csv"
CSV_BACKTEST_TRAIN  = "backtest_models_V3_train.csv"
BUDGET              = 1
RR                  = 2

params = {
    "LIMIT_TIME_MAX":  200,
    "LIMIT_TIME_MIN":  30,
    "STOP_LOSS":      -0.30,
    "MAX_ASK":         0.82,
    "MIN_ASK":         0.32,
    "PROBA":           0.70,
    "TL_LEVEL":        0.20,
    "TR_DROP":         0.20,
    "DROP_LIMIT":      0.60,
    "IMBALANCE":       0.10,
    "STOP_LOSS":       -0.30,
}


FENETRE_COURTE  = 5    # secondes — momentum immédiat
FENETRE_LONGUE  = 20   # secondes — tendance
WHALE_SEUIL     = 3.5    # x fois la moyenne pour être "whale"
WHALE_BONUS     = 0.18 # bonus whale dans le score

#--------------------------------------------------------------------------------------------------------------------------------------------
load_dotenv()
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
FUNDER_ADDRESS = os.getenv("FUNDER_ADDRESS")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ================= CONFIG =================
HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet

signal_event = threading.Event()


binance = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET'),
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'},
})

#--------------------------------------------------------------------------------------------------------------------------------------------

# Derive L2 API credentials
temp_client = ClobClient(
    host=HOST,
    key=PRIVATE_KEY,
    chain_id=CHAIN_ID,
    
)
api_creds = temp_client.create_or_derive_api_key()
temp_client.set_api_creds(api_creds)

print("✅ Client en signature_type=3")
print("Adresse :", temp_client.get_address())

# Initialisation du client
client = ClobClient(
    host = HOST,
    chain_id=CHAIN_ID,
    key=PRIVATE_KEY,
    creds=api_creds,   #  CRUCIAL
    signature_type=SignatureTypeV2.POLY_1271,
    funder=FUNDER_ADDRESS
)

GAMMA_API = "https://gamma-api.polymarket.com"

print("Adresse client (sig=3) :", client.get_address())

client.update_balance_allowance(
    BalanceAllowanceParams(
        asset_type=AssetType.COLLATERAL,
        # signature_type n'est pas toujours nécessaire ici
    )
)
print("✅ Balance sync réussie")

# Récupération de la balance (sous forme dict)
balance_data = client.get_balance_allowance(
    BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
)
#--------------------------------------------------------------------------------------------------------------------------------------------

def get_order_details_safe(order_id):
    """Récupère les détails avec retry car l'ordre peut ne pas être indexé immédiatement"""
    retries=3
    delay=0.5
    
    for i in range(retries):
        try:
            details = client.get_order(order_id)
            if details is not None:
                return details
        except Exception as e:
            print(f"⚠️ Tentative {i+1}/{retries} échouée : {e}")
        
        time_module.sleep(delay)
    
    return None

#--------------------------------------------------------------------------------------------------------------------------------------------

def get_temps_restant():
#    """Retourne le temps restant en secondes dans la bougie M5 en cours"""
    
    now = int(time.time())
    
    # Début de la bougie en cours (arrondi au multiple de 300s)
    window_start = (now // 300) * 300
    window_end   = window_start + 300
    
    restant = window_end - now
    
    minutes = restant // 60
    secondes = restant % 60
    
    #print(f"⏳ Temps restant : {minutes}m {secondes:02d}s ({restant}s)")
    
    return restant
#--------------------------------------------------------------------------------------------------------------------------------------------

toggle_poly_btc = False

async def stream():
    
    global btc_price
    global main_signal
    global variation
    global toggle_poly_btc
    old_tick = None
    restant = get_temps_restant()
    
    main_signal = 0
    variation   = 0
    toggle_poly_btc = False
    
    while True:   # ← AJOUTER
        try:

            if filtre_horaire() == False:
                time.sleep(3)
                continue
                
            async with websockets.connect(
                "wss://stream.binance.com:9443/ws/btcusdc@trade",
                close_timeout=0
            ) as ws:

                async for raw in ws:

                    msg = json.loads(raw)
                    
                    btc_price = float(msg["p"])
                    toggle_poly_btc = True
                    
                    if old_tick != None:
                        variation = (float(msg["p"]) - old_tick) / old_tick * 100

                    if old_tick == None:
                        old_tick = float(msg["p"])
                        
                    elif (btc_price - old_tick) / old_tick * 100 > TICK_VAR:
                        print ("Impulsion Haussière : ", (float(msg["p"]) - old_tick) / old_tick * 100, "  DELTA IMBALANCE : ", delta_yes_imbalance, "   ", delta_no_imbalance, "  IMBALANCE : ", yes_imbalance, "  ", no_imbalance)
                        main_signal = 1
                        signal_event.set()

                    
                    elif (btc_price - old_tick) / old_tick * 100 < -TICK_VAR:
                        print ("Impulsion Baissière : ", (float(msg["p"]) - old_tick) / old_tick * 100, "  DELTA IMBALANCE : ", delta_yes_imbalance, "   ", delta_no_imbalance, "  IMBALANCE : ", yes_imbalance, "  ", no_imbalance) 
                        main_signal = -1
                        signal_event.set()
                    
                    else :
                        main_signal = 0
                    
                    #print("Price LIVE : ", msg["p"], "$     ", old_tick, "  ", (float(msg["p"]) - old_tick) / old_tick * 100)
                    old_tick = float(msg["p"])
                    
                    #print("Signal : ", main_signal, "   ", btc_price)
        
        except Exception as e:
            print(f"❌ WS coupé : {e} — reconnexion dans 3s...")
            traceback.print_exc()
            toggle_poly_btc = False
            await asyncio.sleep(3)
    
#----------------------------------------------------------------------------------------------------------------------------

main_signal = 0
btc_price   = 0
variation   = 0

def start_ws():
    loop = asyncio.new_event_loop()      # ← event loop dédié
    asyncio.set_event_loop(loop)
    loop.run_until_complete(stream())    # ← run_until_complete au lieu de asyncio.run()


"""threading.Thread(
    target=start_ws,
    daemon=True
).start()"""
    
#----------------------------------------------------------------------------------------------------------------------------
# ── Historique des trades avec timestamp ──
yes_trades_history = deque(maxlen=200)  # garde les 200 derniers trades
no_trades_history  = deque(maxlen=200)

yes_token = None
no_token  = None

yes_buy_flow  = 0.0
yes_sell_flow = 0.0
no_buy_flow   = 0.0
no_sell_flow  = 0.0
yes_delta     = 0.0
no_delta      = 0.0
yes_last_size = 0.0
no_last_size  = 0.0
    
# ── Flag de réinitialisation ──
ws_needs_refresh = threading.Event()

imbalance_history_yes  = deque(maxlen=200)
imbalance_history_no   = deque(maxlen=200)

yes_spread  = 0.0  
no_spread   = 0.0     
yes_ask = 0.0
yes_bid = 0.0
no_ask  = 0.0
no_bid  = 0.0  
yes_imbalance = 0.0
no_imbalance  = 0.0
delta_yes_imbalance = 0.0
delta_no_imbalance  = 0.0
old_yes_imbalance   = 0.0
old_no_imbalance    = 0.0
bid_volume = 0.0
ask_volume = 0.0
toggle_poly_ws = False

yes_token = None
no_token  = None

async def poly_ws():     
    
    global yes_bid, yes_ask, no_bid, no_ask
    global yes_spread, no_spread
    global yes_imbalance, no_imbalance
    global delta_yes_imbalance, delta_no_imbalance
    global yes_buy_flow, yes_sell_flow
    global no_buy_flow, no_sell_flow
    global yes_delta, no_delta
    global yes_last_size, no_last_size    
    global yes_token, no_token

    global yes_token, no_token
    global yes_spread
    global no_spread
    global yes_imbalance
    global no_imbalance
    global delta_yes_imbalance
    global delta_no_imbalance
    global bid_volume, ask_volume
    global actual_yes_volume, actual_no_volume    
    global toggle_poly_ws
    
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Origin":     "https://polymarket.com",
    }
    
    toggle_poly_ws = False

    while True:
        try:
            
            if filtre_horaire() == False:
                time.sleep(3)
                continue

            # Attendre que les tokens soient disponibles
            while yes_token is None or no_token is None:
                await asyncio.sleep(0.5)
                print("RAW:", yes_token, "  ", no_token)
            current_yes = yes_token
            current_no  = no_token

            print(f"🔄 Connexion WS avec tokens : {current_yes[:8]}... / {current_no[:8]}...")

            async with websockets.connect(
                uri, 
                additional_headers=headers, 
                open_timeout=15,
                ping_interval=20,    # ← envoie un ping toutes les 20s
                ping_timeout=60,     # ← attend 60s la réponse (au lieu du défaut 20s)
                close_timeout=10,
            ) as ws:
                
                print("✅ WebSocket Polymarket connecté")
                toggle_poly_ws = True

                await ws.send(json.dumps({
                    "assets_ids": [current_yes, current_no],
                    "type":       "market"
                }))

                async for raw in ws:
                    
                    # ── Nouvelle bougie → on ferme et reconnecte ──
                    if yes_token != current_yes or no_token != current_no:
                        print("🔄 Nouveaux tokens détectés — reconnexion WS...")
                        
                        # Reset des prix
                        yes_ask = 0.0
                        yes_bid = 0.0
                        no_ask  = 0.0
                        no_bid  = 0.0
                        break   # ← sort du async for → reconnecte avec nouveaux tokens

                    data   = json.loads(raw)
                    events = data if isinstance(data, list) else [data]
                    bid_volume = 0.0
                    ask_volume = 0.0
                    
                    for event in events:
                        if not isinstance(event, dict):
                            continue
                     
                        event_type = event.get("event_type", "")

                        # ── PRICE CHANGE — trades exécutés ──
                        if event_type == "price_change":

                            price_changes = event.get("price_changes", [])
                            timestamp     = int(event.get("timestamp", 0))
                            
                            for change in price_changes:
                                asset_id = change.get("asset_id")
                                price      = float(change.get("price",    0))
                                size       = float(change.get("size",     0))
                                side       = change.get("side", "")
                                best_bid   = float(change.get("best_bid", 0))
                                best_ask   = float(change.get("best_ask", 0))

                                if asset_id == current_yes:
                                    # Mettre à jour bid/ask
                                    if best_bid > 0: yes_bid = best_bid
                                    if best_ask > 0: yes_ask = best_ask
                                    yes_spread    = yes_ask - yes_bid
                                    yes_last_size = size
                                    
                                    # ── Ajouter à l'historique avec timestamp ──
                                    add_trade(yes_trades_history, side, size, price, timestamp)

                                    # ── Détection whale ──
                                    whale = detect_whale_trade(yes_trades_history,WHALE_SEUIL)
                                    # if whale:
                                        # print(f"🐋 WHALE YES détectée : {whale['size']:.0f} tokens "
                                              # f"({whale['ratio']:.1f}x moy) | {whale['side']} @ {whale['price']:.3f}")
                  
                                elif asset_id == current_no:
                                    if best_bid > 0: no_bid = best_bid
                                    if best_ask > 0: no_ask = best_ask
                                    no_spread    = no_ask - no_bid
                                    no_last_size = size
                                                                    
                                    add_trade(no_trades_history, side, size, price, timestamp)

                                    whale = detect_whale_trade(no_trades_history, WHALE_SEUIL)
                                    # if whale:
                                        # print(f"🐋 WHALE NO détectée : {whale['size']:.0f} tokens "
                                              # f"({whale['ratio']:.1f}x moy) | {whale['side']} @ {whale['price']:.3f}")
                                    
                                yes_spread = yes_ask - yes_bid
                                no_spread  = no_ask - no_bid
                                                                

                                #print(f"📈 YES {yes_ask:.3f}/{yes_bid:.3f} | 📉 NO {no_ask:.3f}/{no_bid:.3f}", " SPREAD : YES ", yes_spread, "     NO ", no_spread, " IMBALANCE : ", yes_imbalance, "   ", no_imbalance)
                                #print("DELTA : ", delta_yes_imbalance, "    " ,delta_no_imbalance)

                        # ── CARNET D'ORDRES — mise à jour book ──
                        else:
                            bids = event.get("bids", [])
                            asks = event.get("asks", [])
                            asset_id = event.get("asset_id")

                            if bids and asks:
                                best_bid = max((float(b["price"]) for b in bids), default=0)
                                best_ask = min((float(a["price"]) for a in asks), default=0)

                                if best_bid <= 0 or best_ask <= 0 or best_bid >= best_ask:
                                    continue

                                if asset_id == current_yes:
                                    old_yes_imbalance   = yes_imbalance
                                    bid_vol             = sum(float(x["size"]) for x in bids[:3])
                                    ask_vol             = sum(float(x["size"]) for x in asks[:3])
                                    yes_bid             = best_bid
                                    yes_ask             = best_ask
                                    yes_imbalance       = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
                                    delta_yes_imbalance = yes_imbalance - old_yes_imbalance
                                    imbalance_history_yes.append(yes_imbalance)

                                elif asset_id == current_no:
                                    old_no_imbalance   = no_imbalance
                                    bid_vol            = sum(float(x["size"]) for x in bids[:3])
                                    ask_vol            = sum(float(x["size"]) for x in asks[:3])
                                    no_bid             = best_bid
                                    no_ask             = best_ask
                                    no_imbalance       = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
                                    delta_no_imbalance = no_imbalance - old_no_imbalance
                                    imbalance_history_no.append(no_imbalance)

                                yes_spread = yes_ask - yes_bid
                                no_spread  = no_ask  - no_bid

        except Exception as e:
            print(f"❌ WS coupé : {e} — reconnexion dans 3s...")
            traceback.print_exc()
            toggle_poly_ws = False
            await asyncio.sleep(3)
                                     

def start_poly_ws():
    loop = asyncio.new_event_loop()      # ← event loop dédié
    asyncio.set_event_loop(loop)
    loop.run_until_complete(poly_ws())   # ← run_until_complete au lieu de asyncio.run()

#-----------------------------------------------------------------------------------------------------------------------------


def add_trade(history, side, size, price, timestamp):
    """Ajoute un trade à l'historique"""
    history.append({
        "side":      side,
        "size":      size,
        "price":     price,
        "timestamp": timestamp,
    })

def get_flow_window(history, window_seconds):
    """
    Calcule le flux net sur une fenêtre glissante récente
    Plus pertinent que le cumul depuis le début de bougie
    """
    now = time.time() * 1000  # ms
    
    buy_flow  = 0.0
    sell_flow = 0.0
    
    # On fait une copie pour éviter les mutations pendant l'itération
    history_copy = list(history)
    
    for trade in history_copy:
        age = (now - trade["timestamp"]) / 1000  # secondes
        if age > window_seconds:
            continue
        
        if trade["side"] == "BUY":
            buy_flow += trade["size"]
        else:
            sell_flow += trade["size"]
    
    total = buy_flow + sell_flow
    delta = buy_flow - sell_flow
    ratio = delta / total if total > 0 else 0
    
    return {
        "buy_flow":  buy_flow,
        "sell_flow": sell_flow,
        "delta":     delta,
        "ratio":     ratio,
        "total":     total,
    }


def detect_whale_trade(history, seuil_multiplicateur):
    """
    Détecte un trade anormalement gros par rapport
    à la moyenne récente — signal fort d'impact prix
    """
    if len(history) < 10:
        return None
    
    history_copy = list(history)  # Copie pour éviter l'erreur
    
    sizes = [t["size"] for t in history_copy]
    moyenne = sum(sizes[:-1]) / len(sizes[:-1])  # moyenne sans le dernier
    dernier = sizes[-1]
    
    if dernier > moyenne * seuil_multiplicateur and moyenne > 0:
        #print ("Infos Whale : moyenne ", moyenne, "    Baleine ", dernier)
        return {
            "size":   dernier,
            "ratio":  dernier / moyenne,
            "side":   history_copy[-1]["side"],
            "price":  history_copy[-1]["price"],
        }
    
    return None

#-----------------------------------------------------------------------------------------------------------------------------

ratio_archive_yes  = deque(maxlen=200)
ratio_archive_no   = deque(maxlen=200)

trend_archive_yes  = deque(maxlen=200)
trend_archive_no   = deque(maxlen=200)

def get_flow_signal(side, fenetre_courte, fenetre_longue):
    """
    Signal combinant :
    - Flux récent (5s) — momentum immédiat
    - Flux moyen (20s) — tendance de fond
    - Whale detection — impact ponctuel fort
    """
    history = yes_trades_history if side == "YES" else no_trades_history
    
    if len(history) < 5:
        return False, 0, None, 0, 0, 0
        


    flow_court = get_flow_window(history, fenetre_courte)
    flow_long  = get_flow_window(history, fenetre_longue)
    whale      = detect_whale_trade(history, WHALE_SEUIL)

    # ── Score pondéré ──
    score = (
        flow_court["ratio"] * 0.50 +   # momentum immédiat = poids fort
        flow_long["ratio"]  * 0.30 +   # tendance de fond
        (yes_imbalance if side == "YES" else no_imbalance) * 0.20
    )

    # ── Bonus whale dans le bon sens ──
    whale_bonus = 0
    if whale:
        if whale["side"] == "BUY":
            whale_bonus = +WHALE_BONUS
        else:
            whale_bonus = -WHALE_BONUS

    score_final = score + whale_bonus

    if side == "YES":
        ratio_archive_yes.append(score_final)
    
        if len(ratio_archive_yes) < 10:
            return False, 0, None, 0, 0, 0
        archive_list  = list(ratio_archive_yes)

    elif side == "NO":
        ratio_archive_no.append(score_final)
    
        if len(ratio_archive_no) < 10:
            return False, 0, None, 0, 0, 0
        archive_list  = list(ratio_archive_no)
        
    # ── Moyenne des 10 DERNIERS scores (les plus récents) ──
    
    dix_derniers  = archive_list[-10:]
    moyenne_ratio = sum(dix_derniers) / len(dix_derniers)

    # ── Tendance — la moyenne monte-t-elle ou descend-elle ? ──
    if len(archive_list) >= 20:
        dix_precedents      = archive_list[-20:-10]
        moyenne_precedente  = float(sum(dix_precedents)) / len(dix_precedents)
        tendance            = moyenne_ratio - moyenne_precedente
        
        if side == "YES":
            trend_archive_yes.append(tendance)
            if len(trend_archive_yes) < 20:
                return False, 0, None, 0, 0, 0
            trend_list = list(trend_archive_yes)
            moyenne_tendance = sum(trend_list[-5:]) / len(trend_list[-5:])
            
            x = np.arange(20)
            y = np.array(trend_list[-20:])

            slope = np.polyfit(x, y, 1)[0]

                
        elif side == "NO":
            trend_archive_no.append(tendance)
            if len(trend_archive_no) < 20:
                return False, 0, None, 0, 0, 0
            trend_list = list(trend_archive_no)
            moyenne_tendance = sum(trend_list[-5:]) / len(trend_list[-5:])     
            
            x = np.arange(20)
            y = np.array(trend_list[-20:])

            slope = np.polyfit(x, y, 1)[0]
            
    else:
        tendance = 0
        return False, 0, None, 0, 0, 0
    

    details = {
        "flow_court": flow_court,
        "flow_long":  flow_long,
        "whale":      whale,
        "score":      score_final,
    }
    # print(f"📊 Flow {side} | Court(5s): {flow_court['ratio']:+.2f} | "
           # f"Long(20s): {flow_long['ratio']:+.2f} | "
           # f"Whale: {'✅' if whale else '❌'} | Score: {score_final:+.3f}")
   
    # print("SCORE : ", score_final, "    Moyenne : ", moyenne_ratio, "   TREND:  ", tendance, "  moyenne : ", moyenne_tendance, "  slope  ", slope, "   ", side) 

    return score_final > 0, score_final, details, tendance, moyenne_ratio, slope
                
#--------------------------------------------------------------------------------------------------------------------------

def cloturer_position(trade_took, token_id, size):
    """Revend les tokens au prix bid actuel"""
    
    try:
                    
        # Prix légèrement sous le bid pour exécution rapide
        prix_vente = yes_bid if trade_took["side"] == "BUY" else no_bid
        
        print(f"🔄 Clôture : vente de {size} tokens à {prix_vente:.2f}$")
        
        order = client.create_market_order(
            MarketOrderArgs(
                token_id = trade_took["token"],
                amount   = size,
                side     = SELL,
                price    = 0.01,
            ),
            options = PartialCreateOrderOptions(
                tick_size = "0.01",
                neg_risk  = False,
            ),
        )
        response = client.post_order(order, OrderType.FOK)
        
        print("✅ Position clôturée !")

        return response
        
    except Exception as e:
        print(f"❌ Erreur clôture : {e}")
        traceback.print_exc()

        return None

#--------------------------------------------------------------------------------------------------------------------------
def send_telegram(message):

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("Erreur Telegram :", e)
 
#--------------------------------------------------------------------------------------------------------------------------

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")  # ← point-virgule
            writer.writerow([
                "heure_entry", "temps", "side",
                "stake", "prix_exec", "price", "imbalance", "delta_imbalance", "variation", "ask_entry", "ratio_entry", "moyenne_ratio_entry", "tendance_entry", "slope_entry", "whale", "resultat", "BENEF", "DD", "PM","EV", "TTP", "TTP_DD", "target"
                
            ])

#------------------------------------------------------------------------------------------------------------------------

def maj_csv(trade_took):

    with open(CSV_FILE, "a", newline="", encoding="utf-8-sig") as f:
        
        # row = {
            # 'heure_entry'         : trade_took["heure_entry"],
            # 'temps'               : trade_took["temps"],
            # 'side'                : trade_took["side"],
            # 'stake'               : trade_took["stake"],
            # 'prix_exec'           : trade_took["prix_exec"],
            # 'price'               : trade_took["price"],
            # 'imbalance'           : trade_took["imbalance"],
            # 'delta_imbalance'     : trade_took["delta_imbalance"],
            # 'variation'           : trade_took ["variation"],
            # 'ask_entry'           : trade_took ["ask_entry"],
            # 'ratio_entry'         : trade_took["ratio_entry"],
            # 'moyenne_ratio_entry' : trade_took["moyenne_ratio_entry"],
            # 'tendance_entry'      : trade_took["tendance_entry"],
            # 'slope_entry'         : trade_took["slope_entry"],
            # 'whale'               : trade_took["whale"],
            # 'resultat'            : trade_took["resultat"],
            # 'target'              : 1 if trade_took["resultat"] > 0 else 0
        # }
        # df = pd.DataFrame([row])
        # df.to_csv("training_data.csv", mode='a', header=not os.path.exists("training_data.csv"), index=False)        
        
        writer = csv.writer(f, delimiter=";")  # ← point-virgule
        writer.writerow([
            trade_took["heure_entry"], trade_took["temps"], trade_took["side"], trade_took["stake"], 
            trade_took["prix_exec"], trade_took["price"], trade_took["imbalance"],  trade_took["delta_imbalance"], trade_took ["variation"], trade_took["ask_entry"], trade_took["ratio_entry"],  
            trade_took["moyenne_ratio_entry"], trade_took["tendance_entry"], trade_took["slope_entry"],  trade_took["whale"],  trade_took["resultat"],  trade_took["benef"],
            trade_took["dd"],  trade_took["PM"],  trade_took["ev"], trade_took["ttp"], trade_took["ttp_dd"], 1 if trade_took["resultat"] > 0 else 0
        ])
            
#---------------------------------------------------------------------------------------------------------------------------

def get_current_btc_5m_slug():
    now = int(time.time())                    # timestamp actuel
    window_start = (now // 300) * 300         # arrondi au multiple de 300 secondes (5 min)
    slug = f"btc-updown-5m-{window_start}"
    return slug
    
#--------------------------------------------------------------------------------------------------------------------------
   
def get_btc_5m_market():
    slug = get_current_btc_5m_slug()
    
    # Méthode 1 : via events (souvent plus complet)
    response = requests.get(f"{GAMMA_API}/events/slug/{slug}")
    if response.status_code == 200:
        event = response.json()
        # L'event contient souvent les markets
        if event.get("markets"):
            market = event["markets"][0]   # pour les marchés Up/Down, il y en a généralement 2 (Up et Down)
            return market
        
    return None
    
#---------------------------------------------------------------------------------------------------------------------------

def get_position_size_clob(token_id):
    """Via balance_allowance du client CLOB"""
    try:
        balance = client.get_balance_allowance(
            BalanceAllowanceParams(
                asset_type = AssetType.CONDITIONAL,
                token_id   = token_id,
            )
        )
        size = float(balance.get("balance", 0)) / 1_000_000  # 6 décimales
        print(f"📊 Position CLOB : {size} tokens")
        return size

    except Exception as e:
        print(f"❌ Erreur balance CLOB : {e}")
        return 0

#-----------------------------------------------------------------------------------------------------------------------------

ratio_buy = 0 
ratio_sell = 0
moyenne_yes = 0
moyenne_no = 0
slope_yes  = 0
slope_no  = 0
tendance_yes = 0
tendance_no = 0

acc_flow_yes = 0
acc_imba_yes = 0
acc_flow_no = 0
acc_imba_no = 0

csv_lock    = False

def thread_autoregulation():
    
    last_window_log = 0
    global csv_lock
    global regressor_proba
    global calibrator

    while True:
        try:
            
            if filtre_horaire() == False:
                time.sleep(3)
                continue    

            window = (int(time.time()) // 3600) * 3600
            
            print ("LOG WINDOWS :" , window, "  ", last_window_log)
                                  
            if window != last_window_log:
                new_period = True
                last_window_log = window
            else:
                new_period = False
                
            if new_period == True:
                
                csv_lock = True
                garder_donnees_recentes(CSV_LIVE, 300_000)
                garder_donnees_recentes(CSV_BACKTEST, 1_000_000)
                shutil.copy2(CSV_LIVE, CSV_LIVE_TRAIN)
                shutil.copy2(CSV_BACKTEST, CSV_BACKTEST_TRAIN)
                csv_lock = False           
                
                a, b, void_lign = classifier_target_vectorized(CSV_LIVE_TRAIN)
                a, b, void_lign = classifier_target_vectorized(CSV_BACKTEST_TRAIN)
                
                df = pd.read_csv(CSV_LIVE_TRAIN, sep=";", encoding="utf-8-sig", low_memory=False)
                df = df.dropna(subset=["target"]).reset_index(drop=True)
                
                df.to_csv(CSV_LIVE_TRAIN, 
                              sep=";", index=False, encoding="utf-8-sig")

                auccurancy, aucc, confusion, importances, accuracy_score, seuil = learning_data_classifier(CSV_LIVE_TRAIN, "xgboost_model_v3_target.pkl")
                # if void_lign == True:
                    # learning_data_regressor(CSV_LIVE, "test_xgboost_regressor_benef_max.pkl", "benef_max")
                    # learning_data_regressor(CSV_LIVE, "test_xgboost_regressor_benef_dd.pkl", "drawdown_max")
                    
                csv_lock = True
                shutil.copy2(CSV_LIVE_TRAIN, CSV_LIVE)
                shutil.copy2(CSV_BACKTEST_TRAIN, CSV_BACKTEST)
                csv_lock = False
            
                regressor_proba = joblib.load("xgboost_model_v3_target.pkl")
                calibrator = joblib.load("xgboost_model_v3_target_calibrator.pkl")
                
                benef, enfoncement_max, benef_moyen, NBR_trades, time_max, time_min, max_ask, min_ask, proba_value, tr_sl, tr_drop, tr_limit_drop, stop_loss_value, imbalance_value, wr = backtest(seuil)
                
                new_period = False
                clock = 0
                no_last_flow = 0
                yes_last_flow = 0
                no_last_imba = 0
                yes_last_imba = 0  
                
                params["LIMIT_TIME_MAX"] = time_max
                params["LIMIT_TIME_MIN"] = time_min
                params["MAX_ASK"]        = max_ask
                params["MIN_ASK"]        = min_ask
                params["TL_LEVEL"]       = tr_sl
                params["PROBA"]          = proba_value
                params["TR_DROP"]        = tr_drop
                params["DROP_LIMIT"]     = tr_limit_drop
                params["STOP_LOSS"]      = stop_loss_value
                params["IMBALANCE"]      = imbalance_value
                
                af = pd.read_csv(CSV_BACKTEST, sep=";", encoding="utf-8-sig", low_memory=False)

                send_telegram(
                f"""
                Entrainement modele classifier 3.3: 

                NBR datas : {len(af)}
                Accuracy  : {auccurancy:.2f}
                AUC       : {aucc:.2f}

                🎯 Score de cohérence  : {accuracy_score}
                🎯 Matrice de Confusion    : {confusion}
                🛑 Best features : {importances.head(15)}
                
                Nouveaux backtest feat : 
                   bénef    {benef:.2f}$
                   win rate {wr:.2f}
                   max dd   {enfoncement_max:.2f}
                   gain moyen {benef_moyen:.2f}
                   nbr trades {NBR_trades}
                   max time {params["LIMIT_TIME_MAX"]:.2f}s
                   min time {params["LIMIT_TIME_MIN"]:.2f}s
                   max ask  {params["MAX_ASK"]:.2f}
                   max ask  {params["MIN_ASK"]:.2f}
                   proba    {params["PROBA"]:.2f}
                   tr sl    {params["TL_LEVEL"]:.2f}
                   tr drop  {params["TR_DROP"]:.2f}
                   drop Li. {params["DROP_LIMIT"]:.2f}
                   stop loss {params["STOP_LOSS"]:.2f}
                   imbalance {params["IMBALANCE"]:.2f}
                """ 
                )
        
        except Exception as e:
            print(f"❌ Erreur autorégulation : {e}")
            import traceback
            traceback.print_exc()

        time.sleep(60)    
    
def start_thread_autoregulation():
    loop = asyncio.new_event_loop()      # ← event loop dédié
    asyncio.set_event_loop(loop)
    loop.run_until_complete(thread_autoregulation())   # ← run_until_complete au lieu de asyncio.run()
    
async def log_trade_data():
    
    new_period = False
    last_window_log = 0
    clock = 0
    no_last_flow = 0
    yes_last_flow = 0
    no_last_imba = 0
    yes_last_imba = 0
    df = pd.read_csv(CSV_LIVE, sep=";", encoding="utf-8-sig", low_memory=False)
    
    if not os.path.exists(CSV_LIVE):
        with open(CSV_LIVE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")  # ← point-virgule
            writer.writerow([
                "side", "restant", "variation", "ask", "bid", "spread", "imbalance", "delta_imbalance", "ratio_flow", "moyenne_ratio", "slope", "tendance", "whale", "imbalance_flow", "flow_momentum", 
                "ask_distance", "price_momentum", "time_pressure", "stake_var", "imb_x_tend", "flow_strength", "momentum_score", "rr_theorique", "flow_acceleration", "imbalance_acceleration",
            ])
            
    while True:
        try:    
                 
            if filtre_horaire() == False:
                time.sleep(3)
                continue          
                
            if yes_ask > 0.00 and no_ask > 0.00 and btc_price > 0.00 and stake > 0.00 and yes_ask != no_ask and toggle_poly_ws == True and csv_lock == False:
                
                last_restant = restant
                acc_flow_yes = ratio_buy - yes_last_flow if clock > restant else 0.00
                acc_imba_yes = yes_imbalance - yes_last_imba if clock > restant else 0.00
                
                acc_flow_no = ratio_sell - no_last_flow if clock > restant else 0.00
                acc_imba_no = no_imbalance - no_last_imba if clock > restant else 0.00
            
                with open(CSV_BACKTEST, "a", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f, delimiter=";")
                  
                    # print("Infos BUY TK LOG features : ", yes_ask, "$   ", restant, "s ", len(df))
                    # print("Infos SELL TK LOG features : ", no_ask, "$   ", restant, "s ", len(df))
                    writer.writerow([
                        1,
                        restant,
                        save_variation,
                        yes_ask, 
                        yes_bid, 
                        yes_spread,
                        yes_imbalance,
                        delta_yes_imbalance,
                        ratio_buy,
                        moyenne_yes,
                        slope_yes,
                        tendance_yes,
                        trade_took ["whale"],
                        yes_imbalance*ratio_buy,
                        delta_yes_imbalance*restant,
                        abs(yes_ask-0.5),
                        save_variation * yes_imbalance,
                        restant / 300,       
                        (btc_price - stake) / btc_price,
                        yes_imbalance * tendance_yes,
                        abs(ratio_buy) * abs(moyenne_yes),
                        ratio_buy * moyenne_yes * tendance_yes,
                        (1 - yes_ask) / yes_ask if yes_ask >= 0.01 else (1 - yes_ask) / 0.01,
                        acc_flow_yes,
                        acc_imba_yes
                    ])
                    
                    writer.writerow([
                        0,
                        restant,
                        save_variation,
                        no_ask, 
                        no_bid,
                        no_spread,
                        no_imbalance,
                        delta_no_imbalance,
                        ratio_sell,
                        moyenne_no,
                        slope_no,
                        tendance_no,
                        trade_took ["whale"],
                        no_imbalance*ratio_sell,
                        delta_no_imbalance*restant,
                        abs(no_ask-0.5),
                        save_variation * no_imbalance,
                        restant / 300,       
                        (btc_price - stake) / btc_price,
                        no_imbalance * tendance_no,
                        abs(ratio_sell) * abs(moyenne_no),
                        ratio_sell * moyenne_no * tendance_no,
                        (1 - no_ask) / no_ask if no_ask >= 0.01 else (1 - no_ask) / 0.01,
                        acc_flow_no,
                        acc_imba_no
                    ])
                    
                with open(CSV_LIVE, "a", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f, delimiter=";")
                  
                    # print("Infos BUY TK LOG features : ", yes_ask, "$   ", restant, "s ", len(df))
                    # print("Infos SELL TK LOG features : ", no_ask, "$   ", restant, "s ", len(df))
                    writer.writerow([
                        1,
                        restant,
                        save_variation,
                        yes_ask, 
                        yes_bid, 
                        yes_spread,
                        yes_imbalance,
                        delta_yes_imbalance,
                        ratio_buy,
                        moyenne_yes,
                        slope_yes,
                        tendance_yes,
                        trade_took ["whale"],
                        yes_imbalance*ratio_buy,
                        delta_yes_imbalance*restant,
                        abs(yes_ask-0.5),
                        save_variation * yes_imbalance,
                        restant / 300,       
                        (btc_price - stake) / btc_price,
                        yes_imbalance * tendance_yes,
                        abs(ratio_buy) * abs(moyenne_yes),
                        ratio_buy * moyenne_yes * tendance_yes,
                        (1 - yes_ask) / yes_ask if yes_ask >= 0.01 else (1 - yes_ask) / 0.01,
                        acc_flow_yes,
                        acc_imba_yes
                    ])
                    
                    writer.writerow([
                        0,
                        restant,
                        save_variation,
                        no_ask, 
                        no_bid,
                        no_spread,
                        no_imbalance,
                        delta_no_imbalance,
                        ratio_sell,
                        moyenne_no,
                        slope_no,
                        tendance_no,
                        trade_took ["whale"],
                        no_imbalance*ratio_sell,
                        delta_no_imbalance*restant,
                        abs(no_ask-0.5),
                        save_variation * no_imbalance,
                        restant / 300,       
                        (btc_price - stake) / btc_price,
                        no_imbalance * tendance_no,
                        abs(ratio_sell) * abs(moyenne_no),
                        ratio_sell * moyenne_no * tendance_no,
                        (1 - no_ask) / no_ask if no_ask >= 0.01 else (1 - no_ask) / 0.01,
                        acc_flow_no,
                        acc_imba_no
                    ])
                    
                    clock = restant
                    no_last_flow = ratio_sell
                    yes_last_flow = ratio_buy
                    no_last_imba = no_imbalance
                    yes_last_imba = yes_imbalance                    
                    
            time.sleep(1) 
            
        except Exception as e:
            print(f"❌ WS coupé : {e} — reconnexion dans 3s...")
            traceback.print_exc()
            await asyncio.sleep(3)
            
def start_log_trade_data():
    loop = asyncio.new_event_loop()      # ← event loop dédié
    asyncio.set_event_loop(loop)
    loop.run_until_complete(log_trade_data())   # ← run_until_complete au lieu de asyncio.run()    

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    

def watchdog():
    global ws_thread, poly_ws_thread, log_trade_data_thread, thread_autoregulation_thread

    while True:
        if ws_thread is None or not ws_thread.is_alive():
            print("⚠️ Thread Binance mort — redémarrage...")
            ws_thread = threading.Thread(target=start_ws, daemon=True)
            ws_thread.start()

        if poly_ws_thread is None or not poly_ws_thread.is_alive():
            print("⚠️ Thread Polymarket mort — redémarrage...")
            poly_ws_thread = threading.Thread(target=start_poly_ws, daemon=True)
            poly_ws_thread.start()

        if log_trade_data_thread is None or not log_trade_data_thread.is_alive():
            print("⚠️ Thread Polymarket mort — redémarrage...")
            log_trade_data_thread = threading.Thread(target=start_log_trade_data, daemon=True)
            log_trade_data_thread.start()
            
        if thread_autoregulation_thread is None or not thread_autoregulation_thread.is_alive():
            print("⚠️ Thread Polymarket mort — redémarrage...")
            thread_autoregulation_thread = threading.Thread(target=start_thread_autoregulation, daemon=True)
            thread_autoregulation_thread.start()            

        time.sleep(10)
          
def backtest(seuil):
    
    add_signal(CSV_BACKTEST)
    optimisation(CSV_OPTI, CSV_BACKTEST, seuil)
    
    df = pd.read_csv(CSV_OPTI, sep=";", encoding="utf-8-sig", low_memory=False)
    
    
    for th in [0.75, 0.70, 0.65, 0.60]:
        if "win rate" in df.columns:
            df_filtered_1 = df[df["win rate"] > th].copy()
        else:
            print("⚠️ Colonne win_rate introuvable, aucune filtration appliquée")
            df_filtered_1 = df.copy()
       
        if "benef" in df.columns:
            df_filtered_2 = df_filtered_1[df_filtered_1["benef"] > 0.00].copy()
        else:
            print("⚠️ Colonne benef introuvable, aucune filtration appliquée")
            df_filtered_2 = df_filtered_1.copy()
            
        if "NBR Trades" in df.columns:
            df_filtered_3 = df_filtered_2[df_filtered_2["NBR Trades"] > 250].copy()
        else:
            print("⚠️ Colonne NBR Trades introuvable, aucune filtration appliquée")
            df_filtered_3 = df_filtered_2.copy()
            
        if "dd_max" in df.columns:
            df_filtered = df_filtered_3[df_filtered_3["dd_max"] < 0.60].copy()
        else:
            print("⚠️ Colonne dd_max introuvable, aucune filtration appliquée")
            df_filtered = df_filtered_3.copy()
            
        if len(df_filtered) == 0:
            print("❌ Aucune combinaison avec win_rate > 0.75")
            if th == 0.60:
                print("Aucun combinaison interressante")
                return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        else:
            break
    
    # df_filtered.to_csv("test.csv", 
              # sep=";", index=False, encoding="utf-8-sig")
    # print(f"✅ Sauvegardé : {"test"}") 
      
    # On réinitialise l'index pour éviter les problèmes
    df_filtered = df_filtered.reset_index(drop=True)
    
    index_features  = df_filtered['benef'].idxmax()
    enfoncement_max = df_filtered["enfoncement_max"].values.astype(np.float32)
    benef_moyen     = df_filtered["benef moyen"].values.astype(np.float32)
    NBR_trades      = df_filtered["NBR Trades"].values.astype(np.float32)
    benef           = df_filtered["benef"].values.astype(np.float32)
    time_max        = df_filtered["TIME MAX"].values.astype(np.float32)
    time_min        = df_filtered["TIME MIN"].values.astype(np.float32)
    max_ask         = df_filtered["MAX ASK"].values.astype(np.float32)
    min_ask         = df_filtered["MIN ASK"].values.astype(np.float32)
    proba           = df_filtered["proba"].values.astype(np.float32)
    tr_sl           = df_filtered["TR SL"].values.astype(np.float32)
    tr_drop         = df_filtered["TR DROP"].values.astype(np.float32)
    tr_limit_drop   = df_filtered["TR LIMIT_DROP"].values.astype(np.float32)
    stop_loss_value = df_filtered["STOP_LOSS"].values.astype(np.float32)
    imbalance_value = df_filtered["IMBA"].values.astype(np.float32)
    win_rate        = df_filtered["win rate"].values.astype(np.float32)
    
    print ("les features d'entrée : ", NBR_trades[index_features], "    ", win_rate[index_features], " ", benef[index_features], " ", time_max[index_features], "  ", time_min[index_features], "  ", max_ask[index_features], "   ", min_ask[index_features], "   ", proba[index_features], "   ", tr_sl[index_features], "   ", tr_drop[index_features], "   ", tr_limit_drop[index_features], " ", stop_loss_value[index_features], "   ", imbalance_value[index_features])

    return benef[index_features], enfoncement_max[index_features], benef_moyen[index_features], NBR_trades[index_features], time_max[index_features], time_min[index_features], max_ask[index_features], min_ask[index_features], proba[index_features], tr_sl[index_features], tr_drop[index_features], tr_limit_drop[index_features], stop_loss_value[index_features], imbalance_value[index_features],  win_rate[index_features]
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def polymarket_taker_fee(shares: float, price: float, fee_rate: float = 0.07) -> float:
    """
    Frais taker Polymarket (marchés crypto).
    fee = shares * fee_rate * p * (1 - p)
    """
    p = max(0.01, min(0.99, float(price)))
    return float(shares) * fee_rate * p * (1.0 - p) * 2
   

def garder_donnees_recentes(csv_file, max_lignes):
    """
    Garde uniquement les N dernières lignes
    = données les plus récentes
    """
    try:
        df = pd.read_csv(csv_file, sep=";", encoding="utf-8-sig",
                         low_memory=False)
        total = len(df)

        if total > max_lignes:
            df = df.tail(max_lignes).reset_index(drop=True)
            df.to_csv(csv_file, sep=";", index=False, encoding="utf-8-sig")
            print(f"✅ {csv_file} : {total} → {len(df)} lignes "
                  f"(supprimé {total - len(df)} lignes obsolètes)")
        else:
            print(f"ℹ️ {csv_file} : {total} lignes — pas de nettoyage nécessaire")

        return len(df)

    except Exception as e:
        print(f"❌ Erreur nettoyage {csv_file} : {e}")
        return 0   
    
def filtre_horaire():
    try:
        now = datetime.now()
        return True
        if int(now.strftime("%H")) >= 7 and int(now.strftime("%H")) < 21:
            # print ("Dans la tranche horaire : ",  now.strftime("%H"))
            return True
        else:
            # print ("En dehors de la tranche horaire : ",  now.strftime("%H"))
            return False
            
    except Exception as e:
        print(f"❌ Erreur fonction horaire : {e}")
        return 0             
            
# ── Lancement ──
ws_thread = threading.Thread(target=start_ws, daemon=True)
ws_thread.start()

poly_ws_thread = threading.Thread(target=start_poly_ws, daemon=True)
poly_ws_thread.start()

log_trade_data_thread = threading.Thread(target=start_log_trade_data, daemon=True)
log_trade_data_thread.start()

thread_autoregulation_thread = threading.Thread(target=thread_autoregulation, daemon=True)
thread_autoregulation_thread.start()

watchdog_thread = threading.Thread(target=watchdog, daemon=True)
watchdog_thread.start()
#-----------------------------------------------------------------------------------------------------------------------------
    

init_csv()
trade_took = {"token" : 0, "yes_token" : 0, "no_token" : 0, "trade": False, "cout": 0, "time" : 0,"date" : 0, "heure_entry" : 0, "heure_exit" : 0,"side" : "", "prix_exec" : 0, 
                "temps" : 0, "resultat" : 0, "price" : 0, "variation" : 0, "profit" : 0, "stake" : 0, "ask_entry" : 0, "bid_exit" : 0, "ratio_entry" : 0, "moyenne_ratio_entry" : 0, "tendance_entry" : 0, "slope_entry" : 0,
                "ratio" : 0, "imbalance" : 0, "moyenne_ratio" : 0, "tendance" : 0, "slope" : 0, "delta_imbalance" : 0, "whale" : 0, "proba" : 0, "gain" : 0, "ttp" : 0, "dd" : 0, "PM" : 0, "benef" : 0, "ev" : 0, "ttp_dd" : 0, "proba_entry": 0.0,}
time_stats = -1
size = 5   
last_window = None
stake = 0
must_sell = False
one_by_period = 0
restant= 0

while True:
    try:
        erreur = "" 
        restant = get_temps_restant()
        now = datetime.now()
        #btc_price = 0
        window = (int(time.time()) // 300) * 300
        save_signal = 0
        save_variation = 0
        
        if filtre_horaire() == False:
            time.sleep(3)
            continue
            
        
        if toggle_poly_btc == False:
            print ("deconnected ... ", toggle_poly_btc,"    ", toggle_poly_ws, "    ", btc_price)
            time.sleep(3)
            continue
                                
        # print ("log var : ", params["LIMIT_TIME_MAX"], "  ", params["LIMIT_TIME_MIN"], "    ",params["MAX_ASK"], "   ", params["MIN_ASK"], "   ", params["PROBA"], "     ", params["TL_LEVEL"], "  ", params["TR_DROP"], "   ", params["DROP_LIMIT"], " ", params["STOP_LOSS"], "   " ,params["IMBALANCE"])
                
       #------------------------------------------------------------------------------------------------------------------------------------------
        if window != last_window:
            
            # print("Nouvelle bougie 5m détectée")
            
            # Reset flux — nouvelle bougie = nouveau compteur
            yes_trades_history.clear()
            no_trades_history.clear()
            ratio_archive_yes.clear()
            ratio_archive_no.clear()
            trend_archive_yes.clear()
            trend_archive_no.clear()

            slug = f"btc-updown-5m-{window}"
            
            if trade_took ["trade"] == True:
                
                if trade_took["side"] == "BUY" and yes_bid > 0.9:
                    trade_took["resultat"] = trade_took["profit"]
                elif trade_took["side"] == "BUY" and yes_bid < 0.1:
                    trade_took["resultat"] = -trade_took["cout"]
     
                if trade_took["side"] == "SELL" and no_bid > 0.9:
                    trade_took["resultat"] = trade_took["profit"]
                elif trade_took["side"] == "SELL" and no_bid < 0.1:
                    trade_took["resultat"] = -trade_took["cout"]          

                trade_took ["heure_exit"]  = now.strftime("%H:%M:%S")  
                trade_took["trade"]        = False
                trade_took["price"]        = btc_price
                trade_took["bid_exit"] = no_bid if trade_took["side"] == "SELL" else yes_bid
                trade_took["imbalance"] = yes_imbalance if trade_took ["side"] == "BUY" else no_imbalance
                trade_took["moyenne_ratio"] = moyenne_yes if trade_took ["side"] == "BUY" else moyenne_no
                trade_took["tendance"] = tendance_yes if trade_took ["side"] == "BUY" else tendance_no
                trade_took["slope"] = slope_yes if trade_took["side"] == "BUY" else slope_no 
                maj_csv(trade_took)# cloture de position
                # print ("retranscription convergence : ", trade_took["resultat"], "$")
                
                send_telegram(
                    f"""
                🛑 POSITION CLOTURE - Convergence - order_flow_v3.3 : 

                    Side      : {trade_took['side']}
                    BTC Price : {btc_price}
                    Size      : {size}

                💰 Token    : {yes_bid if trade_took["side"] == "BUY" else no_bid:.2f} 
                💰 Coût     : {trade_took['cout']:.2f}
                🎯 Resultat : {trade_took['resultat']:.2f}
                """ 
                )
                
            klines = requests.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol":   "BTCUSDC",
                    "interval": "5m",
                    "limit":    1 + 1,
                }
            ).json()
            bougie_en_cours = klines[-1]
            stake = float(bougie_en_cours[1])
            # print ("Nouveau STAKE : ", stake)
            
            one_by_period =0
            
            #print(slug)
            
            last_window = window   

            # BTC 5mn
            market = get_btc_5m_market()
            
            if not market:
                print("❌ Marché non trouvé")
                time.sleep(10)
                continue
           
            question = market.get("question")
            clob_token_ids = market.get("clobTokenIds")   # liste de 2 strings : [Yes_token, No_token]
                
            if isinstance(clob_token_ids, str):
                try:
                    # Nettoyage et conversion de la string en vraie liste
                    clob_token_ids = json.loads(clob_token_ids.replace("'", '"'))
                except json.JSONDecodeError:
                    clob_token_ids = []   # en cas d'erreur     
            
            # Recherche marché -------------------------------------------------------------------------------------------------------------------#
            response = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"slug": get_current_btc_5m_slug()})
            
            market_data = response.json()[0]

            condition_id = market_data["conditionId"]
            
            clob_ids = market_data["clobTokenIds"]

            if isinstance(clob_ids, str):
                clob_ids = json.loads(clob_ids.replace("'", '"'))

            yes_token = clob_ids[0]
            no_token  = clob_ids[1]
            
            url = f"https://clob.polymarket.com/book?token_id={yes_token}"

            response = requests.get(url)
            
            # print ("new token : ", yes_token, "  ", no_token)
           

#--------------------------------------------------------------------------------------------------------------------------
        # if (yes_ask > 0 and no_ask > 0):
            # signal_buy, ratio_buy, details_yes_flow, tendance_yes, moyenne_yes, slope_yes = get_flow_signal("YES", FENETRE_COURTE, FENETRE_LONGUE)
            # signal_sell, ratio_sell, details_no_flow, tendance_no, moyenne_no, slope_no  = get_flow_signal("NO", FENETRE_COURTE, FENETRE_LONGUE)
            # time.sleep(1)
            # continue
#--------------------------------------------------------------------------------------------------------------------------

        signal_buy, ratio_buy, details_yes_flow, tendance_yes, moyenne_yes, slope_yes = get_flow_signal("YES", FENETRE_COURTE, FENETRE_LONGUE)
        #print ("Infos BUY flow : ",  signal_buy, "   ", ratio_buy)
        
   
        signal_sell, ratio_sell, details_no_flow, tendance_no, moyenne_no, slope_no = get_flow_signal("NO", FENETRE_COURTE, FENETRE_LONGUE)
        #print ("Infos SELL flow : ",  signal_sell, "   ", ratio_sell)            

        save_signal = main_signal
        save_variation = variation 
                
        if details_yes_flow == None or details_yes_flow["whale"] == None or details_yes_flow["whale"]["ratio"] == None:
            trade_took ["whale"] = 0
        else:
            trade_took["whale"] = details_yes_flow["whale"]["ratio"]
                        
        features_yes = [[
            1,
            restant,
            save_variation,
            yes_spread,
            yes_imbalance,
            delta_yes_imbalance,
            ratio_buy,
            moyenne_yes,
            slope_yes,
            tendance_yes,
            trade_took ["whale"],
            yes_imbalance*ratio_buy,
            delta_yes_imbalance*restant,
            save_variation * yes_imbalance,
            restant / 300,
            (btc_price - stake) / btc_price,
            yes_imbalance * tendance_yes,
            abs(ratio_buy) * abs(moyenne_yes),
            ratio_buy * moyenne_yes * tendance_yes,
            (1 - yes_ask) / yes_ask if yes_ask >= 0.01 else (1 - yes_ask) / 0.01,
            acc_flow_yes,
            acc_imba_yes
        ]]   

        proba = calibrator.predict(regressor_proba.predict_proba(features_yes)[:, 1])
        benef = 1#regressor_maxbenef.predict(features_yes)[0]
        dd    = 1#regressor_maxdd.predict(features_yes)[0]
        
        if details_no_flow == None or details_no_flow["whale"] == None or details_no_flow["whale"]["ratio"] == None:
            trade_took ["whale"] = 0
        else:
            trade_took["whale"] = details_no_flow["whale"]["ratio"]           
        
        features_no = [[
            0,
            restant,
            save_variation,
            no_spread,
            no_imbalance,
            delta_no_imbalance,
            ratio_sell,
            moyenne_no,
            slope_no,
            tendance_no,
            trade_took ["whale"],
            no_imbalance*ratio_sell,
            delta_no_imbalance*restant,
            save_variation * no_imbalance,
            restant / 300,
            (btc_price - stake) / btc_price,
            no_imbalance * tendance_no,
            abs(ratio_sell) * abs(moyenne_no),
            ratio_sell * moyenne_no * tendance_no,
            (1 - no_ask) / no_ask if no_ask >= 0.01 else (1 - no_ask) / 0.01,
            acc_flow_no,
            acc_imba_no
        ]]  


        proba_no = calibrator.predict(regressor_proba.predict_proba(features_no)[:, 1])
        benef_no = 1#regressor_maxbenef.predict(features_no)[0]
        dd_no    = 1#regressor_maxdd.predict(features_no)[0]
#---------------------------------------------------------------------------------------------------------------------------------------------------
        
        if trade_took["trade"] == True:   

            current_proba = proba if (trade_took["side"] == "BUY") else proba_no
            
            shares = BUDGET / price_token
            fee_entry = polymarket_taker_fee(shares, price_token)                        
         
            trade_took ["resultat"]   = (yes_bid*size) - trade_took["cout"] - fee_entry if trade_took ["side"] == "BUY" else (no_bid*size) - trade_took["cout"] - fee_entry
            
            proba_drop = trade_took["proba_entry"] - current_proba
            
            if (proba_drop > params["TR_DROP"] or (trade_took["side"] == "BUY" and (trade_took["resultat"] > 0.0 and  yes_bid >= trade_took["ask_entry"] + (params["TL_LEVEL"]*2)))) and trailing_stop < yes_bid-params["TL_LEVEL"]:
                trailing_stop = yes_bid-params["TL_LEVEL"]
                # print ("Trailing stop is moving :", trailing_stop)
            elif (proba_drop > params["TR_DROP"] or (trade_took["side"] == "SELL" and (trade_took["resultat"] > 0.0 and  no_bid >= trade_took["ask_entry"] + (yes_bid-params["TL_LEVEL"]*2)))) and trailing_stop < no_bid-yes_bid-params["TL_LEVEL"]:
                trailing_stop = no_bid-yes_bid-params["TL_LEVEL"]
                # print ("Trailing stop is moving :", trailing_stop)
     
            print ("Trailing SL : ",  trailing_stop, " ", proba_drop > params["TR_DROP"], "   ", (trade_took["side"] == "BUY" and (trade_took["resultat"] > 0.0 and  yes_bid >= trade_took["ask_entry"] + (params["TL_LEVEL"]*2)))
                if trade_took["side"] == "BUY" else (trade_took["side"] == "SELL" and (trade_took["resultat"] > 0.0 and  no_bid >= trade_took["ask_entry"] + (params["TL_LEVEL"]*2))), "  ",trailing_stop < yes_bid-params["TL_LEVEL"])
     
            print ("TEST MANA : ",  trade_took ["resultat"], "  ", trailing_stop, " ",yes_bid if trade_took["side"] == "BUY" else no_bid, "  ",  proba_drop, " ",current_proba, "   ", restant, "s")
                
                
            if params["STOP_LOSS"] > trade_took ["resultat"] or proba_drop > params["DROP_LIMIT"] or (trade_took["side"] == "BUY" and (trailing_stop >= yes_bid or yes_bid >= 0.98)) or (trade_took["side"] == "SELL" and (trailing_stop >= no_bid or no_bid >= 0.98)) or (restant < 30 and trade_took ["resultat"] > 0):
               
                # if cloturer_position(trade_took, trade_took["token"], size) == None :
                    # continue 
                        
                trade_took ["price"] = btc_price 
                trade_took ["heure_exit"] = now.strftime("%H:%M:%S")  
                trade_took["ratio"] = ratio_buy if trade_took["side"] == "BUY" else ratio_sell
                # trade_took ["resultat"]   = (yes_bid*size) - trade_took["cout"] if trade_took ["side"] == "BUY" else (no_bid*size) - trade_took["cout"]
                trade_took["moyenne_ratio"] = moyenne_yes if trade_took ["side"] == "BUY" else moyenne_no
                trade_took["tendance"] = tendance_yes if trade_took ["side"] == "BUY" else tendance_no
                trade_took["bid_exit"] = no_bid if trade_took["side"] == "SELL" else yes_bid
                trade_took["slope"] = slope_yes if trade_took["side"] == "BUY" else slope_no 
                
                maj_csv(trade_took)                                     # cloture de position
                trade_took["trade"] = False                  
                # print( "temps dépassé : cloture de position et retransciption ...") 
                print ("Condition cloture : ", trade_took["resultat"] , "$  ", restant, "   ", trade_took["side"], "   ", yes_bid if trade_took ["side"] == "BUY" else no_bid, "    ", trade_took["cout"]) 
                
            if trade_took["trade"] == False:
                send_telegram(
                    f"""
                🛑 POSITION CLOTURE order_flow_v3.3 : 

                    Side      : {trade_took['side']}
                    BTC Price : {btc_price}
                    Size      : {size}

                💰 Token    : {yes_bid if trade_took["side"] == "BUY" else no_bid:.2f} 
                💰 Coût     : {trade_took['cout']:.2f}
                🎯 Resultat : {trade_took['resultat']:.2f}
                   Restant  :  {restant}
                """
                ) 
            
            else :
                time.sleep(1)
                continue
                
        #-----------------------------------------------------------------------------------------------------------------------------------------
        
        if trade_took["trade"] == True:
            time.sleep(1)
            continue
                 
     
        if yes_ask == 0 or no_ask == 0 or yes_ask == no_ask or toggle_poly_ws != True:
            time.sleep(1)
            continue
                    
        rr = benef / dd if benef > 0 and dd != 0 else 0.0
                    
        print ("HAUT Probas XBOOST  TS : ", yes_ask, "$  ",  proba," imba : ", yes_imbalance, " ", yes_ask > params["MIN_ASK"], " ", proba > params["PROBA"], "    " ,yes_ask < params["MAX_ASK"], "  ", restant, "s") 
        
        if yes_ask >  params["MIN_ASK"] and proba > params["PROBA"] and yes_ask < params["MAX_ASK"] and restant < params["LIMIT_TIME_MAX"] and restant > params["LIMIT_TIME_MIN"] and yes_imbalance > params["IMBALANCE"]:
        
            trade_took ["trade"]     = True
            trade_took ["prix_exec"] = btc_price
            trade_took ["date"]      = now.strftime("%d/%m/%Y") 
            trade_took ["heure_entry"]     = now.strftime("%H:%M:%S")  
            trade_took ["yes_token"] = yes_token
            trade_took ["no_token"]  = no_token
            trade_took["token"]      = yes_token  
            trade_took ["side"]      = "BUY"  
            trade_took ["time"] = restant
            trade_took["variation"] = save_variation
            trade_took["imbalance"] = yes_imbalance
            trade_took["delta_imbalance"] = delta_yes_imbalance
            trade_took["proba_entry"] = proba
            trade_took["PM"] = benef
            trade_took["dd"] = dd
            save_signal = 0
            price_token = yes_ask
            spread = yes_spread
        else:
            None# print ("Prédiction négative ...")          
              
        print ("BAS Probas XBOOST  TS : ", no_ask, "$  ",  proba_no," imba : ", no_imbalance,  "    ", no_ask > params["MAX_ASK"] , " ",   proba_no > params["PROBA"], " ", no_ask < params["MAX_ASK"], "  ", restant, "s")
        
        rr = benef_no / dd_no if benef_no > 0 and dd_no != 0 else 0.0
            
        if no_ask > params["MIN_ASK"] and proba_no > params["PROBA"] and no_ask < params["MAX_ASK"] and restant < params["LIMIT_TIME_MAX"] and restant > params["LIMIT_TIME_MIN"] and no_imbalance > params["IMBALANCE"]:
            
            trade_took ["trade"]     = True
            trade_took ["prix_exec"] = btc_price
            trade_took ["date"]      = now.strftime("%d/%m/%Y") 
            trade_took ["heure_entry"]     = now.strftime("%H:%M:%S")  
            trade_took ["yes_token"] = yes_token
            trade_took ["no_token"]  = no_token
            trade_took ["token"]     = no_token  
            trade_took ["side"]      = "SELL"  
            trade_took ["time"] = restant
            trade_took["variation"] = save_variation  
            trade_took["imbalance"] = no_imbalance
            trade_took["delta_imbalance"] = delta_no_imbalance
            trade_took["proba_entry"] = proba_no
            trade_took["PM"] = benef_no
            trade_took["dd"] = dd_no
            save_signal = 0
            price_token = no_ask
            spread = no_spread
        else:
            None# print ("Prédiction négative ...")
                        
        
        if trade_took ["trade"] == True:
            # print ("ACHAT YES OU NO Prix du token :",  yes_ask, "   NO ", no_ask)
                                                 
            trade_took["stake"] = stake
            
            erreur   = ""    # ← définir avant le try
            response = None

            # Option 2 — create_market_order + post_order séparés
            # order = client.create_market_order(
                # MarketOrderArgs(
                    # token_id = trade_took["token"],
                    # amount   = BUDGET,
                    # side     = BUY,
                    # price    = MAX_ASK + 0.01,
                # ),
                # options = PartialCreateOrderOptions(
                    # tick_size = "0.01",
                    # neg_risk  = False,
                # ),
            # )
            # response = client.post_order(order, OrderType.FOK)
             
            # order_id = response["orderID"]

            # details = get_order_details_safe(order_id)
          
            # pprint.pp(details) 
            # print("DETAILS : ", details)
                         
            # Récupération du prix réel d'exécution
            
            one_by_period = one_by_period +  1
            
            price_token = yes_ask if trade_took["side"] == "BUY" else no_ask
            
            executed_price = yes_ask if trade_took["side"] == "BUY" else no_ask  # par défaut
            if response and "avgPrice" in response:
                executed_price = float(response.get("avgPrice", executed_price))
            elif response and "price" in response:
                executed_price = float(response.get("price", executed_price))
                
            if True: #details == None:
                trade_took ["cout"] = BUDGET + spread
            else:
                trade_took ["cout"] = round(float(details["size_matched"]) * executed_price + spread, 2)
            
            trade_took["profit"]    = (1 / executed_price * trade_took ["cout"]) - trade_took ["cout"]
            
            size =  round(trade_took["cout"] / executed_price, 2)
            
            trade_took["ask_entry"] = no_ask if trade_took["side"] == "SELL" else yes_ask
            
            trade_took["ratio_entry"] = ratio_buy if trade_took["side"] == "BUY" else ratio_sell
            
            trade_took["temps"] = restant
            
            trade_took["moyenne_ratio_entry"] = moyenne_yes if trade_took["side"] == "BUY" else moyenne_no 
            trade_took["tendance_entry"] = tendance_yes if trade_took["side"] == "BUY" else tendance_no
            trade_took["slope_entry"] = slope_yes if trade_took["side"] == "BUY" else slope_no
            
            trailing_stop = 0
            # size = float(details["size_matched"]) if details != None and float(details["size_matched"]) > 0 else round(trade_took["cout"] / executed_price, 2)- 0.1
            
            print( "Analyse de volumme' : ", tendance_yes if trade_took["side"] == "BUY" else tendance_no, "    ", ratio_buy if trade_took["side"] == "BUY" else ratio_sell, " slope : ", slope_yes if trade_took["side"] == "BUY" else slope_no)
            
            # print ("Prise de position : ", trade_took ["side"], "  size : ", size, "   variation  : ", trade_took["variation"], "  ASK : ", price_token, "     cout : ", trade_took["cout"], " TARGET : ", executed_price * (1+PROFIT_MIN), " STAKE : ", stake)  
            # print(f"✅ Ordre exécuté | Prix réel : {executed_price:.4f} | Quantité : {trade_took ["cout"]/executed_price} profit potentiel : ", trade_took["profit"], " ratio order flow : ", ratio_buy if trade_took["side"] == "BUY" else ratio_sell)                
            # print("✅ Ordre placé !")
            # print("Status    :", response.get("status")) 

            send_telegram(
                f"""
            📈 POSITION OUVERTE order_flow_v3.3 - (scalping)

            Side      : {trade_took['side']}
            BTC Price : {btc_price}
            Size      : {size}

            💰 Coût   : {trade_took['cout']:.2f}
            💰 Token   : {executed_price:.2f}     
            
            ⏳ Expire dans {restant}s
            """
            )
            
        time.sleep(1)

            
    except KeyboardInterrupt:
        print("🛑 Bot arrêté manuellement")
        break
    except Exception as e:
        print(f"❌ Erreur : {e}")
        traceback.print_exc()
        if "no orders found" in erreur or "FAK" in erreur or "FOK" in erreur:
            print("⚠️ Pas de contrepartie — skip cette opportunité")        
     
        trade_took ["trade"]     = False
        # sys.exit()
        time.sleep(1)   # attendre avant de réessayer
        continue    
    
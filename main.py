#-------------- Standard Library -------------
import os
import sys
import subprocess
from datetime import datetime, timezone
#---------- Third-Party Libraries ------------
import ulid
import pandas as pd
#------------- Project Modules ---------------
from database_creation import create_database
from analysis import analyse_game_results

from sim.gbm import (
    simulate_gbm,
    simulate_from_yfinance,
)

from sim.runner import play_game

from SQL_Functions import (
    insert_run_row,
    insert_gbm_config_row,
    insert_hist_config_row,
    insert_fundamental_series,
    insert_agent_population,
    insert_agent_round_rows,
    insert_market_round_rows,
)
#---------------------------------------------


DB_PATH = "testdataset.duckdb"

if not os.path.exists(DB_PATH):
    create_database(DB_PATH)

#INITIAL PARAMETER SETS DO NOT CHANGE
S0 = None
volatility = None
drift = None
seed = None
ticker = None
interval = None
start_date = None
price_col = None
auto_adjust = None
fundamental_path = None
completion_time = None
#INITIAL PARAMETER SETS DO NOT CHANGE

def compute_end_date(start_date, n_rounds, interval):
    start_date = pd.Timestamp(start_date)

    interval_map = {
        "1m": pd.Timedelta(minutes=1),
        "2m": pd.Timedelta(minutes=2),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "60m": pd.Timedelta(hours=1),
        "1h": pd.Timedelta(hours=1),
        "1d": pd.Timedelta(days=1),
        "1wk": pd.Timedelta(weeks=1),
        "1mo": pd.DateOffset(months=1)
    }

    step = interval_map[interval]

    end_date = start_date + n_rounds * step

    return end_date


# ----------------------------
# Example params + run
# ----------------------------

def generate_ULID() -> str: #Function for generating the ULID for each run
    return str(ulid.new())

def generate_py_Vers() -> str: #Function for generating the python version for each run
    return str(sys.version)

def generate_time() -> str: #Function for generating the creation time of the run
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") 

def generate_code_Vers() -> str: #Function for getting the current code version (based off github)
    try:
        return subprocess.check_output(["git", "describe", "--tags", "--dirty", "--always"],stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

#-----------------------------

#RUN DATA RUN DATA RUN DATA RUN DATA RUN DATA RUN DATA RUN DATA RUN DATA

n_zi_agents = 10                # Number of ZI agents
n_signal_following_agents = 10  # Number of Signal Following Agents
n_utility_maximiser_agents = 10 # Number of Utility Maximiser Agents
n_contrarian_agents =  10       # Number of Contrarian Agents
n_adapt_sig_agents = 10         # Number of Adapted Signal Agents

#Run Primary Key
run_id = generate_ULID()         #UAID format

#Run Info
print(f"Run {run_id} created")                       #Test to make sure it works
experiment_name = "Experiment Name"                  #Name of the experiment
experiment_type = "A sub section for the experiment" #Type of experiment
creation_time = generate_time()                      #Generates the time of creation
run_notes = "Notes for the run"                      #Run notes
n_rounds = 100                                       #Number of rounds 
fundamental_source = "GBM"                           #Source of fundamental ["GBM", "Historical"] 


#Run Status
run_status = "STARTED"          #run_status [STARTED, RUNNING, COMPLETED, FAILED]
run_progress = 0                #run_progress (0-100%, changes throughout)

#Version data
py_vers = generate_py_Vers()    #Gets the current Python Version 
code_vers = generate_code_Vers()#Generates code version from github repository 

#Agent Data
n_agents = n_zi_agents + n_signal_following_agents + n_utility_maximiser_agents + n_contrarian_agents + n_adapt_sig_agents #Gets the total number of agents, used for later tests and to store how many agents are deployed per-run
total_initial_cash = 1000 #Sets total initial cash to be split between agents
total_initial_shares = 10 #Sets total initial shares to be split between agents
cash_to_share_ratio = total_initial_cash/total_initial_shares #Cash to Share Ratio

#Market Mechanisms
market_mechanism = "call_auction"  #
pricing_rule = "maximum_volume_minimum_imbalance" #
rationing_rule = "proportional_rationing" # 
tie_break_rule = "previous_price_proximity" #
transaction_cost_rate = 0.000 #Transaction Cost, applied for every buy order and sell order

noise_parameter_distribution_type = "uniform" #Options - [uniform, bimodal, skewed, evenly_spaced]

if noise_parameter_distribution_type == "uniform":
    low = 0.0 
    high = 0.5
    distribution_data = {"low": low, 
                         "high": high}

elif noise_parameter_distribution_type == "bimodal":
    group_a_mean = 0.1
    group_a_std  = 0.05
    group_b_mean = 0.9
    group_b_std  = 0.05
    distribution_data = {
        "group_a_mean": group_a_mean,
        "group_a_std" : group_a_std,
        "group_b_mean": group_b_mean,
        "group_b_std" : group_b_std
    }

elif noise_parameter_distribution_type == "skewed":
    mean = -1
    sigma = 0.5
    distribution_data = {"mean" : mean,
                         "sigma" : sigma}
elif noise_parameter_distribution_type == "evenly_spaced":
    low  = 0.05
    high = 0.1
    distribution_data = {"low": low,
                         "high": high}
else:
    raise ValueError(f"Unknown noise_parameter_distribution_type: {noise_parameter_distribution_type}")

signal_generator_noise_distribution = 'lognormal' #[lognormal, uniform]
bias = 0.0 


insert_run_row(DB_PATH,
               run_id,
               experiment_name,
               experiment_type,
               creation_time,
               completion_time, 
               run_notes, n_rounds, 
               fundamental_source, 
               run_status, 
               run_progress, 
               py_vers, 
               code_vers, 
               n_agents,
               total_initial_cash,
               total_initial_shares,
               market_mechanism,
               pricing_rule,rationing_rule,
               tie_break_rule,transaction_cost_rate,
               noise_parameter_distribution_type,
               distribution_data,
               signal_generator_noise_distribution,
               bias)


#Generate fundamental path
if fundamental_source == "GBM":
    #GBM Config
    S0 = 100
    volatility = 0.01
    drift = 0.01
    seed = "SAGFJAKFGAGFKALFJGAFAGJDGSJGFGGF"
    insert_gbm_config_row(DB_PATH, run_id, S0, volatility, drift, seed)
    fundamental_path = simulate_gbm(S0, volatility, drift, n_rounds, seed)
    fundamental_series = tuple(enumerate(fundamental_path))
    insert_fundamental_series(DB_PATH, run_id, fundamental_series)

elif fundamental_source == "Historical":
    #Historical Config
    ticker = "AAPL"
    interval = "1d"
    start_date = "2024-01-01"
    end_date = compute_end_date(start_date, n_rounds, interval)
    price_col = "Close"
    auto_adjust = True
    insert_hist_config_row(DB_PATH, run_id, ticker, interval, start_date, end_date, price_col, auto_adjust)
    fundamental_path = simulate_from_yfinance(ticker,start_date,n_rounds,interval,price_col,auto_adjust)
    fundamental_series = tuple(enumerate(fundamental_path))
    insert_fundamental_series(DB_PATH, run_id, fundamental_series)

else:
    raise ValueError("Error")
# ----------------------------
# Runner
# ----------------------------

final_score, g = play_game(
    DB_PATH,
    n_agents,
    n_zi_agents,
    n_signal_following_agents,
    n_utility_maximiser_agents,
    n_contrarian_agents,
    n_adapt_sig_agents,
    n_rounds,
    total_initial_shares,
    total_initial_cash,
    cash_to_share_ratio,
    run_id,
    market_mechanism,
    pricing_rule,
    rationing_rule,
    tie_break_rule,
    transaction_cost_rate,
    noise_parameter_distribution_type,
    distribution_data,
    signal_generator_noise_distribution,
    bias,
    fundamental_source,
    S0,
    volatility,
    drift,
    ticker,
    interval,
    start_date,
    price_col,
    auto_adjust,
    fundamental_path,
    seed
)

insert_agent_population(DB_PATH, run_id, g.agents)
insert_market_round_rows(DB_PATH, g.market_round_records)
insert_agent_round_rows(DB_PATH, g.agent_round_records)


results_df = analyse_game_results(
    g,
    final_score,
    title_prefix=f"{experiment_name} | "
)


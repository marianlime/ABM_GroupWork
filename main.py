from Second_Draft_Trading_Strategies.py import *
from signal_gen import *
from order_book.py import *

""" To do:
Parameters:
- Number of signal following agents: n_strategic_agents
- Number of noisy following agents: n_zi_agents
- Number of rounds: n_rounds
- Initial underlying: S0
- Volatility: volatility
- Drift: drift
- Total initial shares : total_initial_shares
- Cash to asset ratio : cash_to_share_ratio
- Distribution of players : information_distribution in {uniform, bimodal, lognormal}
- Distribution of noise: noise_distribution in {lognormal, uniform, ...}
- Run id: ulid

1. initialise game class:
Do the game stock path
  game.stock_path
    - simulate gbm with seed as current
  game.rounds_number
  ... for all parameters
  game.run_id


2. initialise set of players:
- generate noise_parameter 
inputs: number of players, distribution of players in set {normal, uniform, bimodal}
outputs: numpy array of noise_parameter, length n_players 
- assign each strategic player:
  - agent_id
  - agent_type in 'mixed', 'signal_following'
  - initial_cash
  - initial_shares
  - noise_parameter
  
- assign each noise player:
  - agent_id
  - agent_type = 'zi'
  - initial_cash
  - initial_shares
  - noise_parameter
  
3. Running through rounds
- Round Number
- St
- S_t+1

Initialise order list = []
For each strategy player:
  Inputs:
  - signal
      call signal function
        - inputs: noise_parameter, S_t+1
        - output: signal 
  - S_t
  - noise_parameter
  Output:
  {'limit_price':100, 'quantity': 10, 'action':'buy'}
  Update order list for each player

  Pass list through LOB:
  Inputs: 
  - order_list
  Outputs:
  - price and quantity
  - list of all order

  Loop through all agents, update:
  - cash
  - quantity held

  round_number++;

When round number equal n_rounds:
Liquidate each players assets
List with player type, noise parameter and net gain
Check zero sum game
"""

# Parameters:


n_strategic_agents = 100
n_zi_agents = 100
n_rounds = 10
S0 = 100
volatility = 0.02
drift = 0
total_initial_shares = 100
cash_to_share_ratio = 1
distribution_of_players = 'normal'
distribution_of_noise = 'uniform'
run_id = 1


class game:
    def __init__(self, n_strategic_agents, n_zi_agents, n_rounds, S0, volatility, drift, total_initial_shares, cash_to_share_ratio, run_id):
        self.current_round = 0
        self.players = {}
        self.sigma = sigma
        self.n_strategic_agents = n_strategic_agents
        self.n_zi_agents = n_zi_agents
        self.n_players = n_strategic_agents + n_zi_agents
        self.S0 = S0
        self.volatility = volatility
        self.drift = drift
        self.total_initial_shares = total_initial_shares
        self.initial_cash = total_initial_shares * total_initial_shares * cash_to_share_ratio
        self.run_id = run_id
        

        # Order history: tuple {current_round : {buy_orders, sell_orders}}
        self.order_history = {}
        self.price_history = {}
        self.noise_parameter_set = assign_noise_parameter_set(n_agents, dist_type="bimodal")

        # Creates N lots of informed traders
        for i in range(n_strategic_agents):
            self.players[i] = Trader(
              uaid = i, 
              cash = self.total_initial_cash / (self.n_players), 
              shares = self.total_initial_shares / self.n_players, 
              noise_parameter = noise_parameter_set[i],
              strategy_probs = None, 
              trader_type: str = "signal_following")

        # Creates noise traders
        for i in range(n_zi_agents):
            self.players[i] = Trader(
              uaid = i, 
              cash = self.total_initial_cash / self.n_players, 
              shares = self.total_initial_shares / self.n_players, 
              noise_parameter = None,
              strategy_probs = None, 
                trader_type: str = "zi")
        
        def gather_orders(self, current_round):
          order_list = []
        #Iterating through dict of players
          for player_id, player in self.players.items():
              order_list.append(player.place_order(
                signal = signal_generator(player.noise_parameter, 
                S_next= self.stock_path[current_round + 1], noise_distribution='lognormal'), 
                value = self.stock_path[current_round]))

        best_price, total_volume, trades = clear_market(order_list)

        

              


      
      



Parameters:
- Number of signal following agents: n_strategic_agents
- Number of noisy following agents: n_zi_agents
- Number of rounds: n_rounds
- Initial underlying: S0
- Volatility: volatility
- Drift: drift
- Total initial shares : total_initial_shares
- Cash to asset ratio : cash_to_share_ratio
- Distribution of players : information_distribution in {uniform, bimodal, lognormal}
- Distribution of noise: noise_distribution in {lognormal, uniform, ...}
- Run id: ulid



  
  

"""

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


  
  

  
  

"""

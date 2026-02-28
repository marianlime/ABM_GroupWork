from Trading_Strategies import Trader, zero_intelligence, signal_following

# Step 1: Creates traders

# An informed trader using signal-following
t1 = Trader(uaid=1, cash=5000.0, shares=50.0, info_param=0.05, trader_type="signal_following")

# A zero-intelligence trader
t2 = Trader(uaid=2, cash=5000.0, shares=50.0, trader_type="zi")

# Mark 2 zero-intelligence trade
t3 = Trader(uaid=3, cash=5000.0, shares=50.0, trader_type="zi")

# Step 2:

# Current state of the world
# S_t from Matthew's GBM framework
# Placeholders for current cash and shares for each trader (could be different in a real test)
current_price = 100.0  

# Step 3: 

# Generated signals from Marian's signal generator.
# For now, these are just mocked them:
signal_t1 = 1.08  # t1 thinks price goes up 8%
signal_t2 = 0.97  # t2 gets a signal too, but ZI ignores it
signal_t3 = 1.02  # t3 ignores signal too

# Step 4: 

# Each trader generates an order based on their strategy and the current state.
# Note: the actual order dict structure and validation would depend on the implementation of place_order and
# the expected format for Ahmed's String mechanism.
order1 = t1.place_order(signal=signal_t1, value=current_price)
order2 = t2.place_order(signal=signal_t2, value=current_price)
order3 = t3.place_order(signal=signal_t3, value=current_price)

print("Signal-Following Trader:")
print(order1)
print()

print("Zero-Intelligence Trader:")
print(order2)
print()

print("ZI_Trader:")
print(order3)
print()

# Step 5: 

# Collect all orders into the list that Person 2's clearing mechanism expects
# The expected format for the clearing mechanism is a list of dicts.
# Keys: "ID", "Buy", "Sell", "Hold", "Quantity", "Price".
all_orders = [order1, order2, order3]

print("All orders for clearing mechanism:")
for order in all_orders:
    print(f"  Trader {order['ID']}: {'BUY' if order['Buy'] == 1.0 else 'SELL' if order['Sell'] == 1.0 else 'HOLD'} "f"{order['Quantity']} shares @ {order['Price']}")
"""
Generate sample flow data for testing the enhanced flow screener.
Creates realistic multi-leg options flow data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_flow_data(num_symbols=10, flows_per_symbol=20):
    """
    Generate sample multi-leg options flow data.
    """

    # Sample symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX', 'AMD', 'INTC'][:num_symbols]

    # Sample data for each symbol
    sample_data = {
        'AAPL': {'spot': 175, 'sector': 'Technology'},
        'MSFT': {'spot': 380, 'sector': 'Technology'},
        'GOOGL': {'spot': 140, 'sector': 'Technology'},
        'TSLA': {'spot': 250, 'sector': 'Automotive'},
        'NVDA': {'spot': 500, 'sector': 'Technology'},
        'AMZN': {'spot': 155, 'sector': 'Consumer'},
        'META': {'spot': 350, 'sector': 'Technology'},
        'NFLX': {'spot': 450, 'sector': 'Media'},
        'AMD': {'spot': 140, 'sector': 'Technology'},
        'INTC': {'spot': 45, 'sector': 'Technology'}
    }

    flows = []

    for symbol in symbols:
        spot_price = sample_data[symbol]['spot']

        # Generate multiple multi-leg trades for each symbol
        for trade_group in range(flows_per_symbol // 4):  # Each group has ~4 legs

            # Create timestamp for this trade group
            base_time = datetime.now() - timedelta(days=random.randint(0, 30))
            trade_time = base_time.replace(
                hour=random.randint(9, 15),
                minute=random.randint(0, 59),
                second=random.randint(0, 59)
            )

            # Generate expiration date (1-8 weeks out)
            exp_date = base_time + timedelta(days=random.randint(7, 56))
            # Make it a Friday
            exp_date = exp_date + timedelta(days=(4 - exp_date.weekday()) % 7)

            # Determine trade strategy type
            strategy_type = random.choice(['bull_call_spread', 'bear_put_spread', 'iron_condor', 'straddle'])

            if strategy_type == 'bull_call_spread':
                # Buy lower strike call, sell higher strike call
                lower_strike = spot_price + random.randint(-10, 5)
                higher_strike = lower_strike + random.randint(5, 15)

                # Buy call
                flows.append(create_flow_record(
                    symbol, 'BUY', 'CALL', lower_strike, spot_price, exp_date, trade_time,
                    premium=random.randint(200000, 500000),
                    volume=random.randint(300, 1000)
                ))

                # Sell call
                flows.append(create_flow_record(
                    symbol, 'SELL', 'CALL', higher_strike, spot_price, exp_date, trade_time,
                    premium=random.randint(100000, 300000),
                    volume=random.randint(300, 1000)
                ))

            elif strategy_type == 'bear_put_spread':
                # Buy higher strike put, sell lower strike put
                higher_strike = spot_price + random.randint(-5, 10)
                lower_strike = higher_strike - random.randint(5, 15)

                # Buy put
                flows.append(create_flow_record(
                    symbol, 'BUY', 'PUT', higher_strike, spot_price, exp_date, trade_time,
                    premium=random.randint(200000, 500000),
                    volume=random.randint(300, 1000)
                ))

                # Sell put
                flows.append(create_flow_record(
                    symbol, 'SELL', 'PUT', lower_strike, spot_price, exp_date, trade_time,
                    premium=random.randint(100000, 300000),
                    volume=random.randint(300, 1000)
                ))

            elif strategy_type == 'iron_condor':
                # Sell call spread and put spread
                atm_strike = spot_price

                # Put spread (sell higher, buy lower)
                put_sell_strike = atm_strike - random.randint(5, 15)
                put_buy_strike = put_sell_strike - random.randint(5, 10)

                # Call spread (sell lower, buy higher)
                call_sell_strike = atm_strike + random.randint(5, 15)
                call_buy_strike = call_sell_strike + random.randint(5, 10)

                # Four legs
                flows.extend([
                    create_flow_record(symbol, 'SELL', 'PUT', put_sell_strike, spot_price, exp_date, trade_time,
                                     premium=random.randint(150000, 300000), volume=random.randint(300, 800)),
                    create_flow_record(symbol, 'BUY', 'PUT', put_buy_strike, spot_price, exp_date, trade_time,
                                     premium=random.randint(100000, 200000), volume=random.randint(300, 800)),
                    create_flow_record(symbol, 'SELL', 'CALL', call_sell_strike, spot_price, exp_date, trade_time,
                                     premium=random.randint(150000, 300000), volume=random.randint(300, 800)),
                    create_flow_record(symbol, 'BUY', 'CALL', call_buy_strike, spot_price, exp_date, trade_time,
                                     premium=random.randint(100000, 200000), volume=random.randint(300, 800))
                ])

            elif strategy_type == 'straddle':
                # Buy call and put at same strike (but make it not exactly balanced to avoid filtering)
                strike = spot_price + random.randint(-5, 5)

                # Make premiums slightly unbalanced
                call_premium = random.randint(200000, 400000)
                put_premium = random.randint(150000, 350000)

                # Ensure it's not perfectly balanced (40-60% rule)
                total = call_premium + put_premium
                call_ratio = call_premium / total
                if 0.4 <= call_ratio <= 0.6:
                    # Adjust to make it unbalanced
                    call_premium = int(call_premium * 1.3)

                flows.extend([
                    create_flow_record(symbol, 'BUY', 'CALL', strike, spot_price, exp_date, trade_time,
                                     premium=call_premium, volume=random.randint(400, 1000)),
                    create_flow_record(symbol, 'BUY', 'PUT', strike, spot_price, exp_date, trade_time,
                                     premium=put_premium, volume=random.randint(400, 1000))
                ])

    # Convert to DataFrame
    df = pd.DataFrame(flows)

    # Sort by created datetime
    df = df.sort_values('CreatedDateTime').reset_index(drop=True)

    return df

def create_flow_record(symbol, buy_sell, call_put, strike, spot, exp_date, created_time, premium, volume):
    """
    Create a single flow record.
    """

    # Calculate some derived fields
    dte = (exp_date.date() - created_time.date()).days

    # Assign colors based on trade type
    if buy_sell == 'BUY':
        color = 'GREEN' if call_put == 'CALL' else 'BLUE'
    else:
        color = 'RED' if call_put == 'CALL' else 'ORANGE'

    # Random side assignment
    side = random.choice(['A', 'B', 'C'])

    # Open interest (should be higher than volume for realistic data)
    oi = volume + random.randint(200, 1000)

    # Price per contract
    price = premium / (volume * 100)  # Assuming 100 shares per contract

    # Implied volatility (realistic range)
    iv = random.uniform(0.15, 0.45)

    # Earnings flag (10% chance)
    er_flag = 'T' if random.random() < 0.1 else 'F'

    return {
        'Symbol': symbol,
        'Buy/Sell': buy_sell,
        'CallPut': call_put,
        'Strike': strike,
        'Spot': spot,
        'ExpirationDate': exp_date.strftime('%Y-%m-%d'),
        'Premium': premium,
        'Volume': volume,
        'OI': oi,
        'Price': round(price, 2),
        'Side': side,
        'Color': color,
        'CreatedDateTime': created_time.strftime('%Y-%m-%d %H:%M:%S'),
        'ImpliedVolatility': round(iv, 4),
        'Dte': dte,
        'ER': er_flag,
        'MktCap': random.choice(['Large', 'Mid', 'Small']),
        'Sector': 'Technology',  # Simplified
        'StockEtf': 'Stock',
        'Uoa': random.choice(['Unusual', 'Normal']),
        'Weekly': random.choice(['Y', 'N']),
        'Type': 'Options'
    }

if __name__ == "__main__":
    # Generate sample data
    sample_df = generate_sample_flow_data(num_symbols=8, flows_per_symbol=24)

    # Save to CSV
    filename = f"sample_flow_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    sample_df.to_csv(filename, index=False)

    print(f"Generated {len(sample_df)} flow records")
    print(f"Saved to: {filename}")
    print(f"Unique symbols: {sample_df['Symbol'].nunique()}")
    print(f"Date range: {sample_df['CreatedDateTime'].min()} to {sample_df['CreatedDateTime'].max()}")

    # Show preview
    print("\nPreview:")
    print(sample_df.head(10).to_string())

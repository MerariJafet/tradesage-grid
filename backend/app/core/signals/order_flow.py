def order_flow_imbalance(bids, asks):
    """Compute order flow imbalance using top 5 bid/ask levels."""
    bid_volume = sum(float(level[1]) for level in bids[:5])
    ask_volume = sum(float(level[1]) for level in asks[:5])
    return (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-9)

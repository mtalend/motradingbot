import asyncio
import websockets
import json

# Alpaca WebSocket URL (replace 'iex' or 'sip' depending on your subscription)
ALPACA_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"  # or sip for premium data feed

# Authentication keys (replace with your Alpaca API key and secret)
API_KEY = "PKFQXTS1SDN126XJDGKG"
API_SECRET = "oBYVMDftCc3zTwC8o2uVBZGg0IuJ5iccCV0rDBwQ"

# List of symbols you want to subscribe to (can be modified as per your need)
symbols = {
    "trades": ["AAPL"],  # List of symbols for trades
    "quotes": ["AMD", "CLDR"],  # List of symbols for quotes
    "bars": ["AAPL", "VOO"],  # List of symbols for minute bars
}

async def subscribe(ws):
    # Authentication payload
    auth_message = {
        "action": "auth",
        "key": API_KEY,
        "secret": API_SECRET
    }

    # Subscription message for trades, quotes, bars
    subscription_message = {
        "action": "subscribe",
        "trades": symbols["trades"],
        "quotes": symbols["quotes"],
        "bars": symbols["bars"]
    }

    # Send authentication and subscription messages
    await ws.send(json.dumps(auth_message))
    await ws.send(json.dumps(subscription_message))

async def process_message(message):
    # Parse the received message
    data = json.loads(message)

    # Check the message type and handle accordingly
    for event in data:
        if event["T"] == "t":  # Trade data
            print(f"Trade - Symbol: {event['S']}, Price: {event['p']}, Size: {event['s']}, Time: {event['t']}")
        elif event["T"] == "q":  # Quote data
            print(f"Quote - Symbol: {event['S']}, Bid: {event['bp']} @ {event['bs']}, Ask: {event['ap']} @ {event['as']}")
        elif event["T"] == "b":  # Bar data
            print(f"Bar - Symbol: {event['S']}, Open: {event['o']}, High: {event['h']}, Low: {event['l']}, Close: {event['c']}, Volume: {event['v']}")

async def fetch_order_book():
    async with websockets.connect(ALPACA_WS_URL) as ws:
        # Subscribe to trades, quotes, and bars
        await subscribe(ws)

        # Continuously receive and process messages
        while True:
            try:
                message = await ws.recv()
                await process_message(message)
            except websockets.ConnectionClosed:
                print("Connection closed. Reconnecting...")
                break

# Run the event loop
if __name__ == "__main__":
    asyncio.run(fetch_order_book())

# get_access_token.py

import os
from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect

# Load existing .env file
load_dotenv()
env_path = ".env"

# Get API key and secret
api_key = os.getenv("KITE_API_KEY")
api_secret = os.getenv("KITE_API_SECRET")

if not api_key or not api_secret:
    print("❌ Please set KITE_API_KEY and KITE_API_SECRET in your .env file.")
    exit()

# Initialize KiteConnect
kite = KiteConnect(api_key=api_key)

# Step 1: Show login URL
print("\n🔑 Step 1: Open the following login URL in your browser:")
print(kite.login_url())

# Step 2: Ask for request token after login
request_token = input(
    "\n🔗 Step 2: Paste the 'request_token' here from the redirect URL: "
).strip()

# Step 3: Exchange request token for access token
try:
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]

    # Save to .env
    set_key(env_path, "KITE_ACCESS_TOKEN", access_token)

    print("\n✅ ACCESS_TOKEN saved to .env file as KITE_ACCESS_TOKEN.")
    print("🔁 You can now run your WebSocket code.")
except Exception as e:
    print(f"\n❌ Error generating access token: {e}")

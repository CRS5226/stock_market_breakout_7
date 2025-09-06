import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

STOCK_NAME = "WIPRO"
CSV_FILE = "../latest_data_WIPRO.csv"
NUM_ROWS = 10

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

if df.shape[0] < NUM_ROWS:
    raise ValueError(
        f"Not enough rows in the file to analyze (only {df.shape[0]} rows)."
    )

latest_df = df[["Timestamp", "Open", "High", "Low", "Close"]].tail(NUM_ROWS)
latest_csv = latest_df.to_csv(index=False)

prompt = f"""You are a professional stock analyst. Analyze the following real-time stock data and determine the next trading action.

Please provide:
1. A clear recommendation (e.g., Buy, Sell, Hold, Set Stop Loss, Wait)
2. A target price (if applicable)
3. Key reasons based on Close price trend, Volume, Support/Resistance, Bollinger Bands, ADX, and breakout/breakdown patterns
4. Risk Level (Low, Medium, High)

Only respond with concise and precise analysis.

Here is the most recent data:
{latest_csv}"""

response = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a stock market expert."},
        {"role": "user", "content": prompt},
    ],
    model="allam-2-7b",
    temperature=0.4,
    max_tokens=1024,
    top_p=1.0,
    stream=False,
)

output_text = response.choices[0].message.content.strip()

output_file_path = os.path.join(
    os.path.dirname(CSV_FILE), f"stock_analysis_{STOCK_NAME}.txt"
)
with open(output_file_path, "w", encoding="utf-8") as file:
    file.write(output_text)

print("\n📊 Stock Forecast for:", STOCK_NAME)
print("🧠 AI Recommendation:")
print(output_text)
print(f"\nNumber of rows fetched from CSV: {latest_df.shape[0]}")
print(latest_df)
print(f"Analysis saved to: {output_file_path}")

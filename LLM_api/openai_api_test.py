import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    # This is the default and can be omitted
    api_key=API_KEY,
)

response = client.responses.create(
    model="gpt-4o",
    instructions="You are a helper that answers questions.",
    input="What is the capital of India?",
)

print(response.output_text)

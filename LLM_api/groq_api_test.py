import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

stream = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        },
    ],
    model="allam-2-7b",
    temperature=0.5,
    max_completion_tokens=1024,
    top_p=1,
    stop=None,
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="")

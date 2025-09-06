import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core.exceptions import GoogleAPICallError

# Load your .env file to get the API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in environment variables.")
    exit(1)

# Configure the API key
genai.configure(api_key=api_key)


def get_response(prompt_text):
    try:
        print(f"\n📤 Prompt: {prompt_text}")

        # Use a supported Gemini model
        model = genai.GenerativeModel("models/gemini-pro")

        response = model.generate_content(
            prompt_text,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )

        print("\n✅ Response:")
        print(response.text)

    except GoogleAPICallError as e:
        print(f"\n❌ API Error: {e.message}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    get_response("Explain the concept of a black hole in simple terms.")

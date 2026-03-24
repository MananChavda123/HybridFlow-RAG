import os
from anthropic import Anthropic
from anthropic._exceptions import APIStatusError

# Insert your API key here or use environment variable
API_KEY = os.getenv("ANTHROPIC_API_KEY")  # <-- replace if needed

client = Anthropic(api_key=API_KEY)

# Claude 3 models to test
models_to_test = [
    "claude-4-opus",
    "claude-4-sonnet",
    "claude-4-haiku",
    "claude-2.1",        # in case older versions are available
    "claude-1.2"
]

print("🔍 Testing access to Claude models using the API key...\n")

for model in models_to_test:
    try:
        response = client.messages.create(
            model=model,
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}]
        )
        print(f"✅ Access granted: {model}")
    except APIStatusError as e:
        if e.status_code == 404:
            print(f"❌ Not available: {model}")
        elif e.status_code == 401:
            print(f"❌ Invalid API key or unauthorized for: {model}")
            break
        else:
            print(f"⚠️ Error {e.status_code} for {model}: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error for {model}: {e}")

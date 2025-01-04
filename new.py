import google.generativeai as genai
import os

# Configure API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List available models
models = genai.list_models()

# Print available models
for model in models:
    print(f"Model Name: {model.name}, Version: {getattr(model, 'version', 'No version info')}")

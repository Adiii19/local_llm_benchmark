from transformers import AutoTokenizer
import os
from dotenv import load_dotenv

load_dotenv()  # load HF_TOKEN from .env

tokenizer = AutoTokenizer.from_pretrained(
    "mistral/Mistral-7B-Instruct-v0.2",
    use_auth_token=os.getenv("HUGGING_FACE_TOKEN")
)
print("Tokenizer loaded successfully!")
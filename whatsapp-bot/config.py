import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# WhatsApp API configuration
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_APP_ID = os.getenv("WHATSAPP_APP_ID")
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

CALLBACK_URL = os.getenv("CALLBACK_URL", "https://your-ngrok-url.ngrok.io/webhook")

# Instagram API configuration
INSTAGRAM_PAGE_ACCESS_TOKEN = os.getenv("INSTAGRAM_PAGE_ACCESS_TOKEN")
INSTAGRAM_APP_SECRET = os.getenv("INSTAGRAM_APP_SECRET")
INSTAGRAM_APP_ID = os.getenv("INSTAGRAM_APP_ID")

# Facebook API configuration
FACEBOOK_PAGE_ACCESS_TOKEN = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN")
FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET")
FACEBOOK_APP_ID = os.getenv("FACEBOOK_APP_ID")

# Telegram API configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

META_API_VERSION = os.getenv("META_API_VERSION", "v18.0")

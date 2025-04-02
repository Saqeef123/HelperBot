# WhatsApp Chatbot using PyWa and FastAPI

This project implements a WhatsApp chatbot using PyWa (Python WhatsApp Cloud API wrapper) and FastAPI.

## Features

- Receives and responds to WhatsApp messages
- Handles interactive button callbacks
- Processes different types of text messages
- Simple response system with customizable commands

## Prerequisites

- Python 3.8+
- WhatsApp Business API access (Meta for Developers)
- A public URL for your webhook (Ngrok, CloudFlare Tunnel, or a hosted server)

## Setup

1. Clone this repository:
```
git clone <repository-url>
cd whatsapp-bot
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Set up your WhatsApp Business API:
   - Go to [Meta for Developers](https://developers.facebook.com/)
   - Create a Meta App
   - Set up WhatsApp messaging in the app
   - Get your Phone Number ID, Access Token, App ID, and App Secret

4. Configure your environment variables:
   - Rename `.env.example` to `.env`
   - Fill in your WhatsApp API credentials
   - Set your webhook URL and verify token

5. Set up a webhook URL:
   - For testing, you can use [Ngrok](https://ngrok.com/): `ngrok http 8000`
   - Use the generated HTTPS URL as your CALLBACK_URL in .env

## Running the Bot

Start the FastAPI server:

```
python app.py
```

Or use Uvicorn directly:

```
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Extending the Bot

You can extend the bot by:

1. Adding more message handlers in `app.py`
2. Creating custom response functions 
3. Implementing advanced features like NLP integration

## API Documentation

Once running, access the FastAPI automatic documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT

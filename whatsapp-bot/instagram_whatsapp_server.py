from fastapi import FastAPI, Request, Depends, HTTPException
from pywa import WhatsApp, filters, types
import logging
import json
import os
import hmac
import hashlib
import requests
from typing import Dict, Any
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import asyncio
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters as tg_filters, CallbackContext
from config import (
    WHATSAPP_PHONE_ID,
    WHATSAPP_TOKEN,
    CALLBACK_URL,
    WHATSAPP_APP_ID,
    WHATSAPP_APP_SECRET,
    VERIFY_TOKEN,
    INSTAGRAM_PAGE_ACCESS_TOKEN,
    INSTAGRAM_APP_SECRET,
    INSTAGRAM_APP_ID,
    FACEBOOK_PAGE_ACCESS_TOKEN,
    FACEBOOK_APP_SECRET,
    FACEBOOK_APP_ID,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_API_URL,
    META_API_VERSION
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Social Media Bot")

# Initialize WhatsApp client
wa = WhatsApp(
    phone_id=WHATSAPP_PHONE_ID,
    token=WHATSAPP_TOKEN,
    server=app,
    callback_url=CALLBACK_URL,
    verify_token=VERIFY_TOKEN,
    app_id=WHATSAPP_APP_ID,
    app_secret=WHATSAPP_APP_SECRET,
    webhook_challenge_delay=10.0,
)

# Load the FAQ data
with open("c:/Users/SUPER-COMPUTERS/work/Safeeq/datawith100.json", 'r') as file:
    faq_data = json.load(file)

# Set Groq API key
os.environ["GROQ_API_KEY"] = "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"

# Initialize the Groq LLM
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0.2,
)

# Create a prompt template with the FAQ data
faq_text = "\n".join([f"Q: {question}\nA: {answer}" for question, answer in faq_data.items()])
template = f"""You are a helpful assistant for an internship program at IAC (Industry Academia Community).
Your responses should be concise and well-formatted for messaging platforms.

Here is information about the internship program:

{faq_text}

Current conversation:
{{history}}
Human: {{input}}
AI: """

prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# User conversation memories - shared across platforms
user_memories = {}

# Instagram API URL
INSTAGRAM_API_URL = f"https://graph.facebook.com/{META_API_VERSION}/me/messages"

# Facebook API URL
FACEBOOK_API_URL = f"https://graph.facebook.com/{META_API_VERSION}/me/messages"

# Initialize Telegram bot
telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

def verify_webhook_signature(request_data: bytes, signature_header: str, app_secret: str) -> bool:
    """Verify that the webhook request is coming from Meta"""
    if not signature_header:
        return False
    
    expected_signature = hmac.new(
        app_secret.encode('utf-8'),
        msg=request_data,
        digestmod=hashlib.sha1
    ).hexdigest()
    
    return hmac.compare_digest(f"sha1={expected_signature}", signature_header)

async def send_instagram_message(recipient_id: str, message_text: str):
    """Send message to Instagram user"""
    # Instagram messaging API endpoint
    url = f"https://graph.facebook.com/{META_API_VERSION}/me/messages"
    
    # Request payload
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    params = {
        "access_token": INSTAGRAM_PAGE_ACCESS_TOKEN
    }
    
    try:
        logger.info(f"Sending Instagram message to {recipient_id}: {message_text}")
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            params=params
        )
        if response.status_code == 200:
            logger.info(f"Successfully sent Instagram message, response: {response.json()}")
            return response.json()
        else:
            logger.error(f"Failed to send Instagram message. Status code: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error sending Instagram message: {e}")
        logger.exception("Detailed exception:")
        return None

async def send_facebook_message(recipient_id: str, message_text: str):
    """Send message to Facebook user"""
    # Facebook messaging API endpoint
    url = FACEBOOK_API_URL
    
    # Request payload
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text}
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    params = {
        "access_token": FACEBOOK_PAGE_ACCESS_TOKEN
    }
    
    try:
        logger.info(f"Sending Facebook message to {recipient_id}: {message_text}")
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            params=params
        )
        if response.status_code == 200:
            logger.info(f"Successfully sent Facebook message, response: {response.json()}")
            return response.json()
        else:
            logger.error(f"Failed to send Facebook message. Status code: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error sending Facebook message: {e}")
        logger.exception("Detailed exception:")
        return None

async def setup_instagram_webhook():
    """Setup Instagram webhook subscription using Graph API"""
    url = f"https://graph.facebook.com/{META_API_VERSION}/{INSTAGRAM_APP_ID}/subscriptions"
    
    params = {
        "access_token": INSTAGRAM_PAGE_ACCESS_TOKEN,
        "object": "instagram",
        "callback_url": f"{CALLBACK_URL}/instagram-webhook",
        "verify_token": VERIFY_TOKEN,
        "fields": "messages,messaging_postbacks,message_deliveries"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        logger.info(f"Setting up Instagram webhook subscription: {params}")
        response = requests.post(url, params=params, headers=headers)
        response_data = response.json()
        logger.info(f"Instagram webhook subscription response: {response_data}")
        return response_data
    except Exception as e:
        logger.error(f"Error setting up Instagram webhook: {e}")
        return None

async def setup_facebook_webhook():
    """Setup Facebook webhook subscription using Graph API"""
    url = f"https://graph.facebook.com/{META_API_VERSION}/{FACEBOOK_APP_ID}/subscriptions"
    
    params = {
        "access_token": FACEBOOK_PAGE_ACCESS_TOKEN,
        "object": "page",
        "callback_url": f"{CALLBACK_URL}/facebook-webhook",
        "verify_token": VERIFY_TOKEN,
        "fields": "messages,messaging_postbacks,message_deliveries"
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        logger.info(f"Setting up Facebook webhook subscription: {params}")
        response = requests.post(url, params=params, headers=headers)
        response_data = response.json()
        logger.info(f"Facebook webhook subscription response: {response_data}")
        return response_data
    except Exception as e:
        logger.error(f"Error setting up Facebook webhook: {e}")
        return None

@wa.on_message(filters.text)
def handle_whatsapp_message(client: WhatsApp, msg: types.Message):
    """Handle incoming WhatsApp text messages"""
    logger.info(f"Received WhatsApp message from {msg.from_user.name}: {msg.text}")
    
    user_message = msg.text.lower()
    user_id = f"whatsapp_{msg.from_user.wa_id}"  # Prefix with platform for multi-platform support
    
    try:
        # Get user-specific memory or create a new one
        if user_id not in user_memories:
            user_memories[user_id] = ConversationBufferMemory(return_messages=True)
        
        # Create a user-specific conversation chain
        user_chain = ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=user_memories[user_id],
            verbose=False
        )
        
        # Get response from the AI
        response = user_chain.predict(input=user_message)
        logger.info(f"AI response: {response}")
        
        # Send the response to the user
        msg.reply_text(text=response)
    
    except Exception as e:
        logger.error(f"Error processing WhatsApp message: {e}")
        logger.exception("Detailed exception:")
        msg.reply_text(text="Sorry, I'm having trouble answering your question right now. Please try again later.")

async def handle_telegram_message(update: Update, context: CallbackContext):
    """Handle Telegram messages"""
    message_text = update.message.text
    user_id = f"telegram_{update.effective_user.id}"
    
    logger.info(f"Received Telegram message from {update.effective_user.first_name}: {message_text}")
    
    try:
        # Get user-specific memory or create a new one
        if user_id not in user_memories:
            user_memories[user_id] = ConversationBufferMemory(return_messages=True)
        
        # Create a user-specific conversation chain
        user_chain = ConversationChain(
            llm=llm,
            prompt=prompt,
            memory=user_memories[user_id],
            verbose=False
        )
        
        # Get response from the AI
        response = user_chain.predict(input=message_text)
        logger.info(f"AI response for Telegram: {response}")
        
        # Send the response
        await update.message.reply_text(response)
    
    except Exception as e:
        logger.error(f"Error processing Telegram message: {e}")
        logger.exception("Detailed exception:")
        await update.message.reply_text("Sorry, I'm having trouble answering your question right now. Please try again later.")

async def setup_telegram_handlers():
    """Set up Telegram message handlers"""
    # Command handler for /start
    async def start_command(update: Update, context: CallbackContext):
        await update.message.reply_text("Hello! I'm the IAC internship assistant bot. How can I help you?")
    
    # Add handlers
    telegram_app.add_handler(CommandHandler("start", start_command))
    telegram_app.add_handler(MessageHandler(tg_filters.TEXT & ~tg_filters.COMMAND, handle_telegram_message))
    
    # Start polling (in a non-blocking way)
    await telegram_app.initialize()
    await telegram_app.start()
    logger.info("Telegram bot started")

async def setup_telegram_webhook():
    """Setup Telegram webhook instead of polling"""
    webhook_url = f"{CALLBACK_URL}/telegram-webhook"
    api_url = f"{TELEGRAM_API_URL}/setWebhook"
    
    params = {
        "url": webhook_url,
        "allowed_updates": ["message", "callback_query"]
    }
    
    try:
        logger.info(f"Setting up Telegram webhook: {webhook_url}")
        response = requests.post(api_url, json=params)
        response_data = response.json()
        logger.info(f"Telegram webhook setup response: {response_data}")
        return response_data
    except Exception as e:
        logger.error(f"Error setting up Telegram webhook: {e}")
        return None

@app.post("/instagram-webhook")
async def instagram_webhook(request: Request):
    """Handle Instagram webhook events"""
    body = await request.body()
    body_text = body.decode('utf-8')
    signature = request.headers.get("X-Hub-Signature", "")
    
    logger.info(f"Received POST request to /instagram-webhook. Headers: {request.headers}")
    logger.info(f"Raw body: {body_text}")
    
    # Verify this is a legitimate request from Meta
    if signature and not verify_webhook_signature(body, signature, INSTAGRAM_APP_SECRET):
        logger.warning(f"Invalid signature for Instagram webhook: {signature}")
        raise HTTPException(status_code=403, detail="Invalid signature")
    
    try:
        data = await request.json()
        logger.info(f"Received Instagram webhook data: {data}")
        
        # Try different Instagram message formats
        if 'entry' in data:
            for entry in data['entry']:
                # Check standard messaging format
                if 'messaging' in entry:
                    for messaging_event in entry['messaging']:
                        logger.info(f"Processing messaging event: {messaging_event}")
                        if 'message' in messaging_event and 'text' in messaging_event['message']:
                            sender_id = messaging_event['sender']['id']
                            message_text = messaging_event['message']['text']
                            logger.info(f"Found text message from sender {sender_id}: {message_text}")
                            
                            # Process the message with our AI
                            user_id = f"instagram_{sender_id}"
                            
                            # Get or create user memory
                            if user_id not in user_memories:
                                user_memories[user_id] = ConversationBufferMemory(return_messages=True)
                            
                            # Create conversation chain
                            user_chain = ConversationChain(
                                llm=llm,
                                prompt=prompt,
                                memory=user_memories[user_id],
                                verbose=False
                            )
                            
                            # Get AI response
                            response = user_chain.predict(input=message_text)
                            logger.info(f"AI response for Instagram: {response}")
                            
                            # Send response back to Instagram
                            await send_instagram_message(sender_id, response)
                
                # Check for the changes array format (Instagram Graph API)
                if 'changes' in entry:
                    for change in entry['changes']:
                        logger.info(f"Processing change: {change}")
                        if change.get('field') == 'messages':
                            value = change.get('value', {})
                            if 'messages' in value:
                                for message in value['messages']:
                                    logger.info(f"Processing Instagram graph message: {message}")
                                    if 'from' in value and 'text' in message:
                                        sender_id = value['from']['id']
                                        message_text = message['text']['body']
                                        logger.info(f"Found Graph API message from sender {sender_id}: {message_text}")
                                        
                                        # Process with AI and respond
                                        user_id = f"instagram_graph_{sender_id}"
                                        
                                        # Get or create user memory
                                        if user_id not in user_memories:
                                            user_memories[user_id] = ConversationBufferMemory(return_messages=True)
                                        
                                        # Create conversation chain
                                        user_chain = ConversationChain(
                                            llm=llm,
                                            prompt=prompt,
                                            memory=user_memories[user_id],
                                            verbose=False
                                        )
                                        
                                        # Get AI response
                                        response = user_chain.predict(input=message_text)
                                        logger.info(f"AI response for Instagram Graph: {response}")
                                        
                                        # Send response back to Instagram
                                        await send_instagram_message(sender_id, response)
    except Exception as e:
        logger.error(f"Error processing Instagram webhook: {e}")
        logger.exception("Detailed exception:")
    
    # Return a 200 OK to acknowledge receipt of the webhook
    return {"status": "ok"}

@app.get("/instagram-webhook")
async def verify_instagram_webhook(request: Request):
    """Verify the webhook subscription for Instagram"""
    logger.info(f"Received GET request to /instagram-webhook: {request.query_params}")
    
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    
    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            logger.info(f"Instagram webhook verified with challenge: {challenge}")
            return int(challenge)
        else:
            logger.warning(f"Verification failed. Mode: {mode}, Token: {token}")
            raise HTTPException(status_code=403, detail="Verification failed")
    
    raise HTTPException(status_code=400, detail="Bad request")

@app.post("/facebook-webhook")
async def facebook_webhook(request: Request):
    """Handle Facebook webhook events"""
    body = await request.body()
    body_text = body.decode('utf-8')
    signature = request.headers.get("X-Hub-Signature", "")
    
    logger.info(f"Received POST request to /facebook-webhook. Headers: {request.headers}")
    logger.info(f"Raw body: {body_text}")
    
    # Verify this is a legitimate request from Meta
    if signature and not verify_webhook_signature(body, signature, FACEBOOK_APP_SECRET):
        logger.warning(f"Invalid signature for Facebook webhook: {signature}")
        raise HTTPException(status_code=403, detail="Invalid signature")
    
    try:
        data = await request.json()
        logger.info(f"Received Facebook webhook data: {data}")
        
        # Process Facebook Messenger events
        if 'entry' in data:
            for entry in data['entry']:
                # Standard Facebook Messenger messaging format
                if 'messaging' in entry:
                    for messaging_event in entry['messaging']:
                        logger.info(f"Processing Facebook messaging event: {messaging_event}")
                        if 'message' in messaging_event and 'text' in messaging_event['message']:
                            sender_id = messaging_event['sender']['id']
                            message_text = messaging_event['message']['text']
                            logger.info(f"Found Facebook message from sender {sender_id}: {message_text}")
                            
                            # Process the message with our AI
                            user_id = f"facebook_{sender_id}"
                            
                            # Get or create user memory
                            if user_id not in user_memories:
                                user_memories[user_id] = ConversationBufferMemory(return_messages=True)
                            
                            # Create conversation chain
                            user_chain = ConversationChain(
                                llm=llm,
                                prompt=prompt,
                                memory=user_memories[user_id],
                                verbose=False
                            )
                            
                            # Get AI response
                            response = user_chain.predict(input=message_text)
                            logger.info(f"AI response for Facebook: {response}")
                            
                            # Send response back to Facebook
                            await send_facebook_message(sender_id, response)
    except Exception as e:
        logger.error(f"Error processing Facebook webhook: {e}")
        logger.exception("Detailed exception:")
    
    # Return a 200 OK to acknowledge receipt of the webhook
    return {"status": "ok"}

@app.get("/facebook-webhook")
async def verify_facebook_webhook(request: Request):
    """Verify the webhook subscription for Facebook"""
    logger.info(f"Received GET request to /facebook-webhook: {request.query_params}")
    
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")
    
    if mode and token:
        if mode == "subscribe" and token == VERIFY_TOKEN:
            logger.info(f"Facebook webhook verified with challenge: {challenge}")
            return int(challenge)
        else:
            logger.warning(f"Verification failed. Mode: {mode}, Token: {token}")
            raise HTTPException(status_code=403, detail="Verification failed")
    
    raise HTTPException(status_code=400, detail="Bad request")

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    """Handle Telegram webhook events"""
    try:
        update_data = await request.json()
        logger.info(f"Received Telegram webhook data: {update_data}")
        
        # Process the update
        if "message" in update_data and "text" in update_data["message"]:
            chat_id = update_data["message"]["chat"]["id"]
            user_id = update_data["message"]["from"]["id"]
            message_text = update_data["message"]["text"]
            
            # Get the first name if available
            first_name = update_data["message"]["from"].get("first_name", "User")
            
            logger.info(f"Processing Telegram message from {first_name}: {message_text}")
            
            # Process with AI
            user_id = f"telegram_{user_id}"
            
            # Get or create user memory
            if user_id not in user_memories:
                user_memories[user_id] = ConversationBufferMemory(return_messages=True)
            
            # Create conversation chain
            user_chain = ConversationChain(
                llm=llm,
                prompt=prompt,
                memory=user_memories[user_id],
                verbose=False
            )
            
            # Get AI response
            response = user_chain.predict(input=message_text)
            logger.info(f"AI response for Telegram: {response}")
            
            # Send response
            await send_telegram_message(chat_id, response)
    
    except Exception as e:
        logger.error(f"Error processing Telegram webhook: {e}")
        logger.exception("Detailed exception:")
    
    # Return 200 OK
    return {"ok": True}

async def send_telegram_message(chat_id: int, text: str):
    """Send a message to a Telegram chat"""
    url = f"{TELEGRAM_API_URL}/sendMessage"
    
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }
    
    try:
        logger.info(f"Sending Telegram message to {chat_id}: {text}")
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            logger.info(f"Successfully sent Telegram message, response: {response.json()}")
            return response.json()
        else:
            logger.error(f"Failed to send Telegram message. Status code: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error sending Telegram message: {e}")
        logger.exception("Detailed exception:")
        return None

@app.on_event("startup")
async def startup_event():
    """Run tasks on startup"""
    # Setup Instagram webhook
    await setup_instagram_webhook()
    # Setup Facebook webhook
    await setup_facebook_webhook()
    # Setup Telegram webhook or start polling (choose one)
    # For webhook mode:
    await setup_telegram_webhook()
    # For polling mode (comment out if using webhook):
    # asyncio.create_task(setup_telegram_handlers())

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Social Media Bot API is running!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("instagram_whatsapp_server:app", host="0.0.0.0", port=8000, reload=True)
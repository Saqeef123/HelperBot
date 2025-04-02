import modal 
from modal import asgi_app
from whatsapp_server import whatsapp_server

image = modal.Image.debian_slim(python_version="3.12.3").pip_install([
            "pywa",
            "fastapi",
            "uvicorn",
            "python-dotenv",
            "httpx",
            "firebase_admin",
            "langchain",
        ])

app = modal.App(name="whatsapp_server", image=image)

@app.function(image=image)
@asgi_app()
def fastapi_app():
    """Deploy WhatsApp webhook server on Modal."""
    # Import environment variables from .env when running locally
    if not modal.is_remote():
        from dotenv import load_dotenv
        load_dotenv()
    
    # Return the FastAPI app instance
    return whatsapp_server


if __name__ == "__main__":
    modal.run(fastapi_app)
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from model_prep.inference import *

class BotMessage(BaseModel):
    message: str


app = FastAPI()


@app.get("/")
async def open_main_page():
    return FileResponse("web/index.html")


# Endpoint to receive bot messages
@app.post("/bot/message")
async def receive_bot_message(bot_message: BotMessage):
    answer = get_chatbot_response(bot_message.message)
    return {"bot_response": answer}

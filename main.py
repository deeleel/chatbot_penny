from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from model_prep.inference import *

class BotMessage(BaseModel):
    message: str


app = FastAPI()
all_context = ''
count = 0

@app.get("/")
async def open_main_page():
    global all_context
    global count
    all_context = ''
    count = 0
    return FileResponse("web/index.html")


# Endpoint to receive bot messages
@app.post("/bot/message")
async def receive_bot_message(bot_message: BotMessage):
    global all_context
    global count
    answer, new_context = get_chatbot_response(bot_message.message, all_context, count)
    all_context = new_context
    count += 1

    return {"bot_response": answer}

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

import sys
sys.path.insert(0, '/home/diana/chatbot/chatbot_penny/bert')
from inference import * # type:ignore

sys.path.insert(0, '/home/diana/chatbot/chatbot_penny/gpt')
from gpt_inference import * # type:ignore


user_messages = {}
API_TOKEN = '1412991836:AAEk5JUNkvd8J-FM4OHjSPNK-3ZHbMadYRc'
answer_generator = 'bert'

# Define a command handler. This usually takes two parameters: update and context.
# update: represents an incoming update.
# context: contains data related to the chat and the update.
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Hello! This is ChatBot based on TBBT character Penny!')


def store_message(user_id, text) -> None:
    if user_id not in user_messages:
        user_messages[user_id] = []

    user_messages[user_id].append(text)

    if len(user_messages[user_id]) > 2:
        user_messages[user_id] = user_messages[user_id][-2:]


def generate_response(update: Update, context: CallbackContext):
    user_id = update.message.from_user.id
    text = update.message.text
    store_message(user_id=user_id, text=text)

    if answer_generator == 'bert':
        answer = get_chatbot_response(user_messages[user_id]) # type:ignore
    else:
        answer = get_chatbot_response_gpt(user_messages[user_id]) # type:ignore

    store_message(user_id=user_id, text=answer)
    return update.message.reply_text(answer)


def help_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('You can ask me anything!')
    
def switch_to_gpt(update: Update, context: CallbackContext) -> None:
    global answer_generator
    answer_generator = 'gpt'

def switch_to_bert(update: Update, context: CallbackContext) -> None:
    global answer_generator
    answer_generator = 'bert'


def main() -> None:
    # Create the Updater and pass it your bot's token.
    updater = Updater(API_TOKEN, use_context=True)
    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands, do something
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("gpt", switch_to_gpt))
    dispatcher.add_handler(CommandHandler("bert", switch_to_bert))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, generate_response))

    # Start the Bot
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
# chatbot_penny

## Start WEB
```
$ uvicorn main:app
```

## Start Telegram
```
$ python3 telegram_bot/main.py
```

## Structure

*data* - все необходимые данные для инференса\
*data_clean* - предобработка данных \
*gpt* - работа с моделью генерации текста \
*model_prep* - подготовка модели и код инференса \
*notebooks* - ноутбуки с использованием других подходов (TF-IDF) \
*web* - html файл для отображения чата с ботом \
*main.py* - работа с FastAPI

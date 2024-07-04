import pandas as pd

df = pd.read_csv('data/prepared_with_context+label+negative.csv')
df.dropna(how='any', inplace=True)

from bert.utils_bert import clean_symbols

questions = []
answers = []

with open('data/dialogs.txt', 'r') as f:
    lines = f.readlines()
    
    for i in lines:
        question, answer = i.replace('\n', '').split('\t')
        questions.append(question)
        answers.append(answer)

new_dialogs = pd.DataFrame({'original_response': answers, 'response': answers, 'context': questions, 'label': 1})

new_dialogs['original_response'] = new_dialogs['original_response'].apply(lambda x: clean_symbols(x))
new_dialogs['response'] = new_dialogs['response'].apply(lambda x: clean_symbols(x))
new_dialogs['context'] = new_dialogs['context'].apply(lambda x: clean_symbols(x))

new_dialogs['original_response'] = new_dialogs['original_response'].apply(lambda x: x[0].upper() + x[1:])
new_df = pd.concat([df, new_dialogs])

new_df.to_csv('data/updated_dialogs.csv', index=False)

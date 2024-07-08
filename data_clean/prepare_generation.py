
import pandas as pd

df = pd.read_csv('data/updated_dialogs.csv', index_col=0)
df.dropna(how='any', inplace=True)


df = df[df['label'] == 1]
df.reset_index(inplace=True)

df['response'] = df['response'].map(lambda x: '[A] ' + x + ' ' + '<|endoftext|>')
df['context'] = df['context'].map(lambda x: '[C] ' + x)


# list_res = []
# list_next_word = []

# for row in range(len(df)):
#     res = df['context'].iloc[row]
#     sentence = df['response'].iloc[row].split()
    
#     for k, i in enumerate(sentence):
#             if k == 0:
#                     list_res.append(res)
#             else:
#                     list_res.append(list_res[-1] + ' ' + list_next_word[-1])
#             list_next_word.append(sentence[k])


df.to_csv('data/data_for_generation.csv', index=False)

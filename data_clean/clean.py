import pandas as pd
import numpy as np
import sys
# adding model_utils to the system path
sys.path.insert(0, '/home/diana/chatbot/chatbot_penny/model_utils')
from help_funcs import * # type: ignore

df = pd.read_csv('data/1_10_seasons_tbbt.csv', index_col=False)


# clean 'person_scene'
df['new'] = df['person_scene'].apply(lambda x: x if x[0] == x[0].upper() else '')
df = df[~df['person_scene'].str.contains('\(|\)')]
df = df[df['person_scene'].str.get(0) == df['new'].str.get(0)]

df['person_len'] = df.person_scene.str.len()
df = df[df['person_len'] >= 3]
df = df[df['person_scene'] != 'Scene']

# drop useless data
df = df.drop(['new', 'person_len'], axis=1)
df.dropna(how='any', inplace=True)
df.reset_index(drop=True, inplace=True)


# prepare dialogues
df['original_dialogue'] = df['dialogue']
df['dialogue'] = df['dialogue'].apply(lambda x: clean_symbols(x))
df['dialogue'] = df['dialogue'].apply(lambda x: x.lower())


df['context_1'] = 0
df['context_2'] = 0
df['context_3'] = 0
df['context_4'] = 0

# prepare context with window = 5
for cont in range(1, 5):
    for i in range(cont, len(df)):
        cont_name = f'context_{cont}'
        df.at[i, cont_name] = df.iloc[i-cont]['dialogue'] + "$$$" + df.iloc[i-cont]['episode_name']


## Prepare true examples

def make_df_with_prev(df):
    lines_indexes = df[df['person_scene'] == MAIN_CHARACTER].index
    actual_idx = []

    context_1 = [] # prev
    context_2 = [] # prev + prevprev
    context_3 = []
    context_4 = []

    context_1_now, context_2_now, context_3_now, context_4_now = '', '', '', ''
    
    for i in lines_indexes:
        # если фраза 1я в эпизоде
        if df.iloc[i]['episode_name'] != df.iloc[i]['context_1'].split('$$$')[1]:
            actual_idx.append(i)
            context_1.append('$$$')
        else:
            actual_idx.append(i)
            context_1_now = df.iloc[i]['context_1'].split('$$$')[0]
            context_1.append(context_1_now)


        if df.iloc[i]['episode_name'] != df.iloc[i]['context_2'].split('$$$')[1]:
            context_2.append('$$$')
        else:
            context_2_now = df.iloc[i]['context_2'].split('$$$')[0] + " [SEP]" + context_1_now
            context_2.append(context_2_now)


        if df.iloc[i]['episode_name'] != df.iloc[i]['context_3'].split('$$$')[1]:
            context_3.append('$$$')
        else:
            context_3_now = df.iloc[i]['context_3'].split('$$$')[0] + " [SEP]" +  context_2_now
            context_3.append(context_3_now)


        if df.iloc[i]['episode_name'] != df.iloc[i]['context_4'].split('$$$')[1]:
            context_4.append('$$$')
        else:
            context_4_now = df.iloc[i]['context_4'].split('$$$')[0] + " [SEP]" + context_3_now
            context_4.append(context_4_now)

    print(len(df.iloc[actual_idx]['dialogue'].values))
    print(len(context_1))
    print(len(context_2))
    print(len(context_3))
    print(len(context_4))
    
    new_df = pd.DataFrame(data={
        'response': df.iloc[actual_idx]['dialogue'].values,
        'original_response': df.iloc[actual_idx]['original_dialogue'].values,
        'context_1': context_1,
        'context_2': context_2,
        'context_3': context_3,
        'context_4': context_4
    })
    return new_df

new_df = make_df_with_prev(df)

new_df['context_1'] = new_df['context_1'].apply(lambda x: clean_symbols(x))
new_df['context_2'] = new_df['context_2'].apply(lambda x: clean_symbols(x))
new_df['context_3'] = new_df['context_3'].apply(lambda x: clean_symbols(x))
new_df['context_4'] = new_df['context_4'].apply(lambda x: clean_symbols(x))


# concat all contexts to 2 columns dataframe
tmp1 = new_df[['original_response', 'response', 'context_1']].copy()
tmp1.rename(columns={'context_1': 'context'}, inplace=True)

tmp2 = new_df[['original_response', 'response', 'context_2']].copy()
tmp2.rename(columns={'context_2': 'context'}, inplace=True)

tmp3 = new_df[['original_response', 'response', 'context_3']].copy()
tmp3.rename(columns={'context_3': 'context'}, inplace=True)

tmp4 = new_df[['original_response', 'response', 'context_4']].copy()
tmp4.rename(columns={'context_4': 'context'}, inplace=True)

final_df = pd.concat([tmp1, tmp2, tmp3, tmp4], axis=0).reset_index(drop=True)
# add true label
final_df['label'] = 1


## Add negative examples

sheldon_dialogues = df[df['person_scene'] == 'Sheldon']['dialogue'].tolist()
tmp = pd.DataFrame(data={'original_response': final_df['original_response'], 'response': final_df['response']})
tmp['context'] = np.random.choice(sheldon_dialogues, size=len(final_df), replace=True)
tmp['label'] = 0
complete_df = pd.concat([final_df, tmp])


complete_df.dropna(how='any', inplace=True)
complete_df['response'] = complete_df['response'].apply(lambda x: ' '.join(x.split()))
complete_df['context'] = complete_df['context'].apply(lambda x: ' '.join(x.split()))
complete_df['context'] = complete_df['context'].apply(lambda x: x.replace('[sep]', '[SEP]'))

complete_df.to_csv('data/prepared_dataset.csv', index=False)
import pandas as pd

df = pd.read_csv('data/prepared_with_context+label+negative.csv')
df.dropna(how='any', inplace=True)

with open('data/show_data2.txt', 'w') as f:
    for i in range(len(df)):
        f.writelines('[C] ' + df.iloc[i]['context'] + '\n')
        f.writelines('[A] ' + df.iloc[i]['response'] + ' ' + '<|endoftext|>' + '\n\n')

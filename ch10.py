import numpy as np
import pandas as pd
import statsmodels.api as sm
from io import StringIO


df = pd.DataFrame({'key1': ['a','a','b','b','a'],
                    'key2': ['one', 'two', 'one', 'two', 'one'],
                    'data1': np.random.randn(5),
                    'data2': np.random.randn(5)})
print(df)
grouped = df['data1'].groupby(df['key1'])
print(grouped)
print(grouped.mean())
means = df['data1'].groupby([df['key1'], df['key2']]).mean()
print(means)
print(means.unstack())
states = np.array(['Ohio', 'California','California', 'Ohio','Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
print(df['data1'].groupby([states, years]).mean())
print(df.groupby('key1').mean())
print(df.groupby(['key1', 'key2']).mean())
print(df.groupby(['key1', 'key2']).size())

for name, group in df.groupby('key1'):
    print(name)
    print(group)

for (k1, k2), group in df.groupby(['key1','key2']):
    print((k1,k2))
    print(group)

pieces = dict(list(df.groupby('key1')))
pieces['b']
print(df.dtypes)
grouped = df.groupby(df.dtypes, axis = 1)
for dtype, group in grouped:
    print(dtype)
    print(group)

print(df.groupby('key1')['data1'])
print(df['data1'].groupby(df['key1']))
print(df.groupby('key1')[['data2']])
print(df[['data2']].groupby(df['key1']))

print(df.groupby(['key1', 'key2'])[['data2']].mean())
s_grouped = df.groupby(['key1','key2'])['data2']
print(s_grouped)
print(s_grouped)
print(s_grouped.mean())

people = pd.DataFrame(np.random.randn(5,5),
                        columns =['a','b','c','d','e'],
                        index = ['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
people.iloc[2:3, [1,2]] = np.nan # Add a few NA values
print(people)
mapping = {'a': 'red', 'b': 'red', 'c': 'blue',
           'd':'blue', 'e': 'red', 'f': 'orange'}

by_column = people.groupby(mapping,axis =1)
print(by_column.sum())
map_series = pd.Series(mapping)
print(map_series)
print(people.groupby(map_series, axis = 1).count())
print(people.groupby(len).sum())
key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len, key_list]).min()
columns = pd.MultiIndex.from_arrays([['US', 'US', 'US','JP', 'JP'],
                                        [1,3,5,1,3]], 
                                        names = ['cty','tenor'])
hier_df = pd.DataFrame(np.random.randn(4,5), columns = columns)
print(hier_df)
print(hier_df.groupby(level = 'cty', axis = 1).count())

print(df)
grouped = df.groupby('key1')
print(grouped['data1'].quantile(0.9))

def peak_to_peak(arr):
    return arr.max() - arr.min()
print(grouped.agg(peak_to_peak))
print(grouped.describe())

url = 'https://raw.githubusercontent.com/wesm/pydata-book/3rd-edition/examples/'
urltips = url + 'tips.csv'
tips = pd.read_csv(urltips)
tips['tip_pct'] = tips['tip']/tips['total_bill']
print(tips[:6])
grouped = tips.groupby(['day','smoker'])
grouped_pct = grouped['tip_pct']
print(grouped_pct.agg('mean'))
print(grouped_pct.agg(['mean', 'std', peak_to_peak]))
print(grouped_pct.agg([('foo','mean'), ('bar',np.std)]))
functions = ['count', 'mean', 'max']
result = grouped['tip_pct', 'total_bill'].agg(functions)
print(result)
print(result['tip_pct'])
ftuples = [('Durchschnitt', 'mean'), ('Abweichung', np.var)]
print(grouped['tip_pct', 'total_bill'].agg(ftuples))
print(grouped.agg({'tip': np.max, 'size':'sum'}))
print(grouped.agg({'tip_pct': ['min','max', 'mean','std'],
                    'size': 'sum'}))
print(tips.groupby(['day','smoker'], as_index = False).mean())
def top(df, n = 5, column = 'tip_pct'):
    return df.sort_values(by = column)[-n:]
print(top(tips, n = 6))
print(tips.groupby('smoker').apply(top))
print(tips.groupby(['smoker','day']).apply(top, n = 1, column = 'total_bill'))
result = tips.groupby('smoker')['tip_pct'].describe()
print(result)
print(result.unstack('smoker'))
print(tips.groupby('smoker', group_keys = False).apply(top))

frame = pd.DataFrame({'data1': np.random.randn(1000),
                        'data2': np.random.randn(1000)})
quartiles = pd.cut(frame.data1,4)
print(quartiles[:10])

def get_stats(group):
    return{'min':group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}
grouped = frame.data2.groupby(quartiles)
print(grouped.apply(get_stats).unstack())
grouping = pd.qcut(frame.data1, 10,labels = False)
grouped = frame.data2.groupby(grouping)
print(grouped.apply(get_stats).unstack())

s = pd.Series(np.random.randn(6))
s[::2] = np.nan
print(s)
print(s.fillna(s.mean()))
states = ['Ohio', 'New York', 'Vermont', 'Florida',
            'Oregon', 'Nevada', 'California', 'Idaho']

group_key = ['East']*4 +['West']*4
data = pd.Series(np.random.randn(8), index = states)
print(data)
data[['Vermont', 'Nevada', 'Idaho']] = np.nan
print(data)
print(data.groupby(group_key).mean())
fill_mean = lambda g: g.fillna(g.mean())
print(data.groupby(group_key).apply(fill_mean))
fill_values = {'East': 0.5, 'West': -1}
fill_func = lambda g: g.fillna(fill_values[g.name])
data.groupby(group_key).apply(fill_func)
suits = ['H', 'S', 'C', 'D']
card_val = (list(range(1,11))+[10]*3)*4
base_names = ['A'] + list(range(2,11)) + ['J', 'K', 'Q']
cards = []
#Hearts, Spades, Clubs, Diamonds
for suit in ['H', 'S', 'C', 'D']:
    cards.extend(str(num) + suit for num in base_names)
deck = pd.Series(card_val, index = cards)
print(deck[:13])
def draw(deck, n = 5):
    return deck.sample(n)
print(draw(deck))
get_suit = lambda card: card[-1] # last letter is suit
print(deck.groupby(get_suit).apply(draw, n = 2))
print(deck.groupby(get_suit, group_keys = False).apply(draw, n = 2))

df = pd.DataFrame({'category': ['a', 'a', 'a', 'a',
                                'b', 'b', 'b', 'b'],
                'data': np.random.randn(8),
                'weights': np.random.randn(8)})
print(df)

grouped = df.groupby('category')
get_wavg = lambda g: np.average(g['data'], weights = g['weights'])
print(grouped.apply(get_wavg))
urlstock = url + 'stock_px.csv'
close_px = pd.read_csv(urlstock, parse_dates = True,
                                index_col = 0)
print(close_px.info())
print(close_px[-4:])
spx_corr = lambda x:x.corrwith(x['SPX'])
rets = close_px.pct_change().dropna()
get_year = lambda x: x.year
by_year = rets.groupby(get_year)
print(by_year.apply(spx_corr))
print(by_year.apply(lambda g: g['AAPL'].corr(g['MSFT'])))

def regress(data,yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['Intercept'] = 1.
    result = sm.OLS(Y,X).fit()
    return result.params
print(by_year.apply(regress,'AAPL', ['SPX']))

print(tips.pivot_table(index = ['day', 'smoker']))
print(tips.pivot_table(['tip_pct', 'size'], index = ['time', 'day'],
                            columns ='smoker'))
print(tips.pivot_table(['tip_pct', 'size'], index = ['time', 'day'],
                        columns = 'smoker', margins = True))
print(tips.pivot_table('tip_pct', index = ['time', 'smoker'], columns = 'day',
                        aggfunc = len, margins = True))
print(tips.pivot_table('tip_pct', index = ['time', 'size', 'smoker'],
                        columns = 'day', aggfunc = 'mean', fill_value = 0))
#! blockstart
data = """Sample  Nationality  Handedness
1   USA  Right-handed
2   Japan    Left-handed
3   USA  Right-handed
4   Japan    Right-handed
5   Japan    Left-handed
6   Japan    Right-handed
7   USA  Right-handed
8   USA  Left-handed
9   Japan    Right-handed
10  USA  Right-handed"""
#! blockend
data = pd.read_table(StringIO(data), sep="\s+")
print(data)
print(pd.crosstab(data.Nationality, data.Handedness, margins = True))
print(pd.crosstab([tips.time,tips.day], tips.smoker, margins = True))
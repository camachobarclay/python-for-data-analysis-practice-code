import numpy as np
import pandas as pd
import os
from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse
from pandas.tseries.offsets import Hour, Minute
from pandas.tseries.offsets import Day, MonthEnd
import pytz
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

values = pd.Series(['apple', 'orange', 'apple', 'apple']*2)
print(values)
print(pd.unique(values))
print(pd.value_counts(values))
values = pd.Series([0,1,0,0]*2)
dim = pd.Series(['apple', 'orange'])
print(values)
print(dim)
print(dim.take(values))

fruits = ['apple', 'orange', 'apple', 'apple']*2
N = len(fruits)
df = pd.DataFrame({'fruit':fruits,
                    'basket_id': np.arange(N),
                    'count': np.random.randint(3,15, size = N),
                    'weight': np.random.uniform(0,4,size = N)},
                        columns = ['basket_id', 'fruit', 'count', 'weight'])
print(df)
fruit_cat = df['fruit'].astype('category')
print(fruit_cat)
c = fruit_cat.values
print(type(c))
print(c.categories)
print(c.codes)
df['fruit'] = df['fruit'].astype('category')
print(df.fruit)
my_categories = pd.Categorical(['foo', 'bar', 'baz', 'foo', 'bar'])
print(my_categories)
categories = ['foo', 'bar', 'baz']
codes = [0, 1, 2, 0, 0, 1]
my_cats_2 = pd.Categorical.from_codes(codes, categories)
print(my_cats_2)
ordered_cat = pd.Categorical.from_codes(codes, categories,
                                            ordered = True)
print(ordered_cat)
print(my_cats_2.as_ordered())

np.random.seed(12345)
draws= np.random.randn(1000)
print(draws[:5])
bins = pd.qcut(draws, 4)
print(bins)
bins = pd.qcut(draws, 4, labels = ['Q1', 'Q2', 'Q3', 'Q4'])
print(bins)
print(bins.codes[:10])
bins = pd.Series(bins, name = 'quartile')
results = (pd.Series(draws)
            .groupby(bins)
            .agg(['count','min', 'max'])
            .reset_index())
print(results)
print(results['quartile'])
#N = 10000000
#draws = pd.Series(np.random.randn(N))
#labels = pd.Series(['foo', 'bar', 'baz', 'qux']*(N//4))
#categories = labels.astype('category')
#print(labels.memory_usage())
#print(categories.memory_usage)

#%time _ = labels.astype('category')

s = pd.Series(['a','b','c','d']*2)
cat_s = s.astype('category')
print(cat_s)
print(cat_s.cat.codes)
print(cat_s.cat.categories)
actual_categories = ['a', 'b', 'c', 'd', 'e']
cat_s2 = cat_s.cat.set_categories(actual_categories)
print(cat_s2)
print(cat_s.value_counts())
print(cat_s2.value_counts())
cat_s3 = cat_s[cat_s.isin(['a','b'])]
print(cat_s3)
print(cat_s3.cat.remove_unused_categories())

cat_s = pd.Series(['a','b','c','d']*2,dtype = 'category')
print(pd.get_dummies(cat_s))

df = pd.DataFrame({'key': ['a', 'b', 'c']*4,
                    'value': np.arange(12.)})
print(df)
g = df.groupby('key').value
for key, item in g:
    print(g.get_group(key), "\n")
print(g.mean())
print(g.transform(lambda x: x.mean()))
print(g.transform('mean'))
print(g.transform(lambda x: x*2))
print(g.transform(lambda x: x*1))
print(g.transform(lambda x: x.rank(ascending = False)))

def normalize(x):
    return (x - x.mean())/x.std()

print(g.transform(normalize))
print(g.apply(normalize))
print(g.transform(lambda x:(x-x.mean())/x.std()))
print(g.transform('mean'))
normalized = (df['value'] - g.transform('mean'))/g.transform('std')
print(normalized)

N = 15
times = pd.date_range('2017-05-20 00:00', freq = '1min', periods = N)
df = pd.DataFrame({'time': times,
                    'value': np.arange(N)})
print(df)
print(df.set_index('time').resample('5min').count())
df2 = pd.DataFrame({'time': times.repeat(3),
                    'key': np.tile(['a','b','c'], N),
                    'value': np.arange(N*3.)})
print(df2)
print(df2[:7])
time_key = pd.Grouper(freq = '5min')
resampled = (df2.set_index('time')
            .groupby(['key', time_key])
            .sum())
print(resampled)
print(resampled.reset_index())

# df = load_data()
# df2 = df[df['col2']<0]
# df2['col1_demand'] = df2['col1']-df2['col1'].mean()
# result =df2.groupby('key').col1_demeaned.std()

#Usual non-functional way
# df2 = df.copy()
# df2['k'] = v

#Functional assign way
# df2 = df.assign(k = v)

# result = (df2.assign(col1_demeaned = df2.col1 - df2.col2.mean())
#            .groupby('key')
#            .col1_demeaned.std()

# df = load_data()
#   [lambda x: x['col2']<0]

# result = (load_data()
#            [lambda x: x.col2 < 0]
#            .assign(col1_demeaned = lambda x: x.col1 - x.col1.mean())
#            .groupby('key')
#            .col1_demeaned.std())

# a = f(df,arg1 = v1)
# b = g(a, v2, arg3 = v3)
# c = h(b, arg4 = v4)

# result = (df.pipe(f, arg1 = v1)
#           .pipe(g, v2, arg3 = v3)
#           .pipe(h, arg4 = v4))

# g = df.groupby(['key1', 'key2'])
# df['col1'] = df['col1'] - g.transform('mean')

# def group_demean(df, by, cols):
#   result = df.copy()
#   g = df.groupby(by)
#   for c in cols:
#       result[c] = df[c] - g[c].transform('mean')
#   return result

# result = (df[df.col1 <0]
#            .pipe(group_demean), ['key1', 'key2'], ['col1']))
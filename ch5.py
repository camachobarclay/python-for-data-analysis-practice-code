import pandas as pd
import numpy as np

from pandas import Series, DataFrame

obj = pd.Series([4,7,-5,3])

print(obj)

print(obj.index); print(obj.values)

obj2 = pd.Series([4,7,-5,3], index = ['d', 'b', 'a', 'c'])

print(obj2); print(obj2.index)

print(obj2['a']); 

obj2['d'] = 6; 

print(obj2['d'])

print(obj2[['c','a','d']])

print(obj2[obj2 > 0])

print(obj2*2)

print(np.exp(obj2))

print('b' in obj2)

print('e' in obj2)

sdata = {'Ohio': 2500, 'Texas':71000, 'Oregon': 16000, 'Utah': 5000}

obj3 = pd.Series(sdata)

print(obj3)

states = ['California', 'Ohio', 'Oregon', 'Texas']

obj4 = pd.Series(sdata, index = states)

print(obj4)

print(pd.isnull(obj4))

print(pd.notnull(obj4))

print(obj4.isnull())

print(obj3); print(obj4)

print(obj3 + obj4)

obj4.name = 'population'
obj4.index.name = 'state'

print(obj4)

print(obj)

obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']

print(obj)

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada','Nevada'],
'year': [2000,2001,2002,2001,2002,2003],
'pop': [1.5,1.7,3.6, 2.4, 2.9, 3.2]}

frame = pd.DataFrame(data)

print(frame); print(frame.head())

print(pd.DataFrame(data, columns = ['year', 'state', 'pop']))

frame2 = pd.DataFrame(data, columns = ['year', 'state', 'pop', 'debt'],
index = ['one', 'two', 'three', 'four', 'five', 'six'])

print(frame2)
print(frame2.columns)
print(frame2['state'])

print(frame2.year)

print(frame2.loc['three'])

frame2['debt'] = 16.5

print(frame2)

frame2['debt'] = np.arange(6.)

print(frame2)

val = pd.Series([-1.2, -1.5, -1.7], index = ['two', 'four', 'five'])

frame2['debt'] = val
print(frame2)

frame2['eastern'] = frame2.state =='Ohio'

del frame2['eastern']

print(frame2.columns)

pop = {'Nevada': {2001:2.4, 2002:2.9},
        'Ohio': {2000: 1.5, 2001:1.7, 2002:3.6}}

frame3 = pd.DataFrame(pop, index = [2000,2001,2002])

print(frame3)

print(frame3.T)

print(pd.DataFrame(pop, index = [2001,2002,2003]))

pdata = {'Ohio': frame3['Ohio'][:-1], 
        'Nevada': frame3['Nevada'][:2]}

myDF = pd.DataFrame(pdata)

print(myDF)

frame3.index.name = 'year'; frame3.columns.name = 'state'

print(frame3)

print(frame3.values)

print(frame2.values)

obj = pd.Series(range(3), index = ['a', 'b', 'c'])

index = obj.index

print(index); print(index[1:])

# index[1] = 'd' = #typeError

labels = pd.Index(np.arange(3))

print(labels)

obj2 = pd.Series([1.5, -2.5, 0], index = labels)

print(obj2)

print(obj2.index is labels)

print(frame3); print(frame3.columns); print('Ohio' in frame3.columns); print(2003 in frame3.index)

dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])

dup_labels = pd.Index(['foo', 'foo', 'bar', 'bar'])

print(dup_labels)

obj = pd.Series([4.5, 7.2, -5.3, 3.6], index = ['d', 'b', 'a', 'c'])

print(obj)

obj2 = obj.reindex(['a','b','c','d','e'])

print(obj2)

obj3 = pd.Series(['blue', 'purple', 'yellow'], index = [0,2,4])

print(obj3); print(obj3.reindex(range(6), method = 'ffill'))

frame = pd.DataFrame(np.arange(9).reshape((3,3)),
                index = ['a', 'c', 'd'],
                columns = ['Ohio', 'Texas', 'California'])

frame2 = frame.reindex(['a','b','c','d'])

print(frame); print(frame2)

states = ['Texas','Utah','California']

frame.reindex(columns = states)

#frame2.loc[['a', 'b', 'c', 'd'], states]

obj = pd.Series(np.arange(5.), index = ['a','b','c','d','e'])

print(obj)

new_obj = obj.drop('c'); print(new_obj)

print(obj); obj.drop(['d','c']); print(obj)

data = pd.DataFrame(np.arange(16).reshape((4,4)),
                index = ['Ohio', 'Colorado', 'Utah', 'New York'],
                columns = ['one', 'two', 'three', 'four'])

print(data)

olddata = data
data = data.drop(['Colorado','Ohio'])
print(data)
print(olddata)
data = olddata
print(data)


print(data.drop('two', axis = 1))

print(data.drop(['two','four'], axis = 'columns'))

print(obj)
print(obj.drop('c', inplace = True))

print(obj)

obj = pd.Series(np.arange(4,), index = ['a', 'b', 'c', 'd'])

print(obj)

print(obj['b']); print(obj[1]); print(obj[2:4]); print(obj[['b','a','d']]) 
print(obj[[1,3]]); print(obj[obj<2]); print(obj['b':'c']); 

obj['b':'c'] = 5

print(obj)

data = pd.DataFrame(np.arange(16).reshape((4,4)),
                        index = ['Ohio', 'Colorado', 'Utah', 'New York'], 
                        columns = ['one', 'two', 'three','four'])

print(data); print(data['two']); print(data[['three','one']])

print(data[2:]); print(data[data['three']>5])
print(data < 5)

data[data<5] = 0

print(data)

print(data.loc['Colorado', ['two', 'three']])
print(data.iloc[2,[3,0,1]])

print(data.iloc[2])

print(data.iloc[[1,2],[3,0,1]])

print(data.loc[:'Utah','two'])

print(data.iloc[:,:3][data.three > 5])

ser = pd.Series(np.arange(3.))

#print(ser[-1]) #produces an error

print(ser)

ser2 = pd.Series(np.arange(3.), index = ['a','b','c'])

print(ser2[[-1,-2,-3]])

print(ser[:1]); print(ser.loc[:1]); print(ser.iloc[:1])

s1 = pd.Series([7.3,-2.5,3.4,1.5], index = ['a','c','d','e'])
s2 = pd.Series([-2.1,3.6,-1.5,4, 3.1], index = ['a','c','e','f', 'g'])

print(s1); print(s2)

print(s1+s2)

df1 = pd.DataFrame(np.arange(9.).reshape((3,3)), columns = list('bcd'),
                        index = ['Ohio','Texas', 'Colorado'])

df2 = pd.DataFrame(np.arange(12.).reshape((4,3)), columns = list('bde'),
                        index = ['Utah','Ohio','Texas', 'Oregon'])

print(df1); print(df2)

print(df1 + df2)

df1 = pd.DataFrame({'A': [1,2]})
df2 = pd.DataFrame({'B':[3,4]})
print(df1); print(df2)
print(df1 - df2)

df1 = pd.DataFrame(np.arange(12.).reshape((3,4)),
columns = list('abcd'))

df2 = pd.DataFrame(np.arange(20.).reshape((4,5)), columns = list('abcde'))
df2.loc[1,'b'] = np.NaN

print(df1); print(df2)

print(df1 + df2)

print(df1.add(df2,fill_value = 0))

print(1/df1)

print(df1.rdiv(1)); print(df1.rdiv(1))

print(df1.reindex(columns = df2.columns, fill_value = 0))

arr = np.arange(12.).reshape((3,4))

print(arr); print(arr[0])

print(arr - arr[0])

frame = pd.DataFrame(np.arange(12.).reshape((4,3)),
                        columns = list('bde'),
                        index = ['Utah', 'Ohio', 'Texas', 'Oregon'])

series = frame.iloc[0]

print(frame); print(series)

print(frame - series)

series2 = pd.Series(range(3), index = ['b','e','f'])

print(frame + series2)

series3 = frame['d']

print(frame)

print(series3)

print(frame.sub(series3, axis = 'index'))

frame = pd.DataFrame(np.random.randn(4,3),columns=list('bde'),
                                index=['Utah','Ohio','Texas','Oregon'])

print(frame)

print(np.abs(frame))

f = lambda x: x.max() - x.min()

print(frame)
frame.apply(f)
print(frame.apply(f))

print(frame.apply(f,axis = 'columns'))

def f(x):
        return pd.Series([x.min(), x.max()], index = ['min', 'max'])

print(frame.apply(f))

format = lambda x: '%.2f' % x
print(frame)
print(frame.applymap(format))
print(frame)

print(frame['e'].map(format))

obj = pd.Series(range(4), index = ['d', 'a', 'b', 'c'])

print(obj.sort_index())

frame = pd.DataFrame(np.arange(8).reshape((2,4)),
                        index = ['three', 'one'],
                        columns = ['d', 'a', 'b', 'c'])

print(frame.sort_index())

print(frame.sort_index(axis = 1))

print(frame.sort_index(axis = 1, ascending= False))

obj = pd.Series([4,7,-3,2])

print(obj.sort_values())

obj = pd.Series([4,np.nan,7, np.nan, -3, 2])

print(obj.sort_values())

frame = pd.DataFrame({'a':[0,1,0,1], 'b':[4,7,-3,2]})

print(frame)

print(frame.sort_values(by = 'b'))

print(frame.sort_values(by = ['a', 'b']))

obj = pd.Series([7,-5,7,4,2,0,4])

print(obj.rank())

print(obj.rank(method = 'first'))

print(obj.rank(ascending = False, method = 'max'))

frame = pd.DataFrame({'b':[4.3,7,-3,2], 'a':[0,1,0,1],
                        'c':[-2,5,8,-2.5]})

print(frame)

print(frame.rank(axis = 'columns'))

obj = pd.Series(range(5), index = ['a', 'a', 'b', 'b', 'c'])

print(obj)

print(obj.index.is_unique)

print(obj['a']); print(obj['c'])

df = pd.DataFrame(np.random.randn(4,3), index = ['a', 'a', 'b', 'b'])

print(df)

print(df.loc['b'])

df = pd.DataFrame([[1.4,np.nan],[7.1,-4.5],
                [np.nan, np.nan], [0.75, -1.3]],
                index = ['a', 'b', 'c', 'd'], 
                columns = ['one', 'two'])

print(df)
print(df.sum())
print(df.sum(axis = 'columns'))
print(df.mean(axis = 'columns', skipna = False))

print(df.idxmax()); print(df.idxmin())

print(df.cumsum())

obj = pd.Series(['a','b','b','c']*4)

print(obj.describe())

#import pandas_datareader.data as web
#all_data = {ticker: web.get_data_yahoo(ticker)
#for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']}

#price = pd.DataFrame({ticker: data['Adj Close'] for ticker, data in all_data.items()})
#volume = pd.DataFrame({ticker: data['Volume'] for ticker, data in all_data.items()})

#returns = price.pct_change()

#print(returns.tail())

#print(returns['MSFT'].corr(returns['IBM']))

#print(returns['MSFT'].cov(returns['IBM']))

#print(returns.MSFT.corr(returns.IBM))

#print(returns.corr())

#print(returns.cov())

#print(returns.corrwith(returns.IBM))

#print(returns.corrwith(volume))

obj = pd.Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])

uniques = obj.unique()

print(uniques)

print(uniques.sort())

print(obj.value_counts())

print(pd.value_counts(obj.values, sort = False))

print(obj)

mask = obj.isin(['b','c'])

print(mask)

print(obj[mask])

to_match = pd.Series(['c','a','b','b','c','a'])

unique_vals = pd.Series(['c','b','a'])

print(pd.Index(unique_vals).get_indexer(to_match))

data = pd.DataFrame({'Qu1':[1,3,4,3,4],
                        'Qu2':[2,3,1,2,3],
                        'Qu3':[1,5,2,4,4]})

print(data)

result = data.apply(pd.value_counts).fillna(0)

print(result)
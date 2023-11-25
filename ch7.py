import pandas as pd
import numpy as np
from numpy import nan as NA
import string
import re


string_data = pd.Series(['aardvark', 'artichoke', np.nan, 'avocado'])
print(string_data)
print(string_data.isnull())
string_data[0] = None
print(string_data.isnull())

data = pd.Series([1,NA,3.5,NA,7])
print(data.dropna())
print(data)
print(data[data.notnull()])

data = pd.DataFrame([[1., 6.5, 3.], [1., NA, NA],
                [NA,NA,NA], [NA,6.5,3.]])
cleaned = data.dropna()
print(data)
print(cleaned)
print(data.dropna(how = 'all'))
data[4] = NA
print(data.dropna(axis = 1,how = 'all'))

df = pd.DataFrame(np.random.rand(7,3))
df.iloc[:4,1] = NA
df.iloc[:2,2] = NA
print(df)
print(df.dropna())
print(df.dropna(thresh = 2))
print(df.fillna(0))
print(df.fillna({1:0.5,2:0}))
_ = df.fillna(0, inplace = True)
print(df)

df = pd.DataFrame(np.random.randn(6,3))
df.iloc[2:,1] = NA
df.iloc[4:,2] = NA
print(df)
print(df.fillna(method = 'ffill'))
df.fillna(method = 'ffill', limit = 2)
data = pd.Series([1., NA, 3.5, NA, 7])
data.fillna(data.mean())

data = pd.DataFrame({'k1':['one','two']*3 + ['two'],
                    'k2': [1,1,2,3,3,4,4]})
print(data)
print(data.duplicated())
print(data.drop_duplicates())
data['v1'] = range(7)
print(data)
print(data.drop_duplicates(['k1']))
print(data)
print(data.drop_duplicates(['k1', 'k2'], keep = 'last'))

data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon',
                        'Pastrami', 'corned beef', 'Bacon',
                        'pastrami', 'honey ham', 'nova lox'],
                    'ounces': [4,3,12,6,7.5,8,3,5,6]})
print(data)
meat_to_animal = {'bacon': ' pig',
                    'pulled pork': 'pig',
                    'pastrami': 'cow',
                    'corned beef': 'cow',
                    'honey ham': 'pig',
                    'nova lox': 'salmon'
                    }
lowercased = data['food'].str.lower()
print(lowercased)
data['animal'] = lowercased.map(meat_to_animal)
data = pd.Series([1., -999., 2., -999., -1000., 3.])
print(data)
print(data.replace(-999, np.nan))
print(data.replace([-999,-1000], np.nan))
print(data.replace([-999, -1000], [np.nan, 0]))
print(data.replace({-999: np.nan, -1000:0}))

data = pd.DataFrame(np.arange(12).reshape((3,4)),
                        index = ['Ohio', 'Colorado', 'New York'],
                        columns = ['one', 'two', 'three', 'four'])
transform = lambda x: x[:4].upper()
print(data.index.map(transform))
data.index = data.index.map(transform)
print(data)
print(data.rename(index = str.title, columns=str.upper))
print(data.rename(index= {'Ohio': 'Indiana'}, columns = {'three': 'peekaboo'}))
data.rename(index = {'OHIO': 'INDIANA'}, inplace = True)
print(data)

ages = [20,22,25,27,21,23,37,31,61,45,41,32]
bins = [18,25,35,60,100]
cats = pd.cut(ages,bins)
print(cats)
print(cats.codes)
print(cats.categories)
print(pd.value_counts(cats))
print(pd.cut(ages, [18,26, 36,61,100], right = False))
group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
data = np.random.rand(20)
print(data)
print(pd.cut(data,4, precision = 2))

data = np.random.randn(1000) #normally distributed
cats = pd.qcut(data,4)
print(cats)
print(pd.value_counts(cats))
print(pd.qcut(data,[0,0.1,0.5,0.9,1.]))

data = pd.DataFrame(np.random.rand(1000,4))
print(data.describe())
col = data[2]
print(col[np.abs(col)>3])
print(data[(np.abs(data)>3).any(1)])
data[np.abs(data>3)] = np.sign(data)*3
print(data.describe())
print(np.sign(data).head())

df = pd.DataFrame(np.arange(5*4).reshape((5,4)))
sampler = np.random.permutation(5)
print(sampler)
print(df)
print(df.take(sampler))
print(df.sample(n=3))
choices = pd.Series([5,7,-1,6,4])
draws = choices.sample(n=10, replace = True)
print(draws)

df = pd.DataFrame({'key': ['b','b','a','c','a','b'],
                    'data1': range(6)})
print(df)
print(pd.get_dummies(df['key']))
dummies = pd.get_dummies(df['key'], prefix = 'key')
df_with_dummy = df[['data1']].join(dummies)
print(df_with_dummy)

mnames = ['movie_id', 'title', 'genres']
dsurl = 'https://raw.githubusercontent.com/wesm/pydata-book/3rd-edition/datasets/'
movurl = dsurl + 'movielens/movies.dat'
movies = pd.read_table(movurl,sep ='::', header =None, names = mnames)
print(movies[:10])
all_genres = []
for x in movies.genres:
    all_genres.extend(x.split('|'))
genres = pd.unique(all_genres)
print(genres)
zero_matrix = np.zeros((len(movies),len(genres)))
dummies = pd.DataFrame(zero_matrix, columns = genres)
gen = movies.genres[0]
print(gen.split('|'))
print(dummies.columns.get_indexer(gen.split('|')))
for i, gen in enumerate(movies.genres):
    indices = dummies.columns.get_indexer(gen.split('|'))
    dummies.iloc[i, indices] = 1
movies_windic = movies.join(dummies.add_prefix('Genre_'))
print(movies_windic.iloc[0])

np.random.seed(12345)
values = np.random.rand(10)
print(values)
print(bins)
print(pd.get_dummies(pd.cut(values,bins)))

val = 'a,b, guido'
print(val.split(','))
pieces = [x.strip() for x in val.split(',')]
first, second, third = pieces
print(first +'::'+second +'::'+third)
print('::'.join(pieces))
print('guido' in val)
print(val.index(','))
print(val.find(':'))
#val.index(':') produces an error
print(val.count(','))
print(val.replace(',','::'))
print(val.replace(',',''))

text = "foo     bar\t baz \tqux"
print(re.split('\s+', text))
regex = re.compile('\s+')
print(regex.split(text))
print(regex.findall(text))

text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}'
# re.IGNORECASE makes the regex case=insensitive
regex = re.compile(pattern, flags = re.IGNORECASE)
print(regex.findall(text))
m = regex.search(text)
print(m)
print(text[m.start():m.end()])
print(regex.match(text))
print(regex.sub('REDACTED', text))
pattern = r'([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})'
regex = re.compile(pattern, flags = re.IGNORECASE)
m = regex.match('wesm@bright.net')
print(m.groups())
print(regex.findall(text))
print(regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text))

data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com',
        'Rob': 'rob@gmail.com', 'Wes': np.nan}
data = pd.Series(data)
print(data)
print(data.isnull())
print(data.str.contains('gmail'))
print(pattern)
print(data.str.findall(pattern,flags=re.IGNORECASE))
matches = data.str.match(pattern, flags = re.IGNORECASE)
print(type(matches))
print(matches)
#matches.str.get(1)
#print(matches.str[0])
#print(data.str[:5])
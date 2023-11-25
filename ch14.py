import numpy as np
import pandas as pd
from collections import defaultdict
from collections import Counter
import json
import seaborn as sns
import matplotlib.pyplot as plt


def pltpnc(t):
    plt.pause(t)
    plt.close()

#path = r"C:\Users\camac\Dropbox (Personal)\Python Scripts\Python for Data Analysis\example.txt"
path = r"C:\Users\Frank Camacho\Dropbox (Personal)\Python Scripts\Python for Data Analysis\example.txt"

print(open(path).readline())
records = [json.loads(line) for line in open(path)]
print('\n')
print(records[0])

time_zones = [rec['tz'] for rec in records if 'tz' in rec]
print(time_zones[:10])

def get_counts(sequence):
	counts = {}
	for x in sequence:
		if x in counts:
			counts[x] +=1
		else:
			counts[x] = 1
	return counts

def get_counts2(sequence):
	counts = defaultdict(int)
	for x in sequence:
		counts[x] += 1
	return counts

counts = get_counts(time_zones)
print(counts['America/New_York'])
print(len(time_zones))

def top_counts(count_dict, n = 10):
	value_key_pairs = [(count, tz) for tz, count in count_dict.items()]
	value_key_pairs.sort()
	return value_key_pairs[-n:]

print(top_counts(counts))
counts = Counter(time_zones)
print(counts.most_common(10))

frame = pd.DataFrame(records)
print(frame.info())
print(frame['tz'][:10])
tz_counts = frame['tz'].value_counts()
print(tz_counts[:10])
clean_tz = frame['tz'].fillna('Missing')
clean_tz[clean_tz ==''] = 'Unknown'
tz_counts = clean_tz.value_counts()
print(tz_counts[:10])
subset = tz_counts[:10]
sns.barplot(y = subset.index, x = subset.values)
pltpnc(0.5)
print(frame['a'][1])
print(frame['a'][50])
print(frame['a'][51][:50]) # long line

results = pd.Series([x.split()[0] for x in frame.a.dropna()])
print(results[:5])
print(results.value_counts()[:8])
cframe = frame[frame.a.notnull()]
cframe['os'] = np.where(cframe['a'].str.contains('Windows'), 
						'Windows','Not Windows')
print(['os'][:5])
by_tz_os = cframe.groupby(['tz', 'os'])
agg_counts = by_tz_os.size().unstack().fillna(0)
print(agg_counts[:10])						
indexer = agg_counts.sum(1).argsort()
print(indexer)
count_subset = agg_counts.take(indexer[-10:])
print(count_subset)
print(agg_counts.sum(1).nlargest(10))
count_subset = count_subset.stack()
count_subset.name = 'total'
count_subset = count_subset.reset_index()
print(count_subset[:10])
sns.barplot(x='total', y='tz', hue='os', data=count_subset)
pltpnc(0.5)
def norm_total(group):
	group['normed_total'] = group.total/group.total.sum()
	return group
results = count_subset.groupby('tz').apply(norm_total)
sns.barplot(x = 'normed_total', y = 'tz', hue = 'os', data = results)
g = count_subset.groupby('tz')
results2 = count_subset.total/g.total.transform('sum')

pd.options.display.max_rows = 3

url = 'https://raw.githubusercontent.com/wesm/pydata-book/3rd-edition/'
urlusers = url+'datasets/movielens/users.dat'
urlratings = url + 'datasets/movielens/ratings.dat'
urlmovies = url+'datasets/movielens/movies.dat'
unames = ["user_id", "gender", "age", "occupation", "zip"]
users = pd.read_table(urlusers, sep="::",
                      header=None, names=unames, engine="python")

rnames = ["user_id", "movie_id", "rating", "timestamp"]
ratings = pd.read_table(urlratings, sep="::",
                        header=None, names=rnames, engine="python")

mnames = ["movie_id", "title", "genres"]
movies = pd.read_table(urlmovies, sep="::",
                       header=None, names=mnames, engine="python")
print(users[:5])
print(ratings[:5])
print(movies[:5])
print(ratings)
data = pd.merge(pd.merge(ratings, users), movies)
print(data)
print(data.iloc[0])
mean_ratings = data.pivot_table('rating', index = 'title',
								columns = 'gender', aggfunc = 'mean')
print(mean_ratings[:5])
ratings_by_title = data.groupby('title').size()
print(ratings_by_title[:10])
active_titles = ratings_by_title.index[ratings_by_title >= 250]
print(active_titles)
mean_ratings = mean_ratings.loc[active_titles]
print(mean_ratings)
top_female_ratings = mean_ratings.sort_values(by = 'F', ascending = False)
print(top_female_ratings[:10])
mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by = 'diff')
print(sorted_by_diff[:10])
print(sorted_by_diff[::-1][:10])
rating_std_by_title = data.groupby('title')['rating'].std()
rating_std_by_title = rating_std_by_title.loc[active_titles]
print(rating_std_by_title.sort_values(ascending = False)[:10])
#yobpath = r"C:\Users\camac\Dropbox (Personal)\Python Scripts\Python for Data Analysis\yobNames"
yobpath = r"C:\Users\Frank Camacho\Dropbox (Personal)\Python Scripts\Python for Data Analysis\yobNames"

yob1880 = yobpath + "\yob1880.txt"
names1880 = pd.read_csv(yob1880, names = ['name', 'sex', 'births'])

print(names1880.head())
print(names1880.groupby('sex').births.sum())

years = range(1880,2011)
pieces = []
columns = ['name', 'sex', 'births']

for year in years:
	path = "\yob%d.txt" % year
	path = yobpath + path
	frame = pd.read_csv(path, names = columns)
	frame['year'] = year
	pieces.append(frame)

names = pd.concat(pieces, ignore_index = True)
print(names)
total_births = names.pivot_table('births', index = 'year', columns = 'sex', aggfunc = sum)
total_births.plot(title ='Total births by sex and year')
pltpnc(0.5)
def add_prop(group):
	group['prop'] = group.births/group.births.sum()
	return group
names = names.groupby(['year', 'sex']).apply(add_prop)
print(names)
print(names.groupby(['year', 'sex']).prop.sum())

def get_top1000(group):
	return group.sort_values(by = 'births', ascending = False)[:1000]
grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)
top1000.reset_index(inplace = True,drop = True)
print(top1000)

pieces = []
for year, group in names.groupby(['year', 'sex']):
	pieces.append(group.sort_values(by = 'births', ascending = False)[:1000])
top1000 = pd.concat(pieces, ignore_index = True)
print(top1000)

boys = top1000[top1000.sex =='M']
girls = top1000[top1000.sex =='F']
total_births = top1000.pivot_table('births', index = 'year',
									columns = 'name',
									aggfunc = sum)
subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots = True, figsize = (12,10), grid = False,
							title = "Number of births per year")
pltpnc(0.5)
print(total_births.info())
table = top1000.pivot_table('prop', index= 'year',
							columns = 'sex', aggfunc = sum)
table.plot(title = 'Sum of table1000.prop by year and sex',
					yticks = np.linspace(0,1.2,13), xticks = range(1880, 2020, 10))
df = boys[boys.year == 2010]
print(df)
prop_cumsum = df.sort_values(by = 'prop', ascending = False).prop.cumsum()
print(prop_cumsum[:10])

df = boys[boys.year == 1900]
in1900 = df.sort_values("prop", ascending=False).prop.cumsum()
print(in1900.searchsorted(0.5) + 1)

def get_quantile_count(group, q=0.5):
    group = group.sort_values("prop", ascending=False)
    return group.prop.cumsum().searchsorted(q) + 1

diversity = top1000.groupby(["year", "sex"]).apply(get_quantile_count)
diversity = diversity.unstack()

print(diversity.head())
diversity.plot(title = "Number of popular names in tp 50%")

get_last_letter = lambda x:x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'

table = names.pivot_table('births', index = last_letters, 
						columns = ['sex', 'year'], aggfunc = sum)

subtable = table.reindex(columns = list(range(1910,2011,50)), level = 'year')
print(subtable.head())
print(subtable.sum())
letter_prop = subtable/subtable.sum()
print(letter_prop)

fig, axes = plt.subplots(2,1,figsize =(10,8))
letter_prop['M'].plot(kind = 'bar', rot = 0, ax = axes[0], title = 'Male')
letter_prop['F'].plot(kind = 'bar', rot = 0, ax = axes[1], title = 'Female',
						legend = False)
letter_prop = table/table.sum()
dny_ts = letter_prop.loc[['d','n','y'], 'M'].T
print(dny_ts.head())
dny_ts.plot()
pltpnc(0.5)

all_names = pd.Series(top1000.name.unique())
lesley_like = all_names[all_names.str.lower().str.contains('lesl')]
print(lesley_like)

filtered = top1000[top1000.name.isin(lesley_like)]
print(filtered.groupby('name').births.sum())
table = filtered.pivot_table('births', index = 'year',
								columns = 'sex', aggfunc = 'sum')
table = table.div(table.sum(1), axis = 0)
print(table.tail())
table.plot(style = {'M': 'k-', 'F': 'k--'})

urlusda = r"C:\Users\Frank Camacho\Dropbox (Personal)\Python Scripts\Python for Data Analysis\database.json"
#urlusda = r"C:\Users\camac\Dropbox (Personal)\Python Scripts\Python for Data Analysis\database.json"
db = json.load(open(urlusda))
print(len(db))
print(db[0].keys())
print(db[0]["nutrients"][0])
nutrients = pd.DataFrame(db[0]["nutrients"])
print(nutrients.head(7))
info_keys = ["description", "group", "id", "manufacturer"]
info = pd.DataFrame(db, columns=info_keys)
print(info.head())
print(info.info())
print(pd.value_counts(info["group"])[:10])
nutrients = []
for rec in db:
    fnuts = pd.DataFrame(rec["nutrients"])
    fnuts["id"] = rec["id"]
    print(nutrients.append(fnuts))
nutrients = pd.concat(nutrients, ignore_index=True)
print(nutrients)
print(nutrients.duplicated().sum())
nutrients = nutrients.drop_duplicates()
col_mapping = {"description" : "food",
               "group"       : "fgroup"}
info = info.rename(columns=col_mapping, copy=False)
print(info.info())
col_mapping = {"description" : "nutrient",
               "group" : "nutgroup"}
nutrients = nutrients.rename(columns=col_mapping, copy=False)
print(nutrients)
ndata = pd.merge(nutrients, info, on="id")
print(ndata.info())
print(ndata.iloc[30000])
fig = plt.figure()
result = ndata.groupby(["nutrient", "fgroup"])["value"].quantile(0.5)
result['Zinc, Zn'].sort_values().plot(kind = 'barh')
pltpnc(0.5)
by_nutrient = ndata.groupby(['nutgroup','nutrient'])
get_maximum = lambda x: x.loc[x.value.idxmax()]
get_minimum = lambda x: x.loc[x.value.idxmin()]
max_foods = by_nutrient.apply(get_maximum)[['value', 'food']]
max_foods.food = max_foods.food.str[:50]
print(max_foods.loc['Amino Acids']['food'])



urlP0sALL = url + 'datasets/fec/P00000001-ALL.csv'
fec = pd.read_csv(urlP0sALL)
print(fec.info())
print(fec.iloc[123456])
unique_cands = fec.cand_nm.unique()
print(unique_cands)
print(unique_cands[2])

parties = {'Bachmann, Michelle': 'Republican',
'Cain, Herman': 'Republican',
'Gingrich, Newt': 'Republican',
'Huntsman, Jon': 'Republican',
'Johnson, Gary Earl': 'Republican',
'McCotter, Thaddeus G': 'Republican',
'Obama, Barack': 'Democrat',
'Paul, Ron': 'Republican',
'Pawlenty, Timothy': 'Republican',
'Perry, Rick': 'Republican',
"Roemer, Charles E. 'Buddy' III": 'Republican',
'Romney, Mitt': 'Republican',
'Santorum, Rick': 'Republican'}
print(fec.cand_nm[123456:123461])
print(fec.cand_nm[123456:123461].map(parties))
fec['party'] = fec.cand_nm.map(parties)
print(fec['party'].value_counts())
print((fec.contb_receipt_amt>0).value_counts)
fec = fec[fec.contb_receipt_amt > 0]
fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]

fec.contbr_occupation.value_counts()[:10]
occ_mapping = {
	'INFORMATION REQUESTED PER BEST EFFORTS': 'NOT PROVIDED',
	'INFORMATION REQUESTED' : 'NOT PROVIDED',
	'INFORMATION REQUESTED (BEST EFFORTS)' : 'NOT PROVIDED',
	'C.E.O' : 'CEO'
}

f = lambda x: occ_mapping.get(x,x)
fec.contbr_occupation = fec.contbr_occupation.map(f)

emp_mapping = {
	'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
	'INFORMATION REQUESTED' : 'NOT PROVIDED',
	'SELF' : 'SELF-EMPLOYED',
	'SELF EMPLOYED' : 'SELF-EMPLOYED',
}

f = lambda x:emp_mapping.get(x,x)
fec.contbr_employer = fec.contbr_employer.map(f)
by_occupation = fec.pivot_table('contb_receipt_amt',
								index = 'contbr_occupation',
								columns = 'party', aggfunc = 'sum')
over_2mm = by_occupation[by_occupation.sum(1)>2000000]
print(over_2mm)
over_2mm.plot(kind = 'barh')
pltpnc(3)
def get_top_amounts(group, key, n = 5):
    totals = group.groupby(key)['contb_receipt_amt'].sum()
    return totals.nlargest(n)
grouped = fec_mrbo.groupby('cand_nm')
print(grouped.apply(get_top_amounts, 'contbr_occupation', n = 7))
print(grouped.apply(get_top_amounts, 'contbr_employer', n = 10))

bins = np.array([10**n if n>0 else n for n in range(0,8)])
labels = pd.cut(fec_mrbo.contb_receipt_amt,bins)
print(labels)
grouped = fec_mrbo.groupby(['cand_nm', labels])
grouped.size().unstack(0)
bucket_sums = grouped.contb_receipt_amt.sum().unstack(0)
normed_sums = bucket_sums.div(bucket_sums.sum(axis = 1), axis = 0)
print(normed_sums)
normed_sums[:-2].plot(kind = 'barh')
pltpnc(3)

grouped = fec_mrbo.groupby(['cand_nm', 'contbr_st'])
totals = grouped.contb_receipt_amt.sum().unstack(0).fillna(0)
totals = totals[totals.sum(1) > 100000]
print(totals[:10])
percent = totals.div(totals.sum(1), axis = 0)
print(percent[:10])
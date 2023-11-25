import os
import sys
import csv
import requests
import json
from urllib.request import urlopen
import pandas as pd
import numpy as np
from lxml import objectify
from io import StringIO
import tables
import sqlite3
import sqlalchemy as sqla

url = 'https://raw.githubusercontent.com/wesm/pydata-book/3rd-edition/examples/'

def urlfun(n):
    return url + 'ex' + str(n) + '.csv'


urlex1 = urlfun(1)
df1ex1 = pd.read_csv(urlex1)
print(df1ex1)
df2ex1 = pd.read_table(urlex1, sep = ',')
print(df2ex1)

urlex2 = urlfun(2)
df1ex2 = pd.read_csv(urlex2, header = None)
print(df1ex2)
df2ex2 = pd.read_csv(urlex2, names = ['a', 'b', 'c', 'd', 'message'])
print(df2ex2)
names = ['a','b','c','d', 'message']
df3ex2 = pd.read_csv(urlex2, names = names, index_col = 'message')

urlcsv_mindex = url + 'csv_mindex.csv'
parsed = pd.read_csv(urlcsv_mindex, index_col = ['key1','key2'])
print(parsed)

urlex3 = url + 'ex3.txt'
#ex3list = list(open(urlex3))
result = pd.read_table(urlex3, sep='\s+')

print(result)

urlex4 = urlfun(4)
df1ex4 = pd.read_csv(urlex4, skiprows = [0,2,3])

urlex5 = urlfun(5)
result = pd.read_csv(urlex5)
print(result)
df1ex5 = pd.isnull(result)
print(df1ex5)
result = pd.read_csv(urlex5, na_values = ['NULL'])
print(result)
sentinels = {'message':['foo','NA'], 'something': ['two']}
df2ex5 = pd.read_csv(urlex5, na_values = sentinels)
print(df2ex5)

pd.options.display.max_rows = 10

urlex6 = urlfun(6)
result = pd.read_csv(urlex6)
print(result)
df1ex6 = pd.read_csv(urlex6, nrows =5)
print(df1ex6)
chunker = pd.read_csv(urlex6,chunksize = 1000)
tot = pd.Series([],dtype = 'float64')
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value = 0)
tot = tot.sort_values(ascending = False)
print(tot[:10])

data  = pd.read_csv(urlex5)
print(data)

ch6path = os.path.dirname(os.path.realpath(__file__))
ch6path = ch6path.removesuffix('ch6.py') + "\ch6examples"
ch6pathalt = ch6path.removesuffix('ch6.py') + '\ch6examples'
#ch6path = repr(ch6path)
#ch6pathhome = r"C:\Users\camac\Dropbox\Python Scripts\Python for Data Analysis\ch6examples"
#ch6pathwork = r"C:\Users\Frank Camacho\Dropbox (Personal)\Python Scripts\Python for Data Analysis\ch6examples"

if not os.path.exists(ch6path):
    os.makedirs(ch6path)

data.to_csv(ch6path + "\out.csv")


data.to_csv(sys.stdout, sep = '|')
data.to_csv(sys.stdout, na_rep = 'NULL')
data.to_csv(sys.stdout, index = False, header = False)
data.to_csv(sys.stdout, index = False, columns = ['a', 'b', 'c'])

dates = pd.date_range('1/1/2000', periods = 7)
ts = pd.Series(np.arange(7), index = dates)
ts.to_csv(ch6path + "\series.csv")
ts.to_csv(sys.stdout)

urlex7 = urlfun(7)
df1ex7 = pd.read_csv(urlex7)
df1ex7.to_csv(ch6path + "\ex7.csv")
f = open(ch6path + "\ex7.csv")
reader = csv.reader(f)
for line in reader:
    print(line)
f.close()

with open(ch6path + "\ex7.csv") as f:
    lines = list(csv.reader(f))
f.close()

header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}

print(data_dict)

class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL

f = open(ch6path + "\ex7.csv")
reader = csv.reader(f,dialect = my_dialect)
reader = csv.reader(f,delimiter = '|')
with open('mydata.csv', 'w') as f:
    writer = csv.writer(f, dialect = my_dialect)
    writer.writerow(('one', 'two', 'three'))
    writer.writerow(('1', '2', '3'))
    writer.writerow(('4', '5', '6'))
    writer.writerow(('7', '8', '9'))

obj = """
{"name": "Wes",
"places_lived": ["United States", "Spain", "Germany"],
"pet": null,
"siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},
            {"name": "Katie", "age": 38,
            "pets": ["Sixes", "Stache", "Cisco"]}]
}
"""
result = json.loads(obj)
print(result)

asjson = json.dumps(result)

siblings = pd.DataFrame(result['siblings'], columns = ['name', 'age'])
print(siblings)

urljson = url + 'example.json'
response = urlopen(urljson)
jsonex = json.loads(response.read())
print(jsonex)
data = pd.read_json(urljson)
print(data)
print(data.to_json())
print(data.to_json(orient = 'records'))

urlFFBL = url + 'fdic_failed_bank_list.html'
tables = pd.read_html(urlFFBL)
print(len(tables))
failures = tables[0]
print(failures.head())
close_timestamps = pd.to_datetime(failures['Closing Date'])
print(close_timestamps.dt.year.value_counts())

urlds = 'https://raw.githubusercontent.com/wesm/pydata-book/3rd-edition/datasets/'
urlMTA = urlds + 'mta_perf/Performance_MNR.xml'
pathMTA = ch6path + '\mta_perf'
if not os.path.exists(pathMTA):
    os.makedirs(pathMTA)
response = requests.get(urlMTA)
#pathPMNR = ch6path + "\mta_perf\Performance_MNR.xml"
pathPMNR = ch6path + '\mta_perf\Performance_MNR.xml'
with open(pathPMNR, 'wb') as file:
    file.write(response.content)
parsed = objectify.parse(open(pathPMNR))
root = parsed.getroot()
data = []
skip_fields = ['PARENT_SEQ','INDICATOR_SEQ',
                'DESIRED_CHANGE', 'DECIMAL_PLACES']
for elt in root.INDICATOR:
    el_data = {}
    for child in elt.getchildren():
        if child.tag in skip_fields:
            continue
        el_data[child.tag] = child.pyval
        data.append(el_data)
perf = pd.DataFrame(data)
print(perf.head())

tag = '<a href="http://www.google.com">Google</a>'
root = objectify.parse(StringIO(tag)).getroot()
print(root)
print(root.get('href'))
print(root.text)

frame = pd.read_csv(urlex1)
frame.to_pickle(ch6path + r"\frame_pickle")

frame = pd.DataFrame({'a':np.random.randn(100)})
store = pd.HDFStore('mydata.h5')
store['obj1'] = frame 
store['obj1_col'] = frame['a']
print(store['obj1'])
print(store['obj1_col'])
store.put('obj2', frame, format = 'table')
store.select('obj2', where = ['index >=10 and index <=15'])
store.close()

frame.to_hdf('mydata.h5', 'obj3', format = 'table')
A = pd.read_hdf('mydata.h5', 'obj3', where = ['index < 5'])
print(A)

urlxclex1 = url + 'ex1.xlsx'
xlsx = pd.ExcelFile(urlxclex1)
print(xlsx)
print(pd.read_excel(xlsx, 'Sheet1'))
frame = pd.read_excel(urlxclex1, 'Sheet1')
print(frame)
urlxclex2 = url + 'ex2.xlsx'
xlsx2path = ch6path + '\ex2.xlsx'
writer = pd.ExcelWriter(xlsx2path)
frame.to_excel(writer,'Sheet1')
writer.save()
frame.to_excel(ch6path + '\ex2ii.xlsx')

URL = 'https://api.github.com/repos/pandas-dev/pandas/issues'
resp = requests.get(URL)
print(resp)

data = resp.json()
print(data[0]['title'])
issues = pd.DataFrame(data,columns=['number','title','labels', 'state'])
print(issues)

query = """
        CREATE TABLE IF NOT EXISTS test
        (a VARCHAR(20), B VARCHAR(20),
        c REAL,         d INTEGER
        ); """
con = sqlite3.connect('mydata.sqlite')
con.execute(query)
con.commit()
data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahasee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]

stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"
con.executemany(stmt, data)
con.commit()
cursor = con.execute('select * from test')
rows = cursor.fetchall()
print(rows)
print(cursor.description)
pd.DataFrame(rows, columns = [x[0] for x in cursor.description])


db = sqla.create_engine('sqlite:///mydata.sqlite')
pd.read_sql('select * from test', db)
print(pd.read_sql('select * from test', db))

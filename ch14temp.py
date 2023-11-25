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

url = 'https://raw.githubusercontent.com/wesm/pydata-book/3rd-edition/'

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
fec.contbr_occupation = fec.conbr_occupation.map(f)

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
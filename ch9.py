import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import randn
from datetime import datetime
from io import BytesIO

def pltpnc(t):
    plt.pause(t)
    plt.close()

data = np.arange(10)
print(data)
plt.plot(data)
pltpnc(0.5)
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
plt.plot(np.random.randn(50).cumsum(), 'k--')
_ = ax1.hist(np.random.randn(100), bins = 20, color = 'k', alpha = 0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3*np.random.randn(30))
pltpnc(0.5)

fig, axes = plt.subplots(2,3)
pltpnc(0.5)

fig, axes = plt.subplots(2,2,sharex = True, sharey = True)
for i in range(2):
    for j in range(2):
        axes[i,j].hist(np.random.randn(500), bins = 50, color = 'k', alpha = 0.5)
plt.subplots_adjust(wspace = 0, hspace = 0)
pltpnc(0.5)

plt.plot(randn(30).cumsum(),'ko--')
pltpnc(0.5)

plt.plot(randn(30).cumsum(), color = 'k', linestyle = 'dashed', marker ='o')
pltpnc(0.5)

data = np.random.randn(30).cumsum()
plt.plot(data,'k--', label = 'Default')
plt.plot(data, 'k-', drawstyle = 'steps-post', label = 'steps-post')
plt.legend(loc = 'best')
pltpnc(0.5)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(np.random.randn(1000).cumsum())
ticks = ax.set_xticks([0,250,500,750,1000])
labels = ax.set_xticklabels(['one','two','three','four','five'],
                            rotation = 30, fontsize = 'small')
#ax.set_title('My first matplotlib plot')
#ax.set_xlabel('Stages')
props = {
    'title': 'My first matplotlib plot',
    'xlabel': 'Stages'
}
ax.set(**props)
pltpnc(0.5)

fig = plt.figure(); ax = fig.add_subplot(1,1,1)
ax.plot(randn(1000).cumsum(), 'k', label = 'one')
ax.plot(randn(1000).cumsum(), 'k--', label = 'two')
ax.plot(randn(1000).cumsum(), 'k', label = 'three')
ax.legend(loc = 'best')
pltpnc(0.5)

#ax.test(x,y, 'Hello world!',
#            family = 'monospace', fontsize = 10)
url = 'https://raw.githubusercontent.com/wesm/pydata-book/3rd-edition/examples/'
urlspx = url + 'spx.csv'
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
data = pd.read_csv(urlspx, index_col = 0, parse_dates = True)
spx = data['SPX']
spx.plot(ax = ax, style = 'k-')
crisis_data = [
    (datetime(2007,10,11), 'Peak of bull market'),
    (datetime(2008,3,12), 'Bear Stearns Fails'),
    (datetime(2008, 9, 15), 'Lehman Bankruptcy')
]

for date, label in crisis_data:
    ax.annotate(label, xy = (date, spx.asof(date)+ 75),
                xytext = (date, spx.asof(date) + 225),
                arrowprops = dict(facecolor = 'black', headwidth = 4, width = 2,
                headlength = 4),
                horizontalalignment = 'left', verticalalignment  = 'top')
ax.set_xlim(['1/1/2007', '1/1/2011'])
ax.set_ylim([600,1800])
ax.set_title('Important dates in the 2008-2009 financial crisis')
pltpnc(0.5)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
rect = plt.Rectangle((0.2,0.75),0.4,0.15, color = 'k', alpha = 0.3)
circ = plt.Circle((0.7,0.2), 0.15, color = 'b', alpha = 0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]], color = 'g', alpha =0.5)
ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)
pltpnc(0.5)

#plt.savefig('figpath.svg')
#plt.savefig('figpath.png', dpi = 400, bbox_inches = tight)
#buffer = BytesIO()
#plt.savefig(buffer)
#plot_data = buffer.getvalue()

s = pd.Series(np.random.randn(10).cumsum(), index = np.arange(0,100,10))
s.plot()
pltpnc(1)

df = pd.DataFrame(np.random.randn(10,4).cumsum(0),
                    columns = ['A', 'B', 'C', 'D'], 
                    index = np.arange(0,100,10))
df.plot()
pltpnc(1)

fig, axes = plt.subplots(2,1)
data = pd.Series(np.random.rand(16), index = list('abcdefghikjlmnop'))
data.plot.bar(ax = axes[0], color = 'k', alpha = 0.7)
data.plot.barh(ax = axes[1], color = 'k', alpha = 0.7)
pltpnc(1)
df = pd.DataFrame(np.random.rand(6,4),
                    index = ['one', 'two', 'three', 'four', 'five', 'six'],
                    columns = pd.Index(['A','B', 'C', 'D'], name = 'Genus'))
print(df)
pltpnc(1)

df.plot.barh(stacked = True, alpha = 0.5)
pltpnc(1)
urltips = url + 'tips.csv'
tips = pd.read_csv(urltips)
party_counts = pd.crosstab(tips['day'], tips['size'])
print(party_counts)
party_counts = party_counts.loc[:,2:5]
party_pcts = party_counts.div(party_counts.sum(1), axis = 0)
print(party_pcts)
party_pcts.plot.bar()
pltpnc(1)

tips['tip_pct'] = tips['tip']/(tips['total_bill'] - tips['tip'])
print(tips.head())
sns.barplot(x = 'tip_pct', y = 'day', data = tips, orient = 'h')
pltpnc(2)
sns.barplot(x = 'tip_pct', y = 'day', hue = 'time', data = tips, orient = 'h')
sns.set(style = "whitegrid")
pltpnc(2)

tips['tip_pct'].plot.hist(bins = 50)
pltpnc(1)
tips['tip_pct'].plot.density()
pltpnc(1)

comp1 = np.random.normal(0,1,size = 200)
comp2 = np.random.normal(10, 2, size = 200)
values = pd.Series(np.concatenate([comp1, comp2]))
sns.distplot(values, bins = 100, color = 'k')
pltpnc(1)

urlmacro = url + 'macrodata.csv'
macro = pd.read_csv(urlmacro)
data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
trans_data = np.log(data).diff().dropna()
trans_data[-5:]
#sns.regplot('m1', 'unemp', trans_data)
#plt.title('Changes in log %s versus log %s' % ('m1', 'unemp'))
#pltpnc(1)
sns.pairplot(trans_data, diag_kind = 'kde', plot_kws = {'alpha':0.2})
pltpnc(2)
sns.factorplot(x = 'day', y = 'tip_pct', hue = 'time', col = 'smoker',
                kind = 'bar', data = tips[tips.tip_pct < 1])
pltpnc(1)
sns.factorplot(x = 'day', y= 'tip_pct', row = 'time', 
                col = 'smoker',
                kind = 'bar', data = tips[tips.tip_pct<1])
sns.factorplot(x = 'tip_pct', y = 'day', kind = 'box',
                data = tips[tips.tip_pct < 0.5])
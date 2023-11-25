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

now = datetime.now()
print(now)
print(now.year, now.month, now.day)

delta = datetime(2011,1,7) - datetime(2008,6,24,8,15)
print(delta)
print(delta.days)
print(delta.seconds)
start = datetime(2011,1,7)
print(start + timedelta(12))
print(start - 2*timedelta(12))
stamp = datetime(2011,1,3)
print(str(stamp))
print(stamp.strftime('%Y-%m-%d'))
value = '2011-01-03'
print(datetime.strptime(value,'%Y-%m-%d'))
datestrs = ['7/6/2011', '8/6/2011']
print([datetime.strptime(x, '%m/%d/%Y') for x in datestrs])
print(parse('2011-01-03'))
print('Jan 31, 1997 10:45 PM')
parse('6/12/2011', dayfirst = True)
datestrs = ['2011-07-06 12:00:00', '2011-08-06 00:00:00']
print(pd.to_datetime(datestrs))
idx = pd.to_datetime(datestrs + [None])
print(idx)
print(idx[2])
print(pd.isnull(idx))

dates = [datetime(2011, 1, 2), datetime(2011,1,5),
        datetime(2011,1,7), datetime(2011,1,8),
        datetime(2011,1,10), datetime(2011,1,12)]
ts = pd.Series(np.random.randn(6), index = dates)
print(ts)
print(ts.index)
print(ts + ts[::2])
print(ts)
print(ts[::2])
print(ts.index.dtype)
stamp = ts.index[0]
print(stamp)

stamp = ts.index[2]
print(ts[stamp])
print(ts['1/10/2011'])
print(ts['20110110'])
longer_ts = pd.Series(np.random.randn(1000),
                        index = pd.date_range('1/1/2000', periods = 1000))
print(longer_ts)
print(longer_ts['2001'])
print(longer_ts['2001-05'])
ts[datetime(2011,1,7):]
print(ts)
print(ts['1/6/2011':'1/11/2011'])
print(ts.truncate(after = '1/9/2011'))
dates = pd.date_range('1/1/2000', periods = 100, freq = 'W-WED')
long_df = pd.DataFrame(np.random.randn(100,4),
                        index = dates,
                        columns = ['Colorado', 'Texas',
                                'New York', 'Ohio'])
print(long_df.loc['5-2001'])
dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000',
                            '1/2/2000', '1/3/2000'])
dup_ts = pd.Series(np.arange(5), index = dates)
print(dup_ts)
print(dup_ts.index.is_unique)
print(dup_ts['1/3/2000'])
print(dup_ts['1/2/2000'])

grouped = dup_ts.groupby(level = 0)
print(grouped.mean())
print(grouped.count())
print(ts)
resampler = ts.resample('D')
print(resampler)

index = pd.date_range('2012-04-01', '2012-06-01')
print(index)
print(pd.date_range(start = '2012-04-01', periods = 20))
print(pd.date_range(end = '2012-06-01', periods = 20))
print(pd.date_range('2000-01-01', '2000-12-01', freq = 'BM'))
print(pd.date_range('2012-05-02 12:56:31', periods = 5))
print(pd.date_range('2012-05-02 12:56:31', periods = 5, normalize = True))

hour = Hour()
print(hour)
four_hours = Hour(4)
print(four_hours)
print(pd.date_range('2000-01-01','2000-01-03 23:59', freq = '4h'))
print(Hour(2) + Minute(30))
print(pd.date_range('2000-01-01', periods = 10, freq = '1h30min'))
rng = pd.date_range('2012-01-01', '2012-09-01', freq = 'WOM-3FRI')
print(list(rng))

ts = pd.Series(np.random.randn(4),
                index = pd.date_range('1/1/2000', periods = 4, freq = 'M'))
print(ts)
print(ts.shift(2))
print(ts.shift(-2))
print(ts/ts.shift(1) -1)
print(ts.shift(2, freq = 'M'))
print(ts.shift(3,freq = 'D'))
print(ts.shift(1,freq = '90T'))

now = datetime(2011,11,17)
print(now + 3*Day())
print(now + MonthEnd())
print(now + MonthEnd(2))

offset = MonthEnd()
print(offset.rollforward(now))
print(offset.rollback(now))
ts = pd.Series(np.random.randn(20), 
        index = pd.date_range('1/15/2000', periods = 20, freq = '4d'))
print(ts)
print(ts.groupby(offset.rollforward).mean())
print(ts.resample('M').mean())

print(pytz.common_timezones[-5:])
tz = pytz.timezone('America/New_York')
print(tz)
rng = pd.date_range('3/9/2012 9:30', periods = 6, freq = 'D')
ts = pd.Series(np.random.randn(len(rng)), index = rng)
print(ts)
print(ts.index.tz)
print(pd.date_range('3/9/2012 9:30', periods = 10, freq = 'D', tz = 'UTC'))
print(ts)
ts_utc = ts.tz_localize('UTC')
print(ts_utc)
print(ts_utc.index)
ts_eastern = ts.tz_localize('America/New_York')
print(ts_eastern)
print(ts_eastern.tz_convert('UTC'))
print(ts_eastern.tz_convert('Europe/Berlin'))
print(ts.index.tz_localize('Asia/Shanghai'))
rng = pd.date_range('3/7/2012 9:30', periods = 10, freq = 'B')
ts = pd.Series(np.random.randn(len(rng)), index = rng)
print(ts)
ts1 = ts[:7].tz_localize('Europe/London')
ts2 = ts1[2:].tz_convert('Europe/Moscow')
result = ts1 + ts2
print(result.index)

p = pd.Period(2007, freq = 'A-DEC')
print(p)
print(p-2)
print(p+5)
print(pd.Period('2014', freq = 'A-DEC') - p)
rng = pd.period_range('2000-01-01', '2000-06-30', freq = 'M')
print(rng)
print(pd.Series(np.random.randn(6), index = rng))
values = ['2001Q3','2002Q2','2003Q1']
index = pd.PeriodIndex(values, freq = 'Q-DEC')
print(index)
p = pd.Period('2007', freq = 'A-DEC')
print(p)
print(p.asfreq('M', how = 'start'))
print(p.asfreq('M', how = 'end'))
p = pd.Period('2007', freq = 'A-JUN')
print(p)
print(p.asfreq('M','start'))
print(p.asfreq('M', how = 'end'))
p = pd.Period('Aug-2007', 'M')
print(p)
print(p.asfreq('A-JUN'))
rng = pd.period_range('2006', '2009', freq = 'A-DEC')
ts = pd.Series(np.random.randn(len(rng)), index = rng)
print(ts)
print(ts.asfreq('M', how = 'start'))
print(ts.asfreq('B', how = 'start'))
p = pd.Period('2012Q4', freq = 'Q-JAN')
print(p)
print(p.asfreq('D', 'start'))
print(p.asfreq('D', 'end'))
p4pm = (p.asfreq('B', 'e') - 1).asfreq('T', 's') + 16*60
print(p4pm)
print(p4pm.to_timestamp())
rng = pd.period_range('2011Q3', '2012Q4', freq = 'Q-JAN')
ts = pd.Series(np.arange(len(rng)), index = rng)
print(ts)
new_rng = (rng.asfreq('B', 'e') - 1).asfreq('T', 's') + 16*60
ts.index = new_rng.to_timestamp()
print(ts)

rng = pd.date_range('2000-01-01', periods =3, freq = 'M')
ts = pd.Series(np.random.randn(3), index = rng)
print(ts)
pts = ts.to_period()
print(pts)
rng = pd.date_range('1/29/2000', periods = 6, freq = 'D')
ts2 = pd.Series(np.random.randn(6), index = rng)
print(ts2)
print(ts2.to_period('M'))
pts = ts2.to_period()
print(pts.to_timestamp(how = 'end'))

url = 'https://raw.githubusercontent.com/wesm/pydata-book/3rd-edition/examples/'
urlmacro = url + 'macrodata.csv'
#urlmacro = 'https://raw.githubusercontent.com/wesm/pydata-book/3rd-edition/examples/macrodata.csv'
data = pd.read_csv(urlmacro)
print(data.head(5))
print(data.year)
print(data.quarter)
index = pd.PeriodIndex(year = data.year,quarter = data.quarter, freq = 'Q-DEC')
print(index)
data.index = index
print(data.infl)

rnd = pd.date_range('2000-01-01', periods = 100, freq = 'D')
ts = pd.Series(np.random.randn(len(rng)), index = rng)
print(ts)
print(ts.resample('M').mean())
print(ts.resample('M', kind = 'period').mean())

rng = pd.date_range('2000-01-01', periods = 12, freq = 'T')
ts = pd.Series(np.arange(12), index = rng)
print(ts)
print(ts.resample('5min', closed = 'right').sum())
print(ts.resample('5min', closed = 'right').sum())
print(ts.resample('5min', closed = 'right', label = 'right').sum())
print(ts.resample('5min', closed = 'right', label = 'right', loffset = '-1s').sum())

print(ts.resample('5min').ohlc())
frame = pd.DataFrame(np.random.randn(2,4),
                        index = pd.date_range('1/1/2000', periods = 2,
                                freq = 'W-WED'),
                        columns = ['Colorado', 'Texas', 'New York', 'Ohio'])
print(frame)
df_daily = frame.resample('D').asfreq()
print(df_daily)
print(frame.resample('D').ffill())
print(frame.resample('D').ffill(limit = 2))
print(frame.resample('W-THU').ffill())

frame = pd.DataFrame(np.random.randn(24,4),
                        index = pd.period_range('1-2000', '12-2001',
                                                freq = 'M'),
                        columns = ['Colorado', 'Texas', 'New York', 'Ohio'])
print(frame[:5])
annual_frame = frame.resample('A-DEC').mean()
print(annual_frame)
print(annual_frame.resample('Q-DEC').ffill())
print(annual_frame.resample('Q-DEC', convention = 'end').ffill())
print(annual_frame.resample('Q-MAR').ffill())

urlstock_px = url + 'stock_px.csv'
close_px_all = pd.read_csv(urlstock_px, 
                                parse_dates = True, index_col = 0)
close_px = close_px_all[['AAPL','MSFT','XOM']]
close_px = close_px.resample('B').ffill()
close_px.AAPL.plot()
close_px.AAPL.rolling(250).mean().plot()
plt.pause(3)
plt.close()
appl_std250 = close_px.AAPL.rolling(250, min_periods = 10).std()
print(appl_std250[5:12])
appl_std250.plot()
expanding_mean = appl_std250.expanding().mean()
plt.pause(3)
plt.close()
close_px.rolling(60).mean().plot(logy = True)
plt.pause(3)
plt.close()
print(close_px.rolling('20D').mean())
aapl_px = close_px.AAPL['2006':'2007']
ma60 = aapl_px.rolling(30, min_periods = 20).mean()
ewma60 = aapl_px.ewm(span = 30).mean()
ma60.plot(style = 'k--', label = 'Simple MA')
ewma60.plot(style = 'k-',label = 'EW MA')
plt.legend()
plt.pause(3)
plt.close()
spx_px = close_px_all['SPX']
spx_rets = spx_px
returns = close_px.pct_change()
corr = returns.AAPL.rolling(125, min_periods = 100).corr(spx_rets)
corr.plot()
plt.pause(3)
plt.close()
corr = returns.rolling(125, min_periods = 100).corr(spx_rets)
corr.plot()
plt.pause(3)
plt.close()

score_at_2percent = lambda x: percentileofscore(x,0.02)
result = returns.AAPL.rolling(250).apply(score_at_2percent)
result.plot()
plt.pause(3)
plt.close()
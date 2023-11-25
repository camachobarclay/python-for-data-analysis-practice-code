import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import random
from statsmodels.tsa.ar_model import AutoReg
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score

data = pd.DataFrame({'x0': [1,2,3,4,5],
                     'x1':[0.01, -0.01, 0.25, -4.1, 0.],
                      'y': [-1.5, 0., 3.6, 1.3, -2.]})
print(data)
print(data.columns)
print(data.values)
df2 = pd.DataFrame(data.values, columns = ['one', 'two', 'three'])
print(df2)
df3 = data.copy()
df3['string'] = ['a', 'b', 'c', 'd','e']
print(df3)
print(df3.values)
models_cols = ['x0', 'x1']
print(data.loc[:, models_cols].values)
print(data.iloc[[1,3],:].values)
data['category'] = pd.Categorical(['a','b','a','a','b'], categories = ['a', 'b'])
print(data)
dummies = pd.get_dummies(data.category, prefix = 'category')
data_with_dummies = data.drop('category', axis = 1).join(dummies)
print(data_with_dummies)

data = pd.DataFrame({
                    'x0': [1,2,3,4,5],
                    'x1': [0.01, -0.01, 0.25, -4.1, 0.],
                    'y': [-1.5, 0., 3.6, 1.3, -2.]})
print(data)
y,X = patsy.dmatrices('y~x0 + x1', data)
print(y)
print(X)
print(np.asarray(y))
print(np.asarray(X))
print(patsy.dmatrices('y~x0 + x1 + 0', data)[1])
coef, resid, _, _ = np.linalg.lstsq(X,y)
print(coef)
coef = pd.Series(coef.squeeze(), index = X.design_info.column_names)
print(coef)
y,X = patsy.dmatrices('y~x0 + np.log(np.abs(x1) + 1)', data)
print(X)
y,X = patsy.dmatrices('y ~ standardize(x0) + center(x1)', data)
print(X)
new_data = pd.DataFrame({
    'x0': [6,7,8,9],
    'x1': [3.1, -0.5, 0, 2.3],
    'y': [1,2,3,4]})
new_X = patsy.build_design_matrices([X.design_info],new_data)
print(new_X)
y,X = patsy.dmatrices('y~I(x0+x1)', data)
print(X)
data = pd.DataFrame({'key1':['a','a','b','b','a','b','a','b'],
                     'key2':[0,1,0,1,0,1,0,0],
                     'v1': [1,2,3,4,5,6,7,8],
                     'v2': [-1,0,2.5,-0.5,4.0,-1.2,0.2,-1.7]
                    })
y,X = patsy.dmatrices('v2 ~ key1', data)
print(X)
y,X = patsy.dmatrices('v2 ~ key1 + 0', data)
print(X)
y,X = patsy.dmatrices('v2 ~ C(key2)', data)
print(X)
data['key2'] = data['key2'].map({0:'zero', 1: 'one'})
print(data)
y,X = patsy.dmatrices('v2 ~ key1 + key2', data)
print(X)
y,X = patsy.dmatrices('v2 ~ key1 + key2 + key1:key2', data)
print(X)

def dnorm(mean, variance, size = 1):
    if isinstance(size, int):
        size = size,
        return mean + np.sqrt(variance)*np.random.randn(*size)
# For reproducability
np.random.seed(12345)
N = 100
X = np.c_[dnorm(0,0.4, size = N),
        dnorm(0,0.6, size = N),
        dnorm(0,0.2, size = N)]
eps = dnorm(0,0.1, size = N)
beta = [0.1,0.3,0.5]
y = np.dot(X,beta) + eps
print(X[:5])
print(y[:5])
X_model = sm.add_constant(X)
print(X_model[:5])
model = sm.OLS(y,X)
results = model.fit()
print(results.params)
print(results.summary())
data = pd.DataFrame(X,columns = ['col0', 'col1', 'col2'])
data['y'] = y
print(data[:5])
results = smf.ols('y ~ col0 + col1 + col2', data = data).fit()
print(results.params)
print(results.tvalues)
print(results.predict(data[:5]))

init_x = 4
values = [init_x, init_x]
N = 1000
b0 = 0.8
b1 = -0.4
noise = dnorm(0,0.1,N)
for i in range(N):
    new_x = values[-1]*b0 + values[-2]*b1 + noise[i]
    values.append(new_x)
MAXLAGS = 5
model = AutoReg(values, MAXLAGS)
results = model.fit()
print(results.params)

url = 'https://raw.githubusercontent.com/wesm/pydata-book/3rd-edition/'
urltrain = url + 'datasets/titanic/train.csv'
urltest = url + 'datasets/titanic/test.csv'
train = pd.read_csv(urltrain)
test = pd.read_csv(urltest)
print(train.head(4))
print(train.isna().sum())
print(test.isna().sum())
impute_value = train['Age'].median()
train['Age'] = train['Age'].fillna(impute_value)
test['Age'] = test['Age'].fillna(impute_value)
train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)
predictors = ['Pclass', 'IsFemale', 'Age']
X_train = train[predictors].to_numpy()
X_test = test[predictors].to_numpy()
y_train = train['Survived'].to_numpy()
print(X_train[:5])
print(y_train[:5])
model = LogisticRegression()
print(model.fit(X_train, y_train))
y_predict = model.predict(X_test)
print(y_predict[:10])

model_cv = LogisticRegressionCV(Cs=10)
model_cv.fit(X_train, y_train)
model = LogisticRegression(C=10)
scores = cross_val_score(model, X_train, y_train, cv=4)
scores


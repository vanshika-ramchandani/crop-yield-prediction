# importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading the dataset
df = pd.read_csv('yield_df.csv')
df.head()
df.drop('Unnamed: 0', axis=1, inplace=True)

df.head()

df.shape

df.isnull().sum()

df.info()

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.duplicated().sum()

df.dtypes

df.describe()

df['average_rain_fall_mm_per_year'].value_counts()

plt.figure(figsize=(10,20))  # 10 = Height and 20 = Width
sns.countplot(y=df['Area'])

country = df['Area'].unique()

yeild_per_country = []
for state in country:
    yeild_per_country.append(df[df['Area'] == state]['hg/ha_yield'].sum())

df['hg/ha_yield'].sum()

plt.figure(figsize=(10,20))
sns.barplot(x=yeild_per_country, y=country)

sns.countplot(y = df['Item'])

crops = df['Item'].unique()
crops

crop_yield_per_item = []
for crop in crops:
    crop_yield_per_item.append(df[df['Item'] == crop]['hg/ha_yield'].sum())
crop_yield_per_item

sns.barplot(y= crops, x=crop_yield_per_item)

col = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[col]
df.head()

x=df.drop('hg/ha_yield', axis=1)
y=df['hg/ha_yield']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train.shape, x_test.shape

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = OneHotEncoder(drop ='first')
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('oneHotEncoder' , ohe, [4,5]),
        ('standardization' , scaler, [0,1,2,3])
    ],
remainder = 'passthrough'
)

preprocessor

x_train_dummy = preprocessor.fit_transform(x_train)
x_test_dummy = preprocessor.transform(x_test)

x_train_dummy

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

models = {
    'lr': LinearRegression(),
    'lss': Lasso(),
    'rg': Ridge(),
    'knr': KNeighborsRegressor(),
    'dtr': DecisionTreeRegressor()
}

for name, model in models.items():
    model.fit(x_train_dummy, y_train)
    y_pred = model.predict(x_test_dummy)
    print(f'{name} : {mean_absolute_error(y_test, y_pred)} Score {model.score(x_test_dummy, y_test)*100}')

dtr = DecisionTreeRegressor()   ## We Will be going to pickle this File
dtr.fit(x_train_dummy, y_train)
dtr.predict(x_test_dummy)  # Yeilds the predicted values per country

def prediction(Year, average_rain_fall_mm_per_year,	pesticides_tonnes, avg_temp, Area,	Item):
    features = np.array([[Year,	average_rain_fall_mm_per_year, pesticides_tonnes,	avg_temp,	Area,	Item]])

    # Now we have to on-hot encode and standarize this input also

    transformed_features = preprocessor.transform(features)
    predicted_value = dtr.predict(transformed_features).reshape(1,-1)
    return predicted_value[0]

x_test.head()
prediction(1996, 1513.0, 152.01, 19.71,	'Madagascar',	'Wheat')

import pickle
pickle.dump(dtr, open('CropYeild.pkl', 'wb'))
pickle.dump(preprocessor, open('Preprocessor.pkl', 'wb'))
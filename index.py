import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # все кроме последней колонки
y = dataset.iloc[:, -1].values  # только последняя колонка

# take care of missing data
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:3])# исключаем текстовую колонку
X[:,1:3]=imputer.transform(X[:,1:3])# поместить в вабраную область, отредактированный масиив


# зашивруем категории в виде вектора
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# 1 элемент - вид трансвормации, 2 - класс странсформера, 3 - колонка для кодирования
# remainder - инструкция для трансформации, говорит расширить
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X=ct.fit_transform(X)

# закодируем наши игрики
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

# разодьем данные на тестовые и проверочные
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=1)#random_state=1 убирает рандом что он всегжа одинаков

# feature scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler() #сколько среднеквадратичных отклонений содержит наша величина
X_train[:,3:]=ss.fit_transform(X_train[:,3:])#применяем к тестовой выборке
# когда мы вызываем fit_transform мы (1) готовим модель кторая конвертирует, а потом на основе ее изменяем наши данные
X_test[:,3:]=ss.transform(X_test[:,3:]) # тут только transform потому что мы ТОЛЬКО ЧТО создали модель странсформации, и среднее и отклонение УЖЕ расчитаны, поэтому только меняем
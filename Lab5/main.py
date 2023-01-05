import pandas
import matplotlib.pyplot as pyplot
import pylab
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#Reading data from csv
teach_df = pandas.read_csv('country_wise_latest.csv')
teach_df.replace([np.inf, -np.inf], np.nan, inplace=True)
teach_df.dropna(inplace=True)
teach_df['Country/Region'] = pandas.factorize(teach_df['Country/Region'])[0]
teach_df['WHO Region'] = pandas.factorize(teach_df['WHO Region'])[0]
x_teach_df , x_test_df , y_teach_df , y_test_df = train_test_split(teach_df.drop('WHO Region', axis=1), teach_df['WHO Region'], test_size=0.5, random_state=0)
x_teach_df = pandas.DataFrame(x_teach_df, index=x_teach_df.index, columns=x_teach_df.columns)
x_test_df = pandas.DataFrame(x_test_df, index=x_test_df.index, columns=x_test_df.columns)

#Scaling the data
scaler = StandardScaler()
X_teach = scaler.fit_transform(x_teach_df)
X_test = scaler.fit_transform(x_test_df)

#We run the test data through classifiers and output the accuracy of predictions

#k nearest neighbors
knn = KNeighborsClassifier(n_neighbors=4).fit(X_teach, y_teach_df)
knn_predictions = pandas.Series(knn.predict(X_test))
print('Точность предсказаний k ближайших соседей: ' + str(knn.score(X_test, y_test_df)*100) + '%')

#Logistic regression method
lr = LogisticRegression().fit(X_teach, y_teach_df)
lr_predictions = pandas.Series(lr.predict(X_test))
print('Точность предсказаний логистической регрессии: ' + str(lr.score(X_test, y_test_df)*100) + '%')

#The method of support vectors
svm = SVC(kernel = 'rbf').fit(X_teach, y_teach_df)
svm_predictions = pandas.Series(svm.predict(X_test))
print('Точность предсказаний методом опорных векторов: ' + str(svm.score(X_test, y_test_df)*100) + '%')

#We draw graphs with predicted and real values of regions

#k nearest neighbors
pylab.figure(figsize=(20,10))
pylab.subplot(1, 2, 1)
pyplot.pie(y_test_df.value_counts().sort_index(), labels = sorted(y_test_df.unique()), autopct='%1.1f%%')
pyplot.title('Реальные регионы')
pylab.subplot(1, 2, 2)
pyplot.pie(knn_predictions.value_counts().sort_index(), labels = sorted(knn_predictions.unique()), autopct='%1.1f%%')
pyplot.title('Регионы предсказанные k ближайших соседей')
pyplot.show()

#Logistic regression method
pylab.figure(figsize=(20,10))
pylab.subplot(1, 2, 1)
pyplot.pie(y_test_df.value_counts().sort_index(), labels = sorted(y_test_df.unique()), autopct='%1.1f%%')
pyplot.title('Реальные регионы')
pylab.subplot(1, 2, 2)
pyplot.pie(lr_predictions.value_counts().sort_index(), labels = sorted(lr_predictions.unique()), autopct='%1.1f%%')
pyplot.title('Регионы предсказанные логистической регрессией')
pyplot.show()

#The method of support vectors
pylab.figure(figsize=(20,10))
pylab.subplot(1, 2, 1)
pyplot.pie(y_test_df.value_counts().sort_index(), labels = sorted(y_test_df.unique()), autopct='%1.1f%%')
pyplot.title('Реальные регионы')
pylab.subplot(1, 2, 2)
pyplot.pie(svm_predictions.value_counts().sort_index(), labels = sorted(svm_predictions.unique()), autopct='%1.1f%%')
pyplot.title('Регионы предсказанные методом опорных векторов')
pyplot.show()
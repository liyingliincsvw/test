# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np


data = pd.read_csv('CarPrice_Assignment.csv', encoding='gbk')
#print(df1)
d = pd.read_excel('Data_Dictionary_carprices.xlsx')
df = pd.DataFrame(data)
aa = []
aa = list(df.columns)
print(aa)
train_x = data[aa]

# LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoder_list = ['CarName','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem']
for item in encoder_list:
    train_x[item] = le.fit_transform(train_x[item])


# 规范化到 [0,1] 空间
min_max_scaler=preprocessing.MinMaxScaler()
train_x=min_max_scaler.fit_transform(train_x)
pd.DataFrame(train_x).to_csv('temp.csv', index=False)
#print(train_x)


### 使用KMeans聚类
kmeans = KMeans(n_clusters=7)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
# 合并聚类结果，插入到原数据中
result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:u'Kmeans result'},axis=1,inplace=True)
print(result)
# 将结果导出到CSV文件中
result.to_csv("CarPrice_Assignment_result.csv",index=False)

"""
# K-Means 手肘法：统计不同K取值的误差平方和
import matplotlib.pyplot as plt
sse = []
for k in range(1, 20):
    # kmeans算法
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(train_x)
    #计算inertia簇内误差平方和
    sse.append(kmeans.inertia_)
x = range(1, 20)
plt.xlabel('K')
plt.ylabel('SSE')
plt.plot(x, sse, 'o-')
plt.show()

"""
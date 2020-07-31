# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from efficient_apriori import apriori
import csv

data = pd.read_csv("order.csv",encoding = "gbk")
df = pd.DataFrame(data)
print(df.shape)
df.duplicated()
print(df.shape)
order_date = df['订单日期'].unique()
transactions = []
print(len(order_date))
time = 0

transaction = []

for date in order_date:
    df1 = df.iloc[list(df['订单日期'] == date)]
    df1.insert(0, 'index2', range(len(df1)))
    df1.set_index('index2')
    customer_ID = df1['客户ID'].unique()
    for customer in customer_ID:
        time += 1
        transaction = []
        for i in range(len(df1)):
            if df1.loc[df1.index2 == i]['客户ID'].item() == int(customer):
                transaction.append(df1.loc[df1.index2 == i]['产品名称'].item())
        print(time)
        transactions.append(tuple(transaction))




itemsets, rules = apriori(transactions1, min_support=0.05,  min_confidence=0.01)
print("频繁项集：", itemsets)
print("关联规则：", rules)
result = '频繁项集：{}，\n 关联规则：{}'.format(itemsets, rules)


import os


with open('apriori_result.txt','a') as file0:
    print(result,file=file0)


print(result)




import scipy

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import xlsxwriter as xlsx
import plotly.express as px
import re

import statsmodels.api as sm
import statsmodels.formula.api as smf
from fontTools.misc.fixedTools import floatToFixedToFloat
from statsmodels.stats.api import anova_lm
from statsmodels.stats.multicomp import (MultiComparison,pairwise_tukeyhsd)

import pingouin as pg  #для дисперсионного анализа

from scipy.special import comb


from scipy import stats

from pandas import read_csv
from scipy.stats import zscore
from scipy.stats import ttest_ind #!!!!!!!!!!!!
from numpy import zeros
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#Читаем файл
data0=read_csv('data.csv',sep=';')
data0['Cost'] = data0['Cost'].str.replace(',', '.')
data0['Cost'] = data0['Cost'].astype(float)
data0=data0.query("Shows!=0 or Conversions!=0 or Cost!=0 or Clicks!=0") #Убираем строчки с полностью нулевыми значениями ввиду их неэфективности
data0=data0.sort_values('Date')
data0['Client']=data0['Client'].replace({'HYACINTH':'ГИАЦИНТ','BUTTERCUP ':'ЛЮТИК','THISTLE ':'ЧЕРТОПОЛОХ','HYACINTH ':'ГИАЦИНТ','GLADIOLUS':'ГЛАДИОЛУС','BELL':'КОЛОКОЛЬЧИК','TULIP':'ТЮЛЬПАН','CORNFLOWER':'ВАСИЛЕК','ANEMONE':'АНЕМОН'})
data0['Category']=data0['Category'].replace({'CASH SETTLEMENT SERVICES':'РАСЧЕТНО_КАССОВОЕ_ОБСЛУЖИВАНИЕ','CREDIT CARDS':'КРЕДИТНЫЕ_КАРТЫ','INVESTMENTS':'ИНВЕСТИЦИИ','Deposits':'ВКЛАДЫ','Debit cards':'ДЕБЕТОВЫЕ_КАРТЫ','CONSUMER LENDING':'ПОТРЕБИТЕЛЬСКОЕ_КРЕДИТОВАНИЕ'})
data=data0.query("Client=='КОЛОКОЛЬЧИК'")
nums=['Clicks','Cost','Conversions','Shows']
letters=['BannerType','CampaignType','Category','Device','Place','QueryType','TargetingType']


#    (0)     СТРОИМ ГРАФИКИ ЗАВИСИМОСТИ КАЖДОГО СТОЛБЦА ТАБЛИЦЫ ДЛЯ КАЖДОГО КЛИЕНТА (ОБЩИЕ 11 ГРАФИКОВ)
def graf(data0,nums,letters):
    for client in data0.Client.unique():
        client_data = data0.query('Client == @client')
        plt.figure(figsize=(20, 15))
        plt.suptitle(f'Анализ для клиента: {client}', fontsize=16)
        plot_idx = 1
        for idx, col in enumerate(nums + letters):
            plt.subplot(3, 4, plot_idx)
            if col in nums:
                metric_sum = client_data.groupby("Date")[col].sum().reset_index()
                sns.barplot(data=metric_sum, x='Date', y=col)
                plt.title(f"Сумма {col} по месяцам")
            elif col in letters:
                sns.histplot(data=client_data, x='Date', hue=col, multiple='dodge')
                plt.title(f"Распределение {col}")
            plt.xlabel('Месяц', fontsize=12)
            plt.ylabel(col if col in nums else 'Количество', fontsize=12)
            plt.xticks(rotation=45)
            plot_idx += 1
        plt.tight_layout()
        plt.show()
# graf(data0,nums,letters)


#       Общая функция которая считает Total, (CPA,CPC,CR,CTR)
def full(column, data):
    results = []
    for i in column.unique():
            grouped=data.query(f"{column.name}==@i").groupby("Date").agg(Total_Cost=("Cost", "sum"),Total_Conversions=("Conversions", "sum"),Total_Clicks=("Clicks", "sum"),Total_Shows=("Shows", "sum"))
            grouped[column.name] = i
            grouped.set_index(column.name, append=True, inplace=True)
            grouped["CPA"] =(grouped["Total_Cost"] / grouped["Total_Conversions"]).replace(0, np.nan)
            grouped["CPC"] = (grouped["Total_Cost"] / grouped["Total_Clicks"]).replace(0, np.nan)
            grouped["CTR"] =( grouped["Total_Clicks"] / grouped["Total_Shows"]).replace(0, np.nan)
            grouped["CR"] = (grouped["Total_Conversions"] / grouped["Total_Clicks"]).replace(0, np.nan)
            results.append(grouped)
    return pd.concat(results).reset_index()
for i in data0.Client.unique():
    res=full(data0['Category'],data0.query("Client==@i and TargetingType=='Phrase' and Category=='ДЕБЕТОВЫЕ_КАРТЫ' and BannerType=='text' and QueryType=='Brand' "))
    print(f'{i} Phrase and text and Competitor')
    print(res[['Date','CPA','CPC','CTR','CR','Total_Cost']])




#     (1)   Для каждого значения из каждого столбца график зависимости CPA... от Date hue=CLIENT
def graf_compare_hue_client(data, column_name):
    unique_values = data[column_name].unique()
    for value in unique_values:
        plt.figure(figsize=(20, 15))
        plt.suptitle(f'Сравнение метрик для {column_name}={value} по клиентам', fontsize=16)
        filtered_data = data[data[column_name] == value]
        grouped = filtered_data.groupby(["Date", "Client"]).agg(Total_Cost=("Cost", "sum"),Total_Conversions=("Conversions", "sum"),Total_Clicks=("Clicks", "sum"),Total_Shows=("Shows", "sum")).reset_index()
        grouped["CPA"] = (grouped["Total_Cost"] / grouped["Total_Conversions"]).replace(0, np.nan)
        grouped["CPC"] = (grouped["Total_Cost"] / grouped["Total_Clicks"]).replace(0, np.nan)
        grouped["CTR"] =( grouped["Total_Clicks"] / grouped["Total_Shows"]).replace(0, np.nan)
        grouped["CR"] = (grouped["Total_Conversions"] / grouped["Total_Clicks"]).replace(0, np.nan)
        metrics = ['CPA', 'CPC', 'CTR', 'CR']
        for idx, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, idx)
            sns.lineplot(
                data=grouped,
                x='Date',
                y=metric,
                hue='Client',
                marker='o'
            )
            plt.title(f'{metric} по датам')
            plt.xticks(rotation=45)
            plt.tight_layout()
        plt.show()
graf_compare_hue_client(data0.query("Client==['КОЛОКОЛЬЧИК', 'ВАСИЛЕК', 'ЛЮТИК', 'ГЛАДИОЛУС', 'ТЮЛЬПАН', 'ГИАЦИНТ','ЧЕРТОПОЛОХ']"), 'Category')



#  (2)  График зависимости для каждого значения раскрашенный в зависимости от CPA, CPC,CR,CTR
# def graf_compare_hue_metric(data, client, column_name):
#     data = data.query("Client == @client")
#     unique_values = data[column_name].unique()
#     num_values = len(unique_values)
#     rows = (num_values + 2) // 3  # Максимум 3 столбца
#     fig, axes = plt.subplots(rows, 3, figsize=(20, 5 * rows))
#     fig.suptitle(f'Сравнение метрик для клиента {client}', fontsize=16)
#     if rows == 1:
#         axes = axes.reshape(1, -1)
#     for idx, value in enumerate(unique_values):
#         row = idx // 3
#         col = idx % 3
#         filtered_data = data[data[column_name] == value]
#         grouped = filtered_data.groupby("Date").agg(
#             Total_Cost=("Cost", "sum"),
#             Total_Conversions=("Conversions", "sum"),
#             Total_Clicks=("Clicks", "sum"),
#             Total_Shows=("Shows", "sum")
#         ).reset_index()
#         for metric in ['CPA', 'CPC', 'CTR', 'CR']:
#             grouped[metric] = pd.NA
#         grouped["CPA"] = (grouped["Total_Cost"] / grouped["Total_Conversions"]).replace(0, np.nan)
#         grouped["CPC"] = (grouped["Total_Cost"] / grouped["Total_Clicks"]).replace(0, np.nan)
#         grouped["CTR"] = (grouped["Total_Clicks"] / grouped["Total_Shows"]).replace(0, np.nan)
#         grouped["CR"] = (grouped["Total_Conversions"] / grouped["Total_Clicks"]).replace(0, np.nan)
#         melted = grouped.melt(
#             id_vars=["Date"],
#             value_vars=["CPA", "CPC", "CTR", "CR"],
#             var_name="Metric",
#             value_name="Value"
#         )
#         ax = axes[row, col] if rows > 1 else axes[col]
#         sns.pointplot(
#             data=melted,
#             x="Date",
#             y="Value",
#             hue="Metric",
#             ax=ax,
#             capsize=.2
#         )
#         ax.set_title(f"{column_name} = {value}")
#         ax.tick_params(axis='x', rotation=45)
#         ax.set_ylabel("")
#     for idx in range(len(unique_values), rows * 3):
#         row = idx // 3
#         col = idx % 3
#         if rows > 1:
#             axes[row, col].axis('off')
#         else:
#             axes[col].axis('off')
#     plt.tight_layout()
#     plt.show()

#  (2)    График зависимости в логарифмированно масштабе для каждого значения раскрашенный в зависимости от CPA, CPC,CR,CTR
def graf_compare_hue_metric_log(data, client, column_name):
    data_filtered = data[data['Client'] == client].copy()
    unique_values = data_filtered[column_name].unique()
    num_values = len(unique_values)
    cols = 3
    rows = (num_values + cols - 1) // cols  # Округление вверх
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), squeeze=False)
    fig.suptitle(f'Метрики (логарифм) для клиента {client}', fontsize=16)

    for idx, value in enumerate(unique_values):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        filtered_data = data_filtered[data_filtered[column_name] == value]
        if filtered_data.empty:
            ax.axis('off')
            continue
        grouped = filtered_data.groupby('Date').agg(
            Total_Cost=('Cost', 'sum'),
            Total_Conversions=('Conversions', 'sum'),
            Total_Clicks=('Clicks', 'sum'),
            Total_Shows=('Shows', 'sum')
        )
        grouped['CPA'] = grouped['Total_Cost'] / grouped['Total_Conversions'].replace(0, np.nan)
        grouped['CPC'] = grouped['Total_Cost'] / grouped['Total_Clicks'].replace(0, np.nan)
        grouped['CTR'] = grouped['Total_Clicks'] / grouped['Total_Shows'].replace(0, np.nan)
        grouped['CR'] = grouped['Total_Conversions'] / grouped['Total_Clicks'].replace(0, np.nan)
        for metric in ['CPA', 'CPC', 'CTR', 'CR']:
            grouped[metric] = np.log(grouped[metric].where(grouped[metric] > 0, np.nan))

        melted = grouped.reset_index().melt(
            id_vars=['Date'],
            value_vars=['CPA', 'CPC', 'CTR', 'CR'],
            var_name='Metric',
            value_name='Value'
        )
        sns.pointplot(
            data=melted,
            x='Date',
            y='Value',
            hue='Metric',
            ax=ax,
            capsize=.2
        )
        ax.set_title(f"{column_name} = {value}")
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel('log(Значение)')
    # Скрытие пустых подграфиков
    total_plots = rows * cols
    for idx in range(len(unique_values), total_plots):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()
for column in letters:
        graf_compare_hue_metric_log(data0,'КОЛОКОЛЬЧИК',column)

# (3)  График зависимости метрик для каждого столбца раскрашенный в зависимости от значения в столбце
def graf_compare_hue_value_in_column(data, client, column_name):
    data_filtered = data.query("Client == @client")
    unique_values = data_filtered[column_name].unique()
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle(f'Метрики для клиента {client} ({column_name})', fontsize=16)
    metrics = ['CPA', 'CPC', 'CTR', 'CR']
    grouped = data_filtered.groupby(["Date", column_name]).agg(
        Total_Cost=("Cost", "sum"),
        Total_Conversions=("Conversions", "sum"),
        Total_Clicks=("Clicks", "sum"),
        Total_Shows=("Shows", "sum")
    ).reset_index()
    grouped["CPA"] = grouped["Total_Cost"] / grouped["Total_Conversions"].replace(0, np.nan)
    grouped["CPC"] = grouped["Total_Cost"] / grouped["Total_Clicks"].replace(0, np.nan)
    grouped["CTR"] = grouped["Total_Clicks"] / grouped["Total_Shows"].replace(0, np.nan)
    grouped["CR"] = grouped["Total_Conversions"] / grouped["Total_Clicks"].replace(0, np.nan)
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        sns.lineplot(
            data=grouped,
            x='Date',
            y=metric,
            hue=column_name,  # Используем имя столбца для hue
            marker='o',
            ax=ax
        )
        ax.set_title(metric)
        ax.set_xlabel('Дата')
        ax.set_ylabel('Значение')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
    plt.tight_layout()
    plt.show()
graf_compare_hue_value_in_column(data0,'КОЛОКОЛЬЧИК','Category')

#print(data0.query("Client == 'КОЛОКОЛЬЧИК' and Category=='ДЕБЕТОВЫЕ_КАРТЫ' ").groupby(['Date','BannerType','Place','TargetingType','QueryType']).Category.value_counts())
# print('КРЕДИТНЫЕ_КАРТЫ')
# print(res.query("Date=='01.07.2023'")[['Date','BannerType','Place','TargetingType','QueryType']].set_index('Date'))

#Построения видов расчетов CPA
#Данные не содержащие 0
data_without0=data.query("Shows!=0 and Conversions!=0 and Cost!=0 and Clicks!=0")
data_without0['CPA'] = data_without0['Cost'] / data_without0['Conversions']
CPA_without0_mean = data_without0.groupby('Date').CPA.mean().rename('CPA_without0_mean_value')
CPA_without0_mean=CPA_without0_mean.reset_index()
CPA_without0_mean['calculation_method']='CPA_without0_mean'
CPA_without0_median = data_without0.groupby('Date').CPA.median()

#Данные содержащие 0 и посчитанные в общем за месяц
CPA_with0=(data.groupby("Date").Cost.sum())/(data.groupby("Date").Conversions.sum())
CPA_with0=CPA_with0.reset_index()
CPA_with0['CPA_with0_value']=CPA_with0.iloc[:,1]
CPA_with0 = CPA_with0.drop(columns =0)
CPA_with0['calculation_method']='CPA_with0'
CPA_data = pd.concat([CPA_with0, CPA_without0_mean], axis=0)
CPA_data=CPA_data.fillna(0)
CPA_data['value']=CPA_data['CPA_with0_value']+CPA_data['CPA_without0_mean_value']

#Даныее с замененым значением 0 на 1
data_0_1=data.copy(deep=True)
data_0_1['Conversions'] = pd.to_numeric(data_0_1['Conversions'], errors='coerce').replace(0, 1)
data_0_1['CPA'] = data_0_1['Cost'] / data_0_1['Conversions']
CPA_0_1_mean = data_0_1.groupby('Date').CPA.mean()
CPA_0_1_median = data_0_1.groupby('Date').CPA.median()

#Прибавим ко всему Conversions+1
data_plus1=data.copy(deep=True)
data_plus1['Conversions']=data_plus1['Conversions']+1
data_plus1['CPA'] = data_plus1['Cost'] / data_0_1['Conversions']
data_plus1_mean = data_plus1.groupby('Date').CPA.mean().rename('CPA_plus1_mean_value')
data_plus1_mean=data_plus1_mean.reset_index()
data_plus1_mean['calculation_method']='data_plus1_mean'
data_plus1_median = data_plus1.groupby('Date').CPA.median().rename('CPA_plus1_median_value')
data_plus1_median=data_plus1_median.reset_index()
data_plus1_median['calculation_method']='data_plus1_median'

# #pointplot с методами счета CPA
СPA_data_0 = pd.concat([CPA_with0, CPA_without0_mean], axis=0)
СPA_data_1=pd.concat([СPA_data_0, data_plus1_mean], axis=0)
СPA_data_res=pd.concat([СPA_data_1, data_plus1_median], axis=0)
СPA_data_res=СPA_data_res.fillna(0)
СPA_data_res['value']=СPA_data_res['CPA_with0_value']+СPA_data_res['CPA_without0_mean_value']+СPA_data_res['CPA_plus1_mean_value']+СPA_data_res['CPA_plus1_median_value']
plt.figure(figsize=(15, 10))
sns.pointplot(x = 'Date', y = 'value', hue = 'calculation_method', data = СPA_data_res,capsize = .2)
plt.show()

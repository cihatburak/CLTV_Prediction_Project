##CLTV Prediction

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns',None)
##pd.set_option('display.max_rows',None)
pd.set_option('display.width',500)
pd.set_option('display.float_format',lambda x: "%.3f" %x)

#Examining data

flo_df = pd.read_csv("Datasets/flo_data_20k.csv")
flo_df.shape
flo_df.head()

def outlier_thresholds(df,variable):
    quartile1 = df[variable].quantile(0.01)
    quartile3 = df[variable].quantile(0.99)
    interquantile = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile)
    low_limit = round(quartile1 - 1.5 * interquantile)
    return up_limit,low_limit

def replace_with_thresholds(df,variable):
    up_limit,low_limit = outlier_thresholds(df,variable)
    df.loc[(df[variable] > up_limit),variable ] = up_limit
    ##df.loc[(df[variable] < low_limit),variable ] = low_limit


flo_df.describe().T
replace_with_thresholds(flo_df,"order_num_total_ever_online")
replace_with_thresholds(flo_df,"order_num_total_ever_offline")
replace_with_thresholds(flo_df,"customer_value_total_ever_offline")
replace_with_thresholds(flo_df,"customer_value_total_ever_online")


flo_df["TOTAL_PURCHASE_AMOUNT"] = flo_df["customer_value_total_ever_offline"] + flo_df["customer_value_total_ever_online"]
flo_df["TOTAL_PURCHASE"] = flo_df["order_num_total_ever_online"] + flo_df["order_num_total_ever_offline"]

flo_df.head()

flo_df["first_order_date"] = pd.to_datetime(flo_df['first_order_date'])
flo_df["last_order_date"] = pd.to_datetime(flo_df['last_order_date'])
flo_df["last_order_date_online"] = pd.to_datetime(flo_df['last_order_date_online'])
flo_df["last_order_date_offline"] = pd.to_datetime(flo_df['last_order_date_offline'])

for col in flo_df:
    print(str(col) + " " + str(flo_df[col].dtype))


flo_df["last_order_date"].max()  ## Timestamp('2021-05-30 00:00:00')

today = dt.datetime(2021,6,2)


##customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg
##Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure (weekly)

# recency: weekly
# T: Age. weekly.
# frequency: repeat purchase(>1)
# monetary: average earning by purchase

flo_df["T"] = (today - flo_df["first_order_date"]).dt.days / 7
flo_df["Recency"] = (flo_df["last_order_date"] - flo_df["first_order_date"]).dt.days / 7
flo_df["Frequency"] = flo_df["TOTAL_PURCHASE"]
flo_df["Monetary"] = flo_df["TOTAL_PURCHASE_AMOUNT"] / flo_df["TOTAL_PURCHASE"]

flo_df.head()

##CLTV
cltv = pd.DataFrame()

flo_df_info = flo_df[["master_id","Recency","T","Frequency","Monetary"]]

flo_df_info.columns = ["customer_id","recency_cltv_weekly","T_weekly","frequency","monetary_cltv_avg"]

flo_df_info.head()

flo_df_info.index = flo_df_info.customer_id


##BG/NBD, Gamma-Gamma

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(flo_df_info["frequency"],flo_df_info["recency_cltv_weekly"],flo_df_info["T_weekly"])

flo_df_info["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(12,flo_df_info['frequency'],
                                                        flo_df_info['recency_cltv_weekly'],
                                                        flo_df_info['T_weekly'])

flo_df_info["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(24,flo_df_info['frequency'],
                                                        flo_df_info['recency_cltv_weekly'],
                                                        flo_df_info['T_weekly'])


ggf = GammaGammaFitter (penalizer_coef=0.001)

ggf.fit(flo_df_info['frequency'],flo_df_info['monetary_cltv_avg'] )

ggf.conditional_expected_average_profit(flo_df_info['frequency'],flo_df_info['monetary_cltv_avg'])

flo_df_info["exp_average_value"] = ggf.conditional_expected_average_profit(flo_df_info['frequency'],flo_df_info['monetary_cltv_avg'])
flo_df_info.head()

cltv.head()

##CLTV 6 Months

cltv_6_months = ggf.customer_lifetime_value(bgf,
                                           flo_df_info['frequency'],
                                           flo_df_info['recency_cltv_weekly'],
                                           flo_df_info['T_weekly'],
                                           flo_df_info['monetary_cltv_avg'],
                                           time = 6,
                                           freq = "W",
                                           discount_rate=0.01
                                           )


cltv_6_months.columns = ["cltv"]
cltv_6_months.head()
cltv_6_months.reset_index()
flo_df_info = flo_df_info.reset_index(drop = True)

flo_df_info.head()


flo_df_final = flo_df_info.merge(cltv_6_months,on = "customer_id",how = "left")


flo_df_final["segment"] = pd.qcut(flo_df_final["clv"], 4, labels=["D", "C", "B", "A"])
flo_df_final
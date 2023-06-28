import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis

data = pd.read_csv('pMRP_actuals.csv')
data = data.set_index(pd.DatetimeIndex(data['CALENDAR_WEEK'])).sort_index()
data.head()

check = data.groupby('SKU_CODE')['CALENDAR_WEEK'].nunique()
check = pd.DataFrame(check)

orders = data.groupby('SKU_CODE')['ORDERED_INVENTORY'].sum()
orders = pd.DataFrame(orders)

cata = data.groupby('SKU_CODE')['PRODUCT_CATEGORY_NAME'].unique()
cata = pd.DataFrame(cata)
items = ['Accessory','Belts','Cufflinks & Studs','Jackets',
'Neckwear','Pants','Shirts','Shoes','Vests']
for i in items:
    dummy = []
    for j in cata['PRODUCT_CATEGORY_NAME']:
        if j == i:
            dummy.append(1)
        else:
            dummy.append(0)
    col = i
    cata[col] = dummy

skus = pd.merge(check,orders,right_index=True,left_index=True)
skus = pd.merge(skus,cata,right_index=True,left_index=True).drop('PRODUCT_CATEGORY_NAME',axis=1)
#wrangle Skus ---------------------------

x = skus

kmeans = KMeans(n_clusters=3, random_state=42).fit(check)
check['label'] =  kmeans.labels_




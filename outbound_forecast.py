import snowflake.connector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ctx = snowflake.connector.connect(
    user= 'rene',
    password= 'Forwardline1',
    account='theblacktux'
    )
cs = ctx.cursor()


def snowflake_fetch(conn, sql_query):
    """
    Function to query snowflake through created cs connection and return as dataframe
    Input:
        cs : Snowflake connection (output of snowflake_connect)
        sql_query: sql query as string to query
    Return:
        raw data set from query
    """
    conn.execute(sql_query)
    df = conn.fetch_pandas_all()
    return(df)

main_reservations_query = """select i.ORDER_ID,
                                   i.SHIPPING_REFERENCE_CODE,
                                   i.INBOUND_TRACKING_NUMBER,
                                   i.IS_REPLACEMENT,
                                   o.ORDER_TYPE_NAME,
                                   i.SHIPMENT_DATE as ob_ship_date,
                                   i.INBOUND_SHIP_DATE,
                                   i.SHIPPER_POSTAL_CODE,
                                   i.DELIVERY_DATE,
                                   i.IS_DELIVERED,
                                   i.PACKAGE_WEIGHT_LBS,
                                   s.EVENT_DATE,
                                   year(s.EVENT_DATE) as event_year,
                                   o.SHIPPING_STATE,
                                   o.IS_EVENT_OWNER,
                                   o.IS_GROOM,
                                   e.EVENT_TYPE_NAME,
                                   e.NUM_MEMBERS,
                                   datediff(day,i.EVENT_DATE,INBOUND_SHIP_DATE) as days_to_ship,
                                   datediff(day,INBOUND_SHIP_DATE,i.DELIVERY_DATE) as days_in_transit,
                                   datediff(day,i.EVENT_DATE,i.DELIVERY_DATE) as total_return_time
                            from DA_DIM.F_FREIGHT_INBOUND as i
                            left join DA_DIM.D_ORDER as o on o.ORDER_ID = i.ORDER_ID
                            left join DA_DIM.F_SHIPMENTS as s on s.SHIPPING_REFERENCE_CODE = i.SHIPPING_REFERENCE_CODE
                            left join DA_DIM.D_EVENT as e on e.EVENT_ID = i.EVENT_ID
                            where o.ORDER_TYPE_NAME = 'Rental'
                            and ob_ship_date < INBOUND_SHIP_DATE
                            and year(s.EVENT_DATE) between 2014 and 2022"""

raw_df = snowflake_fetch(cs, main_reservations_query)
raw_df['EVENT_DATE'] = pd.to_datetime(raw_df['EVENT_DATE']) 
raw_df['EVENT_MONTH'] = raw_df['EVENT_DATE'].dt.to_period('M').dt.to_timestamp()


plt.scatter(raw_df.DAYS_TO_SHIP,raw_df.DAYS_IN_TRANSIT)
plt.xlabel("Days to Ship")
plt.ylabel("Days In Transit")



avgs = raw_df.groupby(['IS_REPLACEMENT','EVENT_MONTH']).mean()
sub_avgs = avgs[['DAYS_TO_SHIP','DAYS_IN_TRANSIT']]
sub_avgs = sub_avgs.iloc[2:]
sub_avgs = sub_avgs.drop((False,'2022-08-01'),axis=0)

f, a = plt.subplots(2,1)
sub_avgs.xs(True).plot(kind='bar',ax=a[0])
sub_avgs.xs(False).plot(kind='bar',ax=a[1])
plt.xticks([])

replace = sub_avgs.loc[(slice(True),slice('2018-01-01','2022-05-01')),:] 
parent = sub_avgs.loc[(slice(False),slice('2018-01-01','2022-05-01')),:]


plt.plot(parent.index.get_level_values(1) , parent['DAYS_TO_SHIP'],color='green', label = "line 1")
plt.plot(replace.index.get_level_values(1), replace['DAYS_TO_SHIP'],color='blue', label = "line 2")
plt.legend()
plt.show()




import pandas as pd

data = pd.read_csv('reservations.csv')
data = data.set_index(pd.DatetimeIndex(data['REG_WEEK'])).sort_index()
data = data.drop(['REG_WEEK','CONVERTED_RESERVATIONS','CONVERSION_RATE','RES_RATE'],axis=1)
data = data.iloc[:len(data)-1,:]
data.head()

def forecast_reservations(data,periods):
    import pandas as pd
    import numpy as np
    import datetime 
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    
    for i in np.arange(1,periods+1,1):
        
        l=[]
        for i in np.arange(0,len(data.index),1):
            m = data.index[i].month
            l.append(m)
        data['month'] = l
        
        for i in np.arange(1,13,1):
            dummy = []
            for j in data['month']:
                if j == i:
                    dummy.append(1)
                else:
                    dummy.append(0)
            col = 'month_'+str(i)
            data[col] = dummy
        
        data = data.drop('month',axis=1)
        
        # append month attribute ----------------------------------------------
        cov = []
        for i in data.index:
            if i >= datetime.date(2020,3,19) and i<= datetime.date(2020,12,30):
                cov.append(1)
            else:
                cov.append(0)
            
        data['COVID'] = cov
        #append COVID attribute--------------------------------------------
        
        reg = data.drop(['AVG_PARTY_SIZE','TOTAL_PARTY','RESERVATION_CNT'],axis=1)
        res = data
        
        #Split dataset for reg and res prediction ------------------------
        
        reslags = []
        for j in range(16):
            time = []
            weeklag = []
            interval = j+1
            for i in res.index:
                try:
                    dayago = i - np.timedelta64(1*interval,'W')
                    week_orders = res.loc[dayago,'RESERVATION_CNT']
                    time.append(i)
                    weeklag.append(week_orders)
                except:
                    np.nan
            lagresdf = pd.DataFrame(weeklag,index=time,columns=['weekLag%s'%interval])
            reslags.append(lagresdf)
        
        reslagdata = res
        for i in reslags:
            reslagdata = reslagdata.merge(i,right_index=True,left_index=True,how='outer')
            
        # creat lag features for reservations---------------------------------
        
        reglags = []
        for j in range(16):
            time = []
            weeklag = []
            interval = j+1
            for i in data.index:
                try:
                    dayago = i - np.timedelta64(1*interval,'W')
                    week_orders = reg.loc[dayago,'CREATOR_REGISTRATIONS']
                    time.append(i)
                    weeklag.append(week_orders)
                except:
                    np.nan
            lagregdf = pd.DataFrame(weeklag,index=time,columns=['weekLag%s'%interval])
            reglags.append(lagregdf)
        
        reglagdata = reg
        for i in reglags:
            reglagdata = reglagdata.merge(i,right_index=True,left_index=True,how='outer')
        
        # create lags for registration --------------------------------------
        
        reglagdata = reslagdata.fillna(value=0)
        Y1 = reglagdata['CREATOR_REGISTRATIONS'].to_numpy().reshape(len(reglagdata),1)
        X1 = reglagdata.drop(['CREATOR_REGISTRATIONS'],axis = 1)
        X_train1,X_test1,Y_train1,Y_test1 = train_test_split(X1, Y1,
                                                             test_size = 0.99,
                                                             random_state=42)
        linear = LinearRegression(fit_intercept=False)
        linear.fit(X_train1,Y_train1)
     
        #Run linear regression for Registrion predition ---------------------
        
        reslagdata = reslagdata.fillna(value=0)
        Y = reslagdata['RESERVATION_CNT'].to_numpy().reshape(len(reslagdata),1)
        X = reslagdata.drop(['RESERVATION_CNT'],axis = 1)
        X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.99,random_state=42)
        Y_train = Y_train.reshape(len(Y_train))
        
        rfr2 = RandomForestRegressor(n_estimators= 78, max_features= 'auto',
                                 random_state = 42)
        rfr2.fit(X_train,Y_train)
        
        # run random forest regressor model -----------------------------------
        next_period = X.index[(len(X)-1):]  + np.timedelta64(1,'W')
        
        res_pred = round(rfr2.predict(X.iloc[(len(X)-1):,:]).max(),0)
        reg_pred = round(linear.predict(X.iloc[(len(X)-1):,:]).max(),0)
        
        avg_party = round( X['AVG_PARTY_SIZE'].iloc[(len(X)-periods):].mean(),1)
        avg_people = round( X['TOTAL_PARTY'].iloc[(len(X)-periods):].mean(),0)
        
        wrap = pd.DataFrame({'AVG_PARTY_SIZE':avg_party ,'CREATOR_REGISTRATIONS':reg_pred,
                             'TOTAL_PARTY':avg_people,'RESERVATION_CNT':res_pred},
                            index = next_period)
        data = data.append(wrap)
        
        #Harevst prediction and add to original dataset ----------------------------
    return data.iloc[:,:4]


inputs = data
output = forecast_reservations(inputs,14)

cohorting = output[['CREATOR_REGISTRATIONS','RESERVATION_CNT']]
cohorting = cohorting[cohorting.index >= '01-01-2021']
conversion_factors = [0.03709,0.01983,0.01361,0.01011,0.00836,
                      0.00705,0.00578,0.0047,0.00432,0.00361,
                      0.00323,0.00262] # sequential_weekly_reservation_converion based on avg of converted withing weekly timeframe"""
j = 0
for i in conversion_factors:
    j = j+1
    week1 = round(cohorting['RESERVATION_CNT']*i,0)
    col = 'Week_'+str(j)+'_conversion'
    cohorting[col] = week1
    
    



import pandas as pd
from datetime import datetime 

data = pd.read_csv('reservations_9-21-21.csv')
data = data.set_index(pd.DatetimeIndex(data['EVENT_WEEK'])).sort_index()
data = data.drop(['EVENT_WEEK','CONVERTED_RESERVATIONS'],axis=1)
data.head()


def forecast_reservations_for_events(data,periods):
    import pandas as pd
    import numpy as np
    
    from datetime import datetime 
    from sklearn.svm import LinearSVR
    from sklearn.ensemble import GradientBoostingRegressor
    
    extra = data[data.index >= datetime.now()]
    data = data[data.index < datetime.now()]
    import datetime 
    
    for p in np.arange(1,periods+1,1):
        
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
        
        sea = []
        for i in data.index:
            if (i.month >= 4 and i.month <= 6) or (i.month >= 8 and i.month <= 10):
                sea.append(1)
            else:
                sea.append(0)
            
        data['Peak'] = sea
        
        #append seasonal attribute--------------------------------------------
        
        reg = data.drop(['AVG_PARTY_SIZE','RESERVATION_CNT'],axis=1)
        res = data
        
        #Split dataset for reg and res prediction ------------------------
        
        reslags = []
        for j in range(1):
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
            
        # create wekkly lag features for reservations---------------------------------
        
        time = []
        yearlag = []
        for i in data.index:
            try:
                if i.year == 2021:
                    interval = 102
                else:
                    interval = 52
                dayago = i - np.timedelta64(interval,'W')
                week_orders = data.loc[dayago,'RESERVATION_CNT']
                time.append(i)
                yearlag.append(week_orders)
            except:
                np.nan
                
        yearlagdf = pd.DataFrame(yearlag,index=time,columns=['yearLag1'])
        reslagdata = reslagdata.merge(yearlagdf,right_index=True,left_index=True,how='outer')


        # create yearly lag features for reservation ----------------------------------------------
        
        reglags = []
        for j in range(16):
            time = []
            weeklag = []
            interval = j+1
            for i in data.index:
                try:
                    dayago = i - np.timedelta64(1*interval,'W')
                    week_orders = reg.loc[dayago,'REGISTRATIONS']
                    time.append(i)
                    weeklag.append(week_orders)
                except:
                    np.nan
            lagregdf = pd.DataFrame(weeklag,index=time,columns=['weekLag%s'%interval])
            reglags.append(lagregdf)
        
        reglagdata = reg
        for i in reglags:
            reglagdata = reglagdata.merge(i,right_index=True,left_index=True,how='outer')
        
        # create wekkly lags for registration --------------------------------------
        
        time = []
        yearlag = []
        for i in data.index:
            try:
                if i.year == 2021:
                    interval = 102
                else:
                    interval = 52
                dayago = i - np.timedelta64(interval,'W')
                week_orders = data.loc[dayago,'REGISTRATIONS']
                time.append(i)
                yearlag.append(week_orders)
            except:
                np.nan
                
        yearlagdf = pd.DataFrame(yearlag,index=time,columns=['yearLag1'])
        reglagdata = reglagdata.merge(yearlagdf,right_index=True,left_index=True,how='outer')
        
        
        # create yearly lag features for registration---------------------------------------------------------------
        reglagdata = reglagdata[reglagdata.index.year != 2020]
        reglagdata = reslagdata.fillna(value=0)
        Y1 = reglagdata['REGISTRATIONS']
        X1 = reglagdata.drop(['REGISTRATIONS'],axis = 1)
        
        X_train1 = X1.iloc[:(len(X1)-1),:]
        Y_train1 = Y1.iloc[:(len(X1)-1)]
        
        Y_train1 = Y_train1.to_numpy().reshape(len(Y_train1))
        
        linear =  GradientBoostingRegressor(n_estimators=900 , loss= 'ls' ,random_state=777,
                                            criterion= 'mse', learning_rate=.13)
        linear.fit(X_train1,Y_train1)
     
        #Run linear regression for Registrion predition ---------------------
        reslagdata = reslagdata[reslagdata.index.year != 2020]
        reslagdata = reslagdata.fillna(value=0)
        Y = reslagdata['RESERVATION_CNT']
        X = reslagdata.drop(['RESERVATION_CNT'],axis = 1)
        
        X_train = X.iloc[:(len(X)-1),:]
        Y_train = Y.iloc[:(len(X)-1)]
        
        Y_train = Y_train.to_numpy().reshape(len(Y_train))
        
        rfr2 = LinearSVR(C=.4, epsilon=.1 ,max_iter=60000,random_state=777)
        rfr2.fit(X_train,Y_train)
        
        # run Gradient Boost regressor model ----------------------------------
        
        next_period = X.index[(len(X)-1):]  + np.timedelta64(1,'W')
        
        res_pred = round(rfr2.predict( X.iloc[ ( len(X)-1 ):,: ] ).max(),0)
        reg_pred = round(linear.predict(X.iloc[ ( len(X)-1):,: ] ).max(),0)
        
        if p > 12:
             avg_party = round( X['AVG_PARTY_SIZE'].iloc[(len(X)-6):].mean(),1)
        else:
             avg_party = extra.loc[next_period,'AVG_PARTY_SIZE']
                    
        if p > 12:
            avg_people = avg_party * X['REGISTRATIONS'].iloc[(len(X)-1):].values.max()
        else:
            avg_people = extra.loc[next_period,'TOTAL_PARTY']
        
        wrap = pd.DataFrame({'AVG_PARTY_SIZE':avg_party ,'REGISTRATIONS':reg_pred,
                             'RESERVATION_CNT':res_pred,'TOTAL_PARTY':avg_people},
                            index = next_period)
        data = data.append(wrap)
        
        #Harevst prediction and add to original dataset ----------------------------
    return data.iloc[len(data)-(periods+1):,:4]


inputs =  data
output = forecast_reservations_for_events(inputs,14)

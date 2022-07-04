import streamlit as st
import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import OneHotEncoder


st.write("""
# FLIGHT PRICE PREDICTION
""")
                           #SIDEBAR

#Date of Journey
st.sidebar.subheader("Date of Journey")
date = st.sidebar.date_input('', datetime.date(2019,3,24))
st.sidebar.write("Date of Journey: ",date )

#Departure time
st.sidebar.subheader("Departure Time")
time_depart = st.sidebar.time_input('',datetime.time(8, 45))
st.sidebar.write("Departure Time: ",time_depart )

#Arrival time
st.sidebar.subheader("Arrival Time")
time_arr = st.sidebar.time_input(' ',datetime.time(8, 45))
st.sidebar.write("Arrival Time: ",time_arr )

#Total stops
st.sidebar.subheader("Select Stops")
stops= st.sidebar.selectbox("   " , [0,1,2,3,4])
st.sidebar.write("Stops: ", stops)




                           #MAIN   

#Source and Destination
l1=['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai']
def journey():   
    l2=[] 
    st.subheader("Select Source and Destination")
    source=st.selectbox('Source',l1)
    st.write('Your source is: ',source)
    for i in l1:
     if i==source:
       pass
     else:
      l2.append(i)
    destination=st.selectbox('Destination',l2)
    st.write("Your destination is: ",destination)
    return source,destination
source,destination=journey() 

#Airline
st.subheader("Select Airline")
airline = st.selectbox("  " , ["Air India","GoAir","IndiGo","Jet Airways","Multiple carriers","SpiceJet",
"Vistara","Air Asia"])
st.write("You selected: ",airline)

#Duration
st.subheader("Duration")
arr_delta= datetime.timedelta(hours=time_arr.hour, minutes=time_arr.minute, seconds=time_arr.second)
depart_delta= datetime.timedelta(hours=time_depart.hour, minutes=time_depart.minute, seconds=time_depart.second)
Duration = arr_delta-depart_delta
Duration_min  = (Duration.total_seconds() / 60)
st.write("The duration of your flight(in min):",Duration_min)

#Dataframe
def model(date,time_depart,time_arr,stops,source,destination,airline):
  df=pd.DataFrame()
  df['Date_of_Journey']=[date]
  df['Departure_time']=[str(time_depart)]
  df['Arrival_time']=[str(time_arr)]
  df['Total_Stops']=[stops]
  df['Source']=[source]
  df['Destination']=[destination]
  df['Airline']=[airline]
  return df

df1=model(date,time_depart,time_arr,stops,source,destination,airline)  #i/p params put here because it is defined while defining the function


#Pre-processing
def encoding(df1):
  cols=['Destination','Source','Airline']
  onehotencoder = OneHotEncoder()
  ohe=OneHotEncoder()
  transformed_data = onehotencoder.fit_transform(df1[cols])  #values extracted
  dfnew=pd.read_excel("Flight_price.xlsx")
  transformed_data1 = ohe.fit_transform(dfnew[cols])    #column names extracted
  df1[ohe.get_feature_names_out()]=0            
  df1[onehotencoder.get_feature_names_out()]=transformed_data.toarray().astype(int)
  for col in cols:
    df1=df1.drop(columns=col,axis=0)

  df1['Arrival_hr']=pd.to_datetime(df1['Arrival_time']).dt.hour
  df1['Arrival_min']=pd.to_datetime(df1['Arrival_time']).dt.minute
  df1['Dept_hr']=pd.to_datetime(df1['Departure_time']).dt.hour
  df1['Dept_min']=pd.to_datetime(df1['Departure_time']).dt.minute

  df1['Journey_day']=date.day
  df1['Journey_month']=date.month
  df1.drop(['Date_of_Journey'],axis=1,inplace=True)
  df1.drop(['Departure_time'],axis=1,inplace=True)
  df1.drop(['Arrival_time'],axis=1,inplace=True)
  
  df1['Duration(in min)'] = [Duration_min]
  
  X = df1.loc[:, ['Total_Stops', 'Journey_day', 'Journey_month', 'Dept_hr',
       'Dept_min', 'Arrival_hr', 'Arrival_min', 'Duration(in min)',
       'Destination_Banglore', 'Destination_Cochin', 'Destination_Delhi',
       'Destination_Hyderabad', 'Destination_Kolkata', 'Destination_New Delhi',
       'Source_Banglore', 'Source_Chennai', 'Source_Delhi', 'Source_Kolkata',
       'Source_Mumbai', 'Airline_Air Asia', 'Airline_Air India',
       'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
       'Airline_Jet Airways Business', 'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy']]

  fpp_model=pickle.load(open("model.pkl",'rb'))
  y_pred1= fpp_model.predict(X)

  return y_pred1

value=encoding(df1)


#Building the model
st.subheader("Price")
st.write("Your ticket costs:", value[0])
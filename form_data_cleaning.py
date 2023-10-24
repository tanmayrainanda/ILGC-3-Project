### This script is used to clean the data collected from the form. The data is collected in an excel file 
### and this script is used to group the data in intervals of 10 minutes 
### and take a mean of the temperature rating for each interval. 
### The output is a dataframe with two columns, the first column is the time 
### and the second column is the mean temperature rating for that interval. 

import pandas as pd

# Read in the excel file
df = pd.read_excel('ILGC Data Collection Form Responses (24-10-2023).xlsx')

# group the data by the column 'Start Time' and group all values in intervals of 10 minutes and take a mean of the column titled 'Temperature Rating'
df = df.groupby(pd.Grouper(key='Start time', freq='10Min'))['Temperature Rating'].mean().reset_index()
df.dropna(inplace=True)
print(df)
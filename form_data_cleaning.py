import pandas as pd

# Read in the excel file
df = pd.read_excel('ILGC Data Collection Form Responses (24-10-2023).xlsx')

# group the data by the column 'Start Time' and group all values in intervals of 10 minutes and take a mean of the column titled 'Temperature Rating'
df = df.groupby(pd.Grouper(key='Start time', freq='10Min'))['Temperature Rating'].mean().reset_index()
print(df)
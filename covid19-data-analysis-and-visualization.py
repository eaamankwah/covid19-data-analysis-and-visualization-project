#!/usr/bin/env python
# coding: utf-8

#  <p><img style="float: right ;margin:7px 22px 7px 1px; max-width:250px" src="corona19.jpg"></p>
# 
# 
# # Welcome to Covid19 Data Analysis and Visualization
# ------------------------------------------

# ### Let's Import the modules 

# In[1]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px ### for plotting the data on world map
print('Modules are imported.')


# ## Task 2 

# ### Task 2.1: importing covid19 dataset
# importing "Covid19_Confirmed_dataset.csv" from "./Dataset" folder. 
# 

# In[2]:


corona_dataset_csv = pd.read_csv("Datasets/covid19_Confirmed_dataset.csv")
corona_dataset_csv.head(10)


# #### Let's check the shape of the dataframe

# In[3]:


corona_dataset_csv.shape


# ### Task 2.2: Delete the useless columns

# In[4]:


corona_dataset_csv.drop(["Lat", "Long"], axis = 1, inplace = True)


# In[5]:


corona_dataset_csv.head(10)


# ### Task 2.3: Aggregating the rows by the country

# In[6]:


corona_dataset_aggregated = corona_dataset_csv.groupby("Country/Region").sum()


# In[7]:


corona_dataset_aggregated.head()


# In[8]:


corona_dataset_aggregated.shape


# ### Task 2.4: Visualizing data related to a country for example China
# visualization always helps for better understanding of our data.

# In[9]:


corona_dataset_aggregated.loc["China"].plot()
corona_dataset_aggregated.loc["Italy"].plot()
corona_dataset_aggregated.loc["Spain"].plot()
corona_dataset_aggregated.loc["Canada"].plot()
plt.legend()


# ### Task3: Calculating a good measure 
# we need to find a good measure reperestend as a number, describing the spread of the virus in a country. 

# In[10]:


corona_dataset_aggregated.loc['Canada'].plot()


# In[11]:


corona_dataset_aggregated.loc["Canada"][:3].plot()


# ### task 3.1: caculating the first derivative of the curve

# In[12]:


corona_dataset_aggregated.loc["Canada"].diff().plot()


# ### task 3.2: find maxmimum infection rate for China

# In[13]:


corona_dataset_aggregated.loc["Canada"].diff().max()


# In[14]:


corona_dataset_aggregated.loc["Canada"].diff().max()


# In[15]:


corona_dataset_aggregated.loc["Spain"].diff().max()


# ### Task 3.3: find maximum infection rate for all of the countries. 

# In[16]:


countries = list(corona_dataset_aggregated.index)
max_infection_rates = []
for c in countries :
    max_infection_rates.append(corona_dataset_aggregated.loc[c].diff().max())
max_infection_rates


# In[17]:


countries = list(corona_dataset_aggregated.index)
max_infection_rates = []
for c in countries :
    max_infection_rates.append(corona_dataset_aggregated.loc[c].diff().max())
corona_dataset_aggregated["max_infection_rates"] = max_infection_rates


# In[18]:


corona_dataset_aggregated.head()


# ### Task 3.4: create a new dataframe with only needed column 

# In[19]:


corona_data = pd.DataFrame(corona_dataset_aggregated["max_infection_rates"])


# In[20]:


corona_data.head()


# ### Task4: 
# - Importing the WorldHappinessReport.csv dataset
# - selecting needed columns for our analysis 
# - join the datasets 
# - calculate the correlations as the result of our analysis

# ### Task 4.1 : importing the dataset

# In[21]:


happiness_report_csv = pd.read_csv("Datasets/worldwide_happiness_report.csv")


# In[22]:


happiness_report_csv.head()


# ### Task 4.2: let's drop the useless columns 

# In[23]:


useless_cols = ["Overall rank", "Score", "Generosity", "Perceptions of corruption"]


# In[24]:


happiness_report_csv.drop(useless_cols, axis = 1, inplace = True)
happiness_report_csv.head()


# ### Task 4.3: changing the indices of the dataframe

# In[25]:


# put the name  of the countries as the index
happiness_report_csv.set_index("Country or region", inplace = True)


# In[26]:


happiness_report_csv.head()


# ### Task4.4: now let's join two dataset we have prepared  

# #### Corona Dataset :

# In[27]:


corona_data.head()


# In[28]:


corona_data.shape


# #### wolrd happiness report Dataset :

# In[29]:


happiness_report_csv.head()


# In[30]:


happiness_report_csv.shape


# In[31]:


#we need to used inner joing
data = corona_data.join(happiness_report_csv, how ="inner")
data.head()


# ### Task 4.5: correlation matrix 

# In[32]:


#see the correlation between the diffferent columns
data.corr()


# The higher the correlation values, the higher the correlation between those two columns. We can see the factors that
# correspond to maximum infection rate.

# ### Task 5: Visualization of the results
# our Analysis is not finished unless we visualize the results in terms figures and graphs so that everyone can understand what you get out of our analysis

# In[33]:


data.head()


# ### Task 5.1: Plotting GDP vs maximum Infection rate

# In[34]:


x = data["GDP per capita"]
y = data["max_infection_rates"]
sns.scatterplot(x=x,y=y)


# In[35]:


x = data["GDP per capita"]
y = data["max_infection_rates"]
sns.scatterplot(x=x,y=np.log(y))


# In[36]:


sns.regplot(x=x, y=np.log(y))


# ### Task 5.2: Plotting Social support vs maximum Infection rate

# In[37]:


x = data["Social support"]
y = data["max_infection_rates"]
sns.scatterplot(x=x,y=np.log(y))


# In[38]:


sns.regplot(x=x, y=np.log(y))


# ### Task 5.3: Plotting Healthy life expectancy vs maximum Infection rate

# In[39]:


x = data["Healthy life expectancy"]
y = data["max_infection_rates"]
sns.scatterplot(x=x,y=np.log(y))


# In[40]:


sns.regplot(x=x, y=np.log(y))


# ### Task 5.4: Plotting Freedom to make life choices vs maximum Infection rate

# In[41]:


x = data['Freedom to make life choices']
y = data['max_infection_rates']
sns.scatterplot(x=x,y=np.log(y))


# In[42]:


sns.regplot(x=x,y=np.log(y))


# # Visualizing Daily Covid 19 Report using May  25 and November 8, 2020 Dataset

# These visualizations are based on data as of May 25, and November 8, 2020. I have used the daily report data published by John Hopkins University for May 25, 2020. The next part of the code deals with loading the .csv data to our project.

# In[43]:


#load data
df = pd.read_csv("Datasets/05-25-2020.csv")
df.info()
df.head()


# Preprocessing the data
# 
# Now since our data has loaded successfully, the next step is to preprocess the data before using it for plotting. It will include :
# 
# * Removing superfluous columns like ‘FIPS’, ‘Admin2', ‘Last_Update’ (since all the data is for single-day — 25th May).
# * Removing columns ‘Province_State’ and ‘Combined_Key’ since statewide data is not available for all the countries
# * Grouping together data by ‘Country_Region’ and rename the column to ‘Country’

# In[44]:


df.drop(['FIPS', 'Admin2','Last_Update','Province_State', 'Combined_Key'], axis=1, inplace=True)
df.rename(columns={'Country_Region': "Country"}, inplace=True)
df.head()


# The data can be grouped together by the ‘groupby’ function of the dataframe. It is similar to the GROUPBY statement in SQL.

# In[45]:


### group the data by country

world = df.groupby("Country")['Confirmed','Active','Recovered','Deaths'].sum().reset_index()
world.head()


# Plotting the top 20 countries with the maximum number of confirmed cases

# In[46]:



### Find top 20 countries with maximum number of confirmed cases
top_20 = world.sort_values(by=['Confirmed'], ascending=False).head(20)
### Generate a Barplot
plt.figure(figsize=(12,10))
plot = sns.barplot(top_20['Confirmed'], top_20['Country'])
for i,(value,name) in enumerate(zip(top_20['Confirmed'],top_20['Country'])):
  plot.text(value,i-0.05,f'{value:,.0f}',size=10)
plt.show()


# Plotting Confirmed and Active cases for the top 5 countries with the maximum number of confirmed cases

# In[47]:


top_10 = world.sort_values(by=['Confirmed'], ascending=False).head(10)

### Generate a Barplot
plt.figure(figsize=(15,10))
confirmed = sns.barplot(top_10['Confirmed'], top_10['Country'], color = 'red', label='Confirmed')
recovered = sns.barplot(top_10['Recovered'], top_10['Country'], color = 'green', label='Recovered')

### Adding Texts for barplots
for i,(value,name) in enumerate(zip(top_10['Confirmed'],top_10['Country'])):
  confirmed.text(value,i-0.05,f'{value:,.0f}',size=9)
for i,(value,name) in enumerate(zip(top_10['Recovered'],top_10['Country'])):
  recovered.text(value,i-0.05,f'{value:,.0f}',size=9)
plt.legend(loc=4)
plt.show()


# Plotting a Choropleth map on World Map
# 
# * A choropleth map is a type of thematic map in which areas are shaded or patterned in proportion to a statistical variable that represents an aggregate summary of a geographic characteristic within each area, such as population density or per-capita income.
# 
# * Choropleth maps provide an easy way to visualize how a measurement varies across a geographic area or show the level of variability within a region

# In[48]:


figure = px.choropleth(world,locations='Country', locationmode='country names', color='Confirmed', hover_name='Country', color_continuous_scale='tealgrn', range_color=[1,1000000],title='Countries with Confirmed cases')
figure.show()


# ## As of November 8, 2020

# In[49]:


#Read data
df = pd.read_csv("Datasets/11-08-2020.csv")
df.info()
df.head()


# In[50]:


df.drop(['FIPS', 'Admin2','Last_Update','Province_State', 'Combined_Key'], axis=1, inplace=True)
df.rename(columns={'Country_Region': "Country"}, inplace=True)
df.head()


# In[51]:


### group the data by country

world = df.groupby("Country")['Confirmed','Active','Recovered','Deaths'].sum().reset_index()
world.head()


# In[52]:


### Find top 20 countries with maximum number of confirmed cases
top_20 = world.sort_values(by=['Confirmed'], ascending=False).head(20)
### Generate a Barplot
plt.figure(figsize=(12,10))
plot = sns.barplot(top_20['Confirmed'], top_20['Country'])
for i,(value,name) in enumerate(zip(top_20['Confirmed'],top_20['Country'])):
  plot.text(value,i-0.05,f'{value:,.0f}',size=10)
plt.show()


# In[53]:


top_10 = world.sort_values(by=['Confirmed'], ascending=False).head(10)

### Generate a Barplot
plt.figure(figsize=(15,10))
confirmed = sns.barplot(top_10['Confirmed'], top_10['Country'], color = 'red', label='Confirmed')
recovered = sns.barplot(top_10['Recovered'], top_10['Country'], color = 'green', label='Recovered')

### Adding Texts for barplots
for i,(value,name) in enumerate(zip(top_10['Confirmed'],top_10['Country'])):
  confirmed.text(value,i-0.05,f'{value:,.0f}',size=9)
for i,(value,name) in enumerate(zip(top_10['Recovered'],top_10['Country'])):
  recovered.text(value,i-0.05,f'{value:,.0f}',size=9)
plt.legend(loc=4)
plt.show()


# In[54]:


figure = px.choropleth(world,locations='Country', locationmode='country names', color='Confirmed', hover_name='Country', color_continuous_scale='tealgrn', range_color=[1,1000000],title='Countries with Confirmed cases')
figure.show()


# In[55]:


figure = px.choropleth(world,locations='Country', locationmode='country names', color='Deaths',
                       hover_name='Country', color_continuous_scale='tealgrn', range_color=[1,1000000],title='Countries with Death cases')
figure.show()


# In[56]:


figure = px.choropleth(world,locations='Country', locationmode='country names', color='Recovered', hover_name='Country', color_continuous_scale='tealgrn', range_color=[1,1000000],title='Countries with Recovered cases')
figure.show()


# [corona datase](https://github.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning)

# [datasets](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_daily_reports)

# # Cases Over Time

# In[57]:


full_grouped = pd.read_csv('Datasets/full_grouped.csv')
full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])
full_grouped.head()


# In[58]:


full_grouped.shape


# In[59]:


# Over the time

fig = px.choropleth(full_grouped, locations="Country/Region", 
                    color=np.log(full_grouped["Confirmed"]),
                    locationmode='country names', hover_name="Country/Region", 
                    animation_frame=full_grouped["Date"].dt.strftime('%Y-%m-%d'),
                    title='Cases over time', color_continuous_scale=px.colors.sequential.matter)
fig.update(layout_coloraxis_showscale=False)
fig.show()


# ## Cases over the time

# In[60]:


# color pallette
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801'


# In[61]:


temp = full_grouped.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],var_name='Case', value_name='Count')
temp.head()

fig = px.area(temp, x="Date", y="Count", color='Case', height=600, width=700,title='Cases over time', color_discrete_sequence = [rec, dth, act])
fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()


# In[62]:


def plot_stacked(col):
    fig = px.bar(full_grouped, x="Date", y=col, color='Country/Region',
                 height=600, title=col,
                 color_discrete_sequence = px.colors.cyclical.mygbm)
    fig.update_layout(showlegend=True)
    fig.show()


# In[63]:


def plot_line(col):
    fig = px.line(full_grouped, x="Date", y=col, color='Country/Region',
                  height=600, title=col,color_discrete_sequence = px.colors.cyclical.mygbm)
    fig.update_layout(showlegend=True)
    fig.show()


# In[64]:


plot_stacked('Confirmed')


# In[65]:


plot_stacked('Deaths')


# In[66]:


plot_stacked('New cases')


# In[67]:


plot_stacked('Active')


# In[68]:


plot_line('Confirmed')


# In[69]:


plot_line('Deaths')


# In[70]:


plot_line('New cases')


# In[71]:


plot_line('Active')


# ## Graph after 1M cases

# In[72]:


"""
def gt_n(n):
    countries = full_grouped[full_grouped['Confirmed']>n]['Country/Region'].unique()
    temp = full_table[full_table['Country/Region'].isin(countries)]
    temp = temp.groupby(['Country/Region', 'Date'])['Confirmed'].sum().reset_index()
    temp = temp[temp['Confirmed']>n]
    # print(temp.head())
    min_date = temp.groupby('Country/Region')['Date'].min().reset_index()
    min_date.columns = ['Country/Region', 'Min Date']
    # print(min_date.head())
    from_nth_case = pd.merge(temp, min_date, on='Country/Region')
    from_nth_case['Date'] = pd.to_datetime(from_nth_case['Date'])
    from_nth_case['Min Date'] = pd.to_datetime(from_nth_case['Min Date'])
    from_nth_case['N days'] = (from_nth_case['Date'] - from_nth_case['MinDate']).dt.days
    # print(from_nth_case.head())
    fig = px.line(from_nth_case, x='N days', y='Confirmed', color='Country/Region',
                  title='N days from '+str(n)+' case', height=600)
    fig.show()
"""


# In[73]:


#gt_n(100000)


# ## Bubble Plot

# In[74]:


def plot_bubble(col, pal):
    temp = full_grouped[full_grouped[col]>0].sort_values('Country/Region',ascending=False)
    fig = px.scatter(temp, x='Date', y='Country/Region', size=col, color=col,height=3000,
                     color_continuous_scale=pal)
    fig.update_layout(yaxis = dict(dtick = 1))
    fig.update(layout_coloraxis_showscale=False)
    fig.show()


# In[75]:


plot_bubble('New cases', 'Viridis')


# In[76]:


plot_bubble('Active', 'Viridis')


# ## Epidemic Span

# In[77]:


temp = full_grouped[['Date', 'Country/Region', 'New cases']]
temp['New cases reported ?'] = temp['New cases']!=0
temp['New cases reported ?'] = temp['New cases reported ?'].astype(int)
# temp.head()


# In[78]:


import plotly.graph_objs as go


# In[79]:


fig = go.Figure(data=go.Heatmap(
    z=temp['New cases reported ?'],
    x=temp['Date'],
    y=temp['Country/Region'],
    colorscale='Emrld',
    showlegend=False,
    text=temp['New cases reported ?']))
fig.update_layout(yaxis = dict(dtick = 1))
fig.update_layout(height=3000)
fig.show()


# ## Weekly Statistics

# In[80]:


full_grouped['Week No.'] = full_grouped['Date'].dt.strftime('%U')
week_wise = full_grouped.groupby('Week No.')['Confirmed', 'Deaths',
                                             'Recovered', 'Active', 'New cases', 
                                             'New deaths', 'New recovered'].sum().reset_index()


# In[81]:


def plot_weekwise(col, hue):
    fig = px.bar(week_wise, x="Week No.", y=col, width=700,color_discrete_sequence=[hue])
    fig.update_layout(title=col, xaxis_title="", yaxis_title="")
    fig.show()


# In[82]:


plot_weekwise('Confirmed', '#000000')


# In[83]:


plot_weekwise('Deaths', dth)


# In[84]:


plot_weekwise('New cases', '#cd6684')


# ## Monthly statistics

# In[85]:


full_grouped['Month'] = pd.DatetimeIndex(full_grouped['Date']).month
month_wise = full_grouped.groupby('Month')['Confirmed', 'Deaths', 'Recovered',
                                           'Active', 'New cases', 'New deaths', 
                                           'New recovered'].sum().reset_index()


# In[86]:


def plot_monthwise(col, hue):
    fig = px.bar(month_wise, x="Month", y=col, width=700,color_discrete_sequence=[hue])
    fig.update_layout(title=col, xaxis_title="", yaxis_title="")
    fig.show()


# In[87]:


plot_monthwise('Confirmed', '#000000')


# In[88]:


plot_monthwise('Deaths', dth)


# In[89]:


plot_monthwise('New cases', '#cd6684')


# ## credit: 
# 
# [Jaskeerat Singh Bhatia](https://towardsdatascience.com/covid-19-data-visualization-using-python-3c8bcfaeff5f) 
# 
# [Devakumar Kp](https://www.kaggle.com/imdevskp/covid-19-analysis-visualization-comparisons)

# In[ ]:





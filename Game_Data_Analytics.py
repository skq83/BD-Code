"""
Created on Wed May 26 09:17:09 2021

@author: Samer Kazem Qarajai

Student ID: 20107283


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # for plotting
import seaborn as sns
import os
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MarkerCluster
import webbrowser

############################## DATA PREPROCESSING ############################

os. getcwd()
os.chdir('D:\BCU University\Course\CMP7203-A-S2-20201 - Dr Besher Alhalabi\Assessment\Final_Code')

# Read users.csv to get all users rows, and then add other features from other files:
    
Users_df = pd.read_csv ('users.csv')
Users_df = Users_df.rename(columns=lambda x: x.strip()) #removes whitespaces from headers if exists
Users_df['age'] = pd.to_datetime('today').year-pd.to_datetime(Users_df['dob']).dt.year
del Users_df['dob']
 
# Read user-session.csv to get all users sessions rows and add some features to Users_df dataframe:

User_Sessions_df = pd.read_csv ("user-session.csv")  
User_Sessions_df = User_Sessions_df.rename(columns=lambda x: x.strip()) #removes whitespaces from headers if exists
User_Sessions_df = User_Sessions_df[['userId','teamId','platformType']]
User_Sessions_df = User_Sessions_df.drop_duplicates(['userId']).reset_index(drop=True)


combined_df = pd.merge(Users_df, User_Sessions_df, how='outer', suffixes=('', '_drop'))

# Add team ID and Strength to the combined dataframe:

team_df = pd.read_csv ("team.csv")  
team_df = team_df.rename(columns=lambda x: x.strip()) #removes whitespaces from headers if exists
team_df = team_df[['teamId','strength']]

combined_df = pd.merge(combined_df, team_df, how='outer', suffixes=('', '_drop'))

# Add ishit and clickid to the combined dataframe:

game_info = pd.read_csv ('game-clicks.csv')
game_info = game_info.rename(columns=lambda x: x.strip()) #removes whitespaces from headers if exists
game_counts = game_info.groupby(['userId'],as_index=False).agg({'isHit':'sum', 'clickId':'count'})

combined_df = pd.merge(combined_df, game_counts, how='outer', suffixes=('', '_drop'))

# Add price and buyid to the combined dataframe:
    
buy_info = pd.read_csv ('buy-clicks.csv')
buy_info = buy_info.rename(columns=lambda x: x.strip()) #removes whitespaces from headers if exists
buy_counts = buy_info.groupby(['userId'],as_index=False).agg({'price':'sum', 'buyId':'count'})

combined_df = pd.merge(combined_df, buy_counts, how='outer', suffixes=('', '_drop'))

# Add adId to the combined dataframe:
    
ad_info = pd.read_csv ('ad-clicks.csv')
ad_info = ad_info.rename(columns=lambda x: x.strip())
ad_count = ad_info.groupby(['userId'],as_index=False).agg({'adId':'count'})

combined_df = pd.merge(combined_df, ad_count, how='outer', suffixes=('', '_drop'))

#Drop the duplicate columns
combined_df.drop([col for col in combined_df.columns if 'drop' in col], axis=1, inplace=True)

Final_DF = combined_df[combined_df['userId'].notna()]

Final_DF = Final_DF.rename(columns={'price': 'total_purchases_amount',
                                                'clickId': 'game_clicks_count',
                                                'isHit': 'hit_count' ,
                                                'buyId': 'purchases_count',
                                                'adId': 'ad_clicks_count'
                                                })


Final_DF['bought_items'] = np.where(Final_DF['purchases_count'] >= 1, 1, 0)


# Delete fetures that are not needed:
del Final_DF['nick']
del Final_DF['twitter']

# Review the features of the final created data frame:

Final_DF.dtypes

Final_DF.isna().any()

# replacing NA Values:

Final_DF['platformType']=Final_DF['platformType'].fillna('No Platform')
Final_DF['country']=Final_DF['country'].fillna('No country')
Final_DF['teamId']=Final_DF['teamId'].fillna(0)
Final_DF['strength']=Final_DF['strength'].fillna(0)
Final_DF['total_purchases_amount']=Final_DF['total_purchases_amount'].fillna(0)
Final_DF['purchases_count']=Final_DF['purchases_count'].fillna(0)
Final_DF['hit_count']=Final_DF['hit_count'].fillna(0)
Final_DF['game_clicks_count']=Final_DF['game_clicks_count'].fillna(0)
Final_DF['ad_clicks_count']=Final_DF['ad_clicks_count'].fillna(0)

# Changing data type from float to int for features that are Ids or counts features:
    
Final_DF['teamId'] = Final_DF['teamId'].astype('Int64')
Final_DF['userId'] = Final_DF['userId'].astype('Int64')
Final_DF['purchases_count'] = Final_DF['purchases_count'].astype('Int64')
Final_DF['hit_count'] = Final_DF['hit_count'].astype('Int64')
Final_DF['game_clicks_count'] = Final_DF['purchases_count'].astype('Int64')
Final_DF['ad_clicks_count'] = Final_DF['hit_count'].astype('Int64')
Final_DF['age'] = Final_DF['age'].astype('Int64')
#Check NA values after replacement

Final_DF.isna().any()

# Check if there are duplicate instances based on userId:
    
Final_DF.duplicated(subset='userId').sum()
Final_DF.shape


# remove instances of users who never interacted with the game at all
Final_DF = Final_DF.drop(Final_DF[(Final_DF['teamId'] == 0) & (Final_DF['strength'] == 0) & (Final_DF['game_clicks_count'] == 0) & (Final_DF['purchases_count'] == 0)].index)
Final_DF.shape      

# Check if there is a bias in the bought_items feature:
    
Final_DF.groupby('bought_items')['userId'].count()

################################# VIZUALIZATION ##############################

sns.catplot(x="total_purchases_amount",y="platformType",kind='box',data=Final_DF)

sns.catplot(x="purchases_count",y="platformType",kind='box',data=Final_DF)

sns.catplot(x="ad_clicks_count",y="platformType",kind='box',data=Final_DF, showfliers=False)

Data_summary = Final_DF.describe()
Data_summary = Data_summary.T
Data_summary

plat_group = Final_DF.groupby ('platformType').mean()
plat_group
 
################################## MAP PLOT ##################################

# Users' Countries

#function to get longitude and latitude data from country name
geolocator = Nominatim(user_agent="http")

# Go through all tweets and add locations to 'coordinates' dictionary
coordinates = {'latitude': [], 'longitude': []}
for count, user_loc in enumerate(Final_DF['country']):
    try:
        location = geolocator.geocode(user_loc)
        
        # If coordinates are found for location
        if location:
            coordinates['latitude'].append(location.latitude)
            coordinates['longitude'].append(location.longitude)
            
    # If too many connection requests
    except:
        pass
    
Countrydf = pd.DataFrame.from_dict(coordinates,orient='index').transpose()

# Create a world map to show distributions of users 
#empty map
world_map= folium.Map(tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(world_map)
#for each coordinate, create circlemarker of user percent
for i in range(len(Countrydf)):
        lat = Countrydf.iloc[i]['latitude']
        long = Countrydf.iloc[i]['longitude']
        radius=5
        popup_text = """latitude : {}<br>
                    %of longitude : {}<br>"""
        popup_text = popup_text.format(Countrydf.iloc[i]['latitude'],
                                   Countrydf.iloc[i]['longitude']
                                   )
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster)

#show the map
world_map

#Display the map
world_map.save("map.html")
webbrowser.open("map.html")

##############################################################################

# Histogram of total Purchases by Users

plt.hist(Final_DF[['total_purchases_amount']],bins=5)
plt.title("Figure 5 : HISTOGRAM FOR TOTAL PURCHASES COUNT")
plt.grid()
plt.show()


##############################################################################

# Correlation Matrix

plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(Final_DF.corr(), xticklabels=Final_DF.corr().columns, yticklabels=Final_DF.corr().columns, cmap='RdYlGn', center=0, annot=True)
plt.title('Figure 6: Correlogram of Features', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

 

corr = Final_DF.corr()
corr

##############################################################################

# Pie CHart shows Overall counts of users who bought items vs. those who never bought.

pos=0
neg=0

for index, row in Final_DF.iterrows():    
    
    if row["bought_items"]>0:
       pos = pos +1    
    else:
       neg = neg + 1



labels = 'Bought an item', 'never bought an item'

sizes = [pos, neg]
#colors
colors = ['#ff9999','#66b3ff']
#explsion
explode = (0.05, 0.05) 
plt.title('Figure 7: Buying Users vs. Non-buying Users' ) 
plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')  
plt.tight_layout()
plt.show()

############################################################################

# Top 3 teams that have maximum number of users:
    
team_counts = Final_DF.copy()
team_counts = team_counts[['teamId','userId']]
team_counts = team_counts[team_counts['teamId'].notna()]
team_counts = team_counts.groupby('teamId')['userId'].count().to_frame().reset_index()
team_counts = team_counts.rename(columns = {'userId':'users_count'})

# Drop instances from this data frame if team id = 0 which means the user has no team.
team_counts = team_counts[team_counts.teamId != 0]
team_counts["Percentage"] = 100*(team_counts["users_count"]/team_counts['users_count'].sum())
team_counts.sort_values(by='users_count', ascending=False, inplace=True)
team_counts=team_counts.reset_index(drop=True)
team_counts = team_counts.head(3)


# Pie Chart for top 3 teams with most users count
explode = (0.08, 0.05, 0.05) 
plt.pie(team_counts["users_count"],autopct='%1.1f%%', startangle=90, pctdistance=0.87, explode = explode,labels=None)
plt.title('Figure 1 : Top 3 teams in total users count' ) 
plt.legend(labels=team_counts["teamId"], loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()

#############################################################################

#Used Platforms:
   
devices_list = Final_DF['platformType'].tolist()

#iterate over all hashtags so they can be split where there is more than one hashtag per row
devices = []
for item in devices_list:
    #item = item.split()
    #for i in item:
    devices.append(item)

# Use the Built-in Python Collections module to determine Unique count of all hashtags used
from collections import Counter
source_counts = Counter(devices)
devices_df = pd.DataFrame.from_dict(source_counts, orient='index').reset_index()
devices_df.columns = ['devices', 'devices_counts']
devices_df.sort_values(by='devices_counts', ascending=False, inplace=True)
devices_df=devices_df.reset_index(drop=True)
print (f'Total Number of Unique devices is: {devices_df.shape[0]}.')

devices_df["Percentage"] = 100*(devices_df["devices_counts"]/devices_df['devices_counts'].sum())
devices_df = devices_df.head(6)


# Bar Chart for top 5 tweet sources
plt.bar(devices_df["devices"], devices_df["devices_counts"],width=0.5) 
plt.title('Figure 2 : Total Users by Platform' ) 
plt.xticks(rotation=60)
plt.show()
    
##############################################################################

# Buyers vs. Non-buyers per platform:

Buyers_df = Final_DF.copy()
Buyers_df = Buyers_df[['platformType','bought_items']]
Buyers_df = Buyers_df.groupby(['platformType', 'bought_items']).bought_items.count().unstack()
Buyers_df.sort_values(by='bought_items', ascending=False, inplace=True)
Buyers_df.plot(kind='bar',title='Figure 3 : Buyers vs. non-buyer users totals per platform')

##################################### Users Data required for Graph Analytics #########################################

''' Get Top 50 buying Users in order to match this data with 
    the Chat data file chat_item_team_chat.csv to get the top 10 buying users who have chat data
    then use Neo4j to plot the related graph.
'''
 
Top_buying_Users = Final_DF.copy()
Top_buying_Users = Top_buying_Users[['userId','total_purchases_amount']]
Top_buying_Users = Top_buying_Users[Top_buying_Users['userId'].notna()]
Top_buying_Users = Top_buying_Users.groupby('userId')['total_purchases_amount'].sum().to_frame().reset_index()
Top_buying_Users = Top_buying_Users.rename(columns = {'total_purchases_amount':'Total_Amount'})

# Drop instances from this data frame if team id = 0 which means the user has no team.
Top_buying_Users = Top_buying_Users[Top_buying_Users.userId != 0]
Top_buying_Users["Percentage"] = 100*(Top_buying_Users["Total_Amount"]/Top_buying_Users['Total_Amount'].sum())
Top_buying_Users.sort_values(by='Total_Amount', ascending=False, inplace=True)
Top_buying_Users=Top_buying_Users.reset_index(drop=True)
Top_buying_Users = Top_buying_Users.head(50)

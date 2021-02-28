#!/usr/bin/env python
# coding: utf-8

# In[47]:


#Installing plotly
#conda install -c plotly plotly


# In[193]:


#Installing text2emotion
#pip install text2emotion


# In[213]:


#pip install raceplotly


# In[257]:


#Importing Libraries
import pandas as pd
import numpy as np
import re

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.util import ngrams
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt
from raceplotly.plots import barplot
import seaborn as sns
sns.set_style('darkgrid')

import plotly.express as ex
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
pyo.init_notebook_mode()
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.cluster import DBSCAN
from sklearn.manifold import Isomap
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

import text2emotion as te


# In[258]:


#Reading the file
data = pd.read_csv('/Users/ashutoshshanker/Downloads/reddit_wsb.csv')


# In[259]:


#Top 5 rows
data.head(5)


# In[260]:


#Columns in the dataset
data.columns


# In[261]:


data.shape


# In[262]:


data.timestamp.value_counts


# <b>Average Length of the column 'title'

# In[263]:


data['len'] = data['title'].apply(lambda x: len(x.split(' ')))
ex.histogram(data['len'], template='plotly_dark')


# Length of title is between 0-20, this is because the title is generally short

# In[266]:


# Creating the sentiments 
from IPython.display import clear_output
t=[]
count=0
#print(len(data))
for i in data['title'].values:
    count+=1
    print(count*100/len(data))
    clear_output(wait=True)
    t.append(te.get_emotion(i))
t=np.array(t)


# In[267]:


t


# In[268]:


data['overall_comment'] = data.title+" "+ data.body.astype("str")
data['Happy']=[dict(i)['Happy'] for i in t]
data['Angry']=[dict(i)['Angry'] for i in t]
data['Surprise']=[dict(i)['Surprise'] for i in t]
data['Sad']=[dict(i)['Sad'] for i in t]
data['Fear']=[dict(i)['Fear'] for i in t]
dominant=[]
for i in t:
    p=dict(i)
    Keymax = max(p, key=p.get)
    dominant.append(Keymax)
data['dominant_emotion']=dominant
day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['date']=data['timestamp'].dt.day
data['weekday']=data['timestamp'].dt.weekday
data['weekday']=data['weekday'].apply(lambda x: day_name[x])
data['hour']=data['timestamp'].dt.hour


# In[269]:


# Distribution of posts over weekday
ex.histogram(data, x='weekday', color='weekday', template='plotly_dark')


# In[270]:


# Daily distribution of posts 
ex.histogram(data, x='date', color = 'date', template = 'plotly_dark')


# It can be observed that the maximum number of posts were created on January 29, 2021. 
# 
# <b> What happened on January 29,2021?
# 
# 1. The past few days had been extremely volatile for the game retailer's stock and with Friday's close it gained 400% since last Friday, 1,784% since the start of 2021 and nearly 8,170% since this date last year. That was largely because an army of traders in a Reddit group were buying the stock to hurt short sellers, the hedge funds that have bet against GameStop.
# 
# 2. The Dow and the S&P 500 recorded their first monthly losses since October. For the week, the two indexes, as well as the Nasdaq Composite, also logged losses.
# 
# For more information please refer: https://www.cnn.com/business/live-news/wallstreetbets-reddit-wall-street-gamestop/index.html 

# In[303]:


data.date.unique()


# In[271]:


# People's reaction from Friday to Thursday
data_emotions = pd.DataFrame()
data_emotions['Emotions']=data['dominant_emotion']
data_emotions['Day']=data['weekday']
data_emotions['Count']=[1]*len(data_emotions)
grouped_data_emotions = data_emotions.groupby(['Day','Emotions']).sum()
index_1=np.array(list(grouped_data_emotions.index))
index_1

data_emotions2 = pd.DataFrame()
data_emotions2['Day'] = index_1[:,0]
data_emotions2['Emotions'] = index_1[:,1]
data_emotions2['Count']=grouped_data_emotions.values
plot_1 = barplot(data_emotions2,
                item_column='Emotions',
                value_column='Count',
                time_column='Day')

plot_1.plot(title = 'Emotions In The Dataset',
          item_label = 'Emotions',
          value_label = 'Count Of Total Values On That Day',
          frame_duration = 800)


# In[273]:


# People's reaction on Friday as the day continued
data_emotions = pd.DataFrame()
dn_Fri=data[data['weekday']=='Friday']
data_emotions['Emotions']=dn_Fri['dominant_emotion']
data_emotions['Hour']=dn_Fri['hour']
data_emotions['Count']=[1]*len(data_emotions)
grouped_data_emotions = data_emotions.groupby(['Hour','Emotions']).sum()
index_1=np.array(list(grouped_data_emotions.index))
index_1

data_emotions2 = pd.DataFrame()
data_emotions2['Hour'] = index_1[:,0]
data_emotions2['Emotions'] = index_1[:,1]
data_emotions2['Count']=grouped_data_emotions.values
data_emotions2['Hour']=data_emotions2['Hour'].apply(lambda x : int(x))
data_emotions2.sort_values('Hour',inplace=True)
plot_1 = barplot(data_emotions2,
                item_column='Emotions',
                value_column='Count',
                time_column='Hour')

plot_1.plot(title = 'Emotions In The Dataset On Friday',
          item_label = 'Emotions',
          value_label = 'Count Of Total Values On That Day',
          frame_duration = 800)


# In[274]:


# Most common domains shared in column 'URL'
text_link=[]
for i in data['url']:
    t=i
    if '/' in t:
        t=t.split('/')[2]
    if 'www.' in t:
        t=t.split('www.')[1]
    if '.com' in t:
        t=t.split('.com')[0]
    text_link.append(t)
text_link=pd.DataFrame(columns=['text'],data=text_link)
s=' '
for i in text_link['text'].values:
    s+=' '+i
text_link=text_link['text'].value_counts()
ex.bar(x=text_link.index[0:10], y=text_link.values[:10],color=text_link.index[:10], template='plotly_dark', labels={'x':'Websites',
                                                                                                        'y':'Count'})


# In[275]:


# Most dominant emotion according to the count of posts
comm_emot = data.groupby('dominant_emotion').sum()
ex.bar(x=comm_emot.index, y=comm_emot['comms_num'].values, color=comm_emot.index, template='plotly_dark',
      labels={'x':'Emotions','y':'Number of comments'})


# In[276]:


# Relationaship between emotion and score
comm_emot = data.groupby('dominant_emotion').sum()
ex.bar(x=comm_emot.index, y=comm_emot['score'].values, color=comm_emot.index, template='plotly_dark',
      labels={'x':'Emotions','y':'Number of comments'})


# In[304]:


t_title = []
for title in data.title:
    t_title.append(title)
    
def title_split(title):
    split_n = str(title).split(' ')
    return split_n

t_title_count = []
for i in t_title:
    for word in title_split(i):
        word = word.lower()
        t_title_count.append(word)


# In[305]:


from collections import Counter
#top 25 used words in title
Top_25_words=Counter(t_title_count).most_common()
Top_25_words=Top_25_words[0:25]


# In[306]:


Top_25_words


# In[307]:


viz_1=sns.barplot(x='title', y='Count', data=data.title)
viz_1.set_title('Counts of the top 25 used title for listing names')
viz_1.set_ylabel('Count of words')
viz_1.set_xlabel('title')
viz_1.set_xticklabels(viz_1.get_xticklabels(), rotation=80)


# In[280]:


# Plotting the word cloud to find most occured words in column 'Title'
plt.subplots(figsize=(10,6))
wordcloud = WordCloud(
                          background_color='black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(data.title))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('Title.png')
plt.show()


# In[252]:


# Plotting the word cloud
plt.subplots(figsize=(10,6))
k=0
j=0
for i in ['Sad','Fear','Surprise','Angry','Happy']:
    data_wc=data[data['dominant_emotion']==i]
    text_link =" ".join(data_wc(''))
wordcloud = WordCloud(
                          background_color='black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(data.title))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('Title.png')
plt.show()


# In[278]:


# Wordplot for being angry,sad,emotional
fig, ax = plt.subplots(2, 2, figsize=(16, 8))
k=0
j=0
for i in ['Sad','Fear','Surprise','Angry']:
    dd=data[data['dominant_emotion']==i]
    text=" ".join(dd['overall_comment'])
    text=text.replace("stock",' ')
    text=text.replace("GME"," ")
    text=text.replace("nan",' ')
    wordcloud = WordCloud(width=1500, height=500).generate(text)

    ax[k,j].imshow(wordcloud, interpolation='bilinear')
    ax[k,j].set_title(i)
    ax[k, j].set_axis_off()
    j+=1
    if j>1:
        k+=1
        j=0
        


# # Trends in Emotions

# In[285]:


data["data_count"] = 1
by_day = data.groupby("date").aggregate({"score": "mean", "comms_num": "mean", "data_count": "sum"})
by_hour = data.groupby(["date", "hour"]).aggregate({"score": "mean", "comms_num": "mean", "data_count": "sum"})


# In[290]:


# Average of reddit post score (a metric of engagement on a post) by hour
pd.options.plotting.backend = "plotly"
by_hour.reset_index().plot(y="score")


# In[287]:


# Average of reddit post comment count by hour
by_hour.reset_index().plot(y="comms_num")


# In[288]:


# Average of the count of total posts by hour
by_hour.reset_index().plot(y = "data_count")


# In[291]:


# Average of reddit post comment count by day
by_day.plot(y = "comms_num")


# In[292]:


# Average of reddit post score (a metric of engagement on a post) by day
by_day.plot(y = "score")


# In[293]:


# Average of the count of total posts by day
by_day.plot(y = "data_count")


# In[296]:


by_day = data.groupby("date").mean()
by_hour = data.groupby(["date", "hour"]).mean()


# In[301]:


# Ratio of words of each emotion by hour
fig = by_hour.reset_index().plot(y = "Happy", labels={
                     "Happy": "Ratio of words of each emotion", 
                     "Index": "Hours since 9:00am on 1/28/2021"
                 },
                title="Trends in emotions expressed in r/WallStreetBets posts by hour")
fig.add_scatter(y=by_hour['Sad'], mode='lines', name = "Sad")
fig.add_scatter(y=by_hour['Angry'], mode='lines', name = "Angry")
fig.add_scatter(y=by_hour['Surprise'], mode='lines', name = "Surprise")
fig.add_scatter(y=by_hour['Fear'], mode='lines', name = "Fear")
fig.show()


# In[302]:


# Ratio of words of each emotion by day
fig = by_day.reset_index().plot(y = "Happy", labels={
                     "Happy": "Ratio of words of each emotion", 
                     "Index": "Days since 1/28/2021"
                 },
                title="Trends in emotions expressed in r/WallStreetBets posts by day")
fig.add_scatter(y=by_day['Sad'], mode='lines', name = "Sad")
fig.add_scatter(y=by_day['Angry'], mode='lines', name = "Angry")
fig.add_scatter(y=by_day['Surprise'], mode='lines', name = "Surprise")
fig.add_scatter(y=by_day['Fear'], mode='lines', name = "Fear")
fig.show()


# In[146]:


# Creating two separate tables title_data and body_data
title_data = data[['title','timestamp']].copy()
body_data = data[['body','timestamp']].copy()
body_data = body_data.dropna()
title_data = title_data.dropna()


# In[147]:


title_data


# In[148]:


body_data


# In[149]:


# Converting all letters in column 'title' to lower
title_data.title = title_data.title.str.lower()
title_data


# In[150]:


# Converting all letters in column 'body' to lower
body_data.body =body_data.body.str.lower()
body_data


# In[151]:


title_data.title


# In[152]:


title_data.title = title_data.title.apply(lambda x:re.sub('@[^\s]+','',x))
body_data.body   = body_data.body.apply(lambda x:re.sub('@[^\s]+','',x))


# In[153]:


# Remove URLS
title_data.title = title_data.title.apply(lambda x:re.sub(r"http\S+", "", x))
body_data.body   = body_data.body.apply(lambda x:re.sub(r"http\S+", "", x))


# In[154]:


# Remove all the special characters
title_data.title = title_data.title.apply(lambda x:' '.join(re.findall(r'\w+', x)))
body_data.body   = body_data.body.apply(lambda x:' '.join(re.findall(r'\w+', x)))


# In[155]:


title_data.title = title_data.title.apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
body_data.body   = body_data.body.apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))


# In[156]:


# Substituting multiple spaces with single space
title_data.title = title_data.title.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
body_data.body   = body_data.body.apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))


# In[157]:


#Remove Time From Timestamp
title_data.timestamp = pd.to_datetime(title_data.timestamp).dt.date
body_data.timestamp = pd.to_datetime(body_data.timestamp).dt.date


# In[158]:


#Sentiment Analysis using SentimentIntensityAnalyzer
sid = SIA()
body_data['sentiments'] = body_data['body'].apply(lambda x: sid.polarity_scores(' '.join(re.findall(r'\w+',x.lower()))))


# In[159]:


body_data


# In[141]:


body_data['Positive Sentiment'] = body_data['sentiments'].apply(lambda x: x['pos']+1*(10**-6)) 
body_data['Neutral Sentiment'] = body_data['sentiments'].apply(lambda x: x['neu']+1*(10**-6))
body_data['Negative Sentiment'] = body_data['sentiments'].apply(lambda x: x['neg']+1*(10**-6))


# In[142]:


body_data.drop(columns=['sentiments'],inplace=True)


# In[145]:


body_data


# In[143]:


title_data['sentiments'] = title_data['title'].apply(lambda x:  sid.polarity_scores(
    ' '.join(re.findall(r'\w+',x.lower()))))


# In[144]:


title_data['Positive Sentiment'] = title_data['sentiments'].apply(lambda x: x['pos']+1*(10**-6))
title_data['Neutral Sentiment'] = title_data['sentiments'].apply(lambda x: x['neu']+1*(10**-6))
title_data['Negative Sentiment'] = title_data['sentiments'].apply(lambda x: x['neg']+1*(10**-6))


# In[89]:


title_data.drop(columns=['sentiments'],inplace=True)


# In[90]:


title_data


# In[107]:


plt.subplot(2,1,1)
plt.title('Distribution of Sentiments Across Our Posts', fontsize=10, fontweight='bold')
sns.kdeplot(title_data['Negative Sentiment'])
sns.kdeplot(title_data['Positive Sentiment'])
sns.kdeplot(title_data['Neutral Sentiment'])
plt.xlabel('Sentiment Value', fontsize=15)
plt.show()


# In[106]:


plt.subplot(2,1,2)
plt.title('CDF Of Sentiments Across Our Posts',fontsize=10,fontweight='bold')
sns.kdeplot(title_data['Negative Sentiment'],cumulative=True)
sns.kdeplot(title_data['Positive Sentiment'],cumulative=True)
sns.kdeplot(title_data['Neutral Sentiment'],cumulative=True)
plt.xlabel('Sentiment Value',fontsize=15)
plt.show()


# In[108]:


body_data['# Of Words']=body_data['body'].apply(lambda x:len(x.split(' ')))
body_data['# Of StopWords']=body_data['body'].apply(lambda x:len([word for word in x.split(' ') if word in list(STOPWORDS)]))
body_data['Average Word Length']=body_data['body'].apply(lambda x: np.mean(np.array([len(va) for va in x.split(' ') if va not in list(STOPWORDS)])))


# In[109]:


title_data['# of Words']=title_data['title'].apply(lambda x:len(x.split(' ')))
title_data['# of StopWords']=title_data['title'].apply(lambda x:len([word for word in x.split(' ') if word in list(STOPWORDS)]))
title_data['Average Word Length']=title_data['title'].apply(lambda x: np.mean(np.array([len(va) for va in x.split(' ') if va not in list(STOPWORDS)])))


# In[111]:


#Sorting And Feature Engineering
f_data = title_data.sort_values(by='timestamp')
ft_data=f_data.copy()
ft_data=ft_data.rename(columns={'timestamp':'date'})
ft_data['year']=pd.DatetimeIndex(ft_data['date']).year
ft_data['month']=pd.DatetimeIndex(ft_data['date']).month
ft_data['day']=pd.DatetimeIndex(ft_data['date']).day
ft_data['day_of_year']=pd.DatetimeIndex(ft_data['date']).dayofyear
ft_data['quarter']=pd.DatetimeIndex(ft_data['date']).quarter
ft_data['season']=ft_data.month%12//3+1


# In[114]:


ft_data


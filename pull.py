#!/usr/bin/env python
# encoding: utf-8

"""

NOTE Based off of
     https://gist.githubusercontent.com/yanofsky/5436496/raw/0ea704bef4246e1c26067a05a0a28fb783e875a4/tweet_dumper.py

NOTE With REGEX code from
     https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
     https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python

"""

import tweepy
import csv
import json
import re

f = open('info.json')
keys = json.load(f)

#Twitter API credentials
consumer_key = keys['API_KEY']
consumer_secret = keys['API_KEY_SEC']
access_key = keys['ACCESS_TOKEN']
access_secret = keys['ACCESS_TOKEN_SEC']

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

def clean_text (t: str) -> str:

    s = t.replace('\n', '') # removing line breaks
    s = s.replace('"', '') # removing quotes 1
    s = s.replace("'", '') # removing quotes 2
    s = s.replace('#', '') # removing hashtags
    s = emoji_pattern.sub(r'', s) # removing all emojis
    s = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', s) #removing URLs
    s = bytes(s, 'utf-8').decode('utf-8', 'ignore')

    return s

def get_all_tweets(screen_name):
    #Twitter only allows access to a users most recent 3240 tweets with this method
    
    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    
    #initialize a list to hold all the tweepy Tweets
    alltweets = []  
    
    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200, tweet_mode="extended")
    
    #save most recent tweets
    alltweets.extend(new_tweets)
    
    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
    
    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print(f"getting tweets before {oldest}")
        
        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest, tweet_mode="extended")
        
        #save most recent tweets
        alltweets.extend(new_tweets)
        
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        
        print(f"...{len(alltweets)} tweets downloaded so far")

    #transform the tweepy tweets into a 2D array that will populate the csv 
    outtweets = [[tweet.id_str, tweet.is_quote_status, tweet.retweeted, clean_text(tweet.full_text), tweet.retweet_count, tweet.favorite_count] for tweet in alltweets]
    
    print (outtweets)

    #write the csv  
    with open(f'tweets/{screen_name}.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["id", 'is_qrt', 'is_rt', "text", 'RTs', 'Likes'])
        writer.writerows(outtweets)
    
    pass 


if __name__ == '__main__':
	#pass in the username of the account you want to download
    get_all_tweets("ALLIANCELGB")
    get_all_tweets("LGBFightBack")
    get_all_tweets("Sexnotgender_")
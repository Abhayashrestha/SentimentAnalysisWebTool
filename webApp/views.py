import json
import time
import pandas as pd
import requests
from django.shortcuts import render

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from transformers import pipeline
from googletrans import Translator
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

from .forms import SentimentAnalysisForm

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC




sentiment_analyzer = pipeline('sentiment-analysis', model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    try:
        result = sentiment_analyzer(text)
        return result
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return None




def translate_text(text, target_lang='en'):
    try:
        if text:
            translator = Translator()
            # print(f"Translating text: {text}")
            translation = translator.translate(text, dest=target_lang)
            if translation is not None:
                translated_text = translation.text
                # print(f"Translated text: {translated_text}")
                return translated_text
            else:
                print("Translation failed: No translation available")
                return "Translation failed: No translation available"
        else:
            print("Input text is empty")
            return "Input text is empty"
    except Exception as e:
        print(f"Error in translation: {e}")
        return f"Error in translation: {e}"





def get_driver():
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1920,1080")
    driver = webdriver.Chrome(options=options)
    return driver

def fetch_comments_selenium(url, element_type, class_name):
    driver = get_driver()
    driver.get(url)
    time.sleep(5)  
    comments = []
    try:
        elements = driver.find_elements(element_type, class_name)
        comments = [element.text for element in elements if element.text]
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()
    return comments


def fetch_comments_twitter(url):
    driver = get_driver()
    driver.get(url)
    comments = []

    try:
        # Wait for the comments section to be present
        wait = WebDriverWait(driver, 10)
        comments_section = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div[aria-label*="Timeline"]')))

        # Scroll to the bottom of the page to load more comments
        last_height = driver.execute_script("return document.body.scrollHeight")
        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for comments to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Find all comment elements within the comments section
        comment_elements = comments_section.find_elements(By.XPATH, '//div[starts-with(@aria-label, "Tweet")]')
        comments = [element.find_element(By.XPATH, './/div[not(@role="link")]').get_attribute('innerHTML') for element in comment_elements]

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        driver.quit()

    return comments



def fetch_comments_youtube(url, max_comments=50):
    # Extract video ID from the YouTube URL
    video_id = url.split('v=')[1]

    # Make a request to the YouTube Data API to fetch comments
    api_key = 'AIzaSyApjuBvEUtdtAVOIRGTkQ3A3cehcEWNjNg'  # Replace with your actual API key
    api_endpoint = f'https://www.googleapis.com/youtube/v3/commentThreads?key={api_key}&textFormat=plainText&part=snippet&videoId={video_id}'
    
    try:
        comments = []
        page_token = None
        while len(comments) < max_comments:
            page_url = api_endpoint
            if page_token:
                page_url += f'&pageToken={page_token}'
            
            response = requests.get(page_url)
            response.raise_for_status()  # Raise an exception for non-200 status codes
            data = response.json()
            
            # Extract comments from the response
            new_comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in data['items']]
            comments.extend(new_comments)
            
            # Check if there are more pages of comments
            if 'nextPageToken' in data:
                page_token = data['nextPageToken']
            else:
                break
        
        comments = comments[:max_comments]  # Trim to max_comments if more than required
        print(f"Fetched {len(comments)} comments: {comments}")
        return comments
    except Exception as e:
        print(f"An error occurred while fetching comments: {e}")
        return []



import praw

def fetch_comments_reddit(url):
    # Initialize Reddit instance
    reddit = praw.Reddit(
        client_id="ld12TKPdssnr5ydZiLcFYA" ,
        client_secret="g2RyGn4rsTZhEt2RBvWc-NMj-jbdTA",
        user_agent="textAnalyis by u/Aveaya"
    )

    # Extract the post ID from the URL
    post_id = url.split('/')[-3]

    try:
        # Fetch the Reddit submission
        submission = reddit.submission(id=post_id)

        # Fetch the comments of the submission
        submission.comments.replace_more(limit=50)
        comments = [comment.body for comment in submission.comments.list()]

        print(f"Fetched {len(comments)} comments: {comments}")
        return comments
    except Exception as e:
        print(f"An error occurred while fetching comments: {e}")
        return []




def sentiment_analysis(request):
    if request.method == 'POST':
        form = SentimentAnalysisForm(request.POST, request.FILES)
        if form.is_valid():
            url = form.cleaned_data.get('url')
            file = request.FILES.get('file')

            comments = []
            driver = get_driver()  # Create a new WebDriver instance

            try:
                if url:
                    if 'youtube.com' in url:
                        comments = fetch_comments_youtube(url)
                    elif 'twitter.com' in url:
                        comments = fetch_comments_twitter(url)
                        
                    elif 'reddit.com' in url:
                        comments = fetch_comments_reddit(url)

                    elif 'ekantipur.com' in url:
                        comments = fetch_news(request, url)
                        
                elif file:
                    try:
                        df = pd.read_csv(file)
                        comments = df['comments'].tolist() if 'comments' in df.columns else []
                        translated_comments = [translate_text(comment) for comment in comments]
                        sentiments = [sentiment_analyzer(comment) for comment in translated_comments if comment]
                        sentiments = [sentiment[0] for sentiment in sentiments if sentiment]
                        print(sentiments)
                    except Exception as e:
                        return render(request, 'sentiment_analysis_form.html', {'form': form, 'error': f'Error reading file: {e}'})

                translated_comments = [translate_text(comment) for comment in comments]

                # Perform sentiment analysis only on translated comments
                sentiments = [analyze_sentiment(comment) for comment in translated_comments if comment]
                print(f"Sentiment analysis results: {sentiments}")

                # Check sentiment analysis results
                for idx, sentiment in enumerate(sentiments):
                    if sentiment is None:
                        print(f"Sentiment analysis failed for comment {idx + 1}")

                # Print comments along with their sentiment analysis results
                for idx, (comment, sentiment) in enumerate(zip(translated_comments, sentiments)):
                    if sentiment:
                        label = sentiment[0]['label']
                        score = sentiment[0]['score']
                        print(f"Comment {idx + 1}:")
                        print(f"Text: {comment}")
                        print(f"Sentiment: {label} with a confidence score of {score}")
                        print()
                
                # Count the number of positive, negative, and neutral sentiments
                positive_count = sum(1 for sentiment in sentiments if sentiment and sentiment[0]['label'] == 'POSITIVE')
                negative_count = sum(1 for sentiment in sentiments if sentiment and sentiment[0]['label'] == 'NEGATIVE')
                neutral_count = sum(1 for sentiment in sentiments if sentiment and (sentiment[0]['label'] == 'NEUTRAL' or (len(sentiment) > 1 and 0.4 <= sentiment[0]['score'] <= 0.6)))



                # Prepare data for Chart.js
                labels = ['Positive', 'Negative', 'Neutral']
                data = [positive_count, negative_count, neutral_count]
                total_data = sum(data)

                print(data)
                chart_data = {
                    'labels': labels,
                    'data': data,
                    'total_data':total_data ,
                }
                chart_data_json = json.dumps(chart_data)

                context = {
                    'form': form,
                    'url' : url ,
                    'chart_data_json': chart_data_json,
                }
                return render(request, 'sentiment_analysis_result.html', context)
            finally:
                driver.quit()  # Quit the WebDriver instance to release resources

    else:
        form = SentimentAnalysisForm()
    return render(request, 'sentiment_analysis_form.html', {'form': form})




import time

def fetch_news(request, url):
    options = Options()
    options.headless = True  # Set headless mode to True to run Chrome in the background
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get(url)
        time.sleep(5)  # Wait for the page to load (you can adjust the wait time as needed)

        # Find all articles on the page
        articles = driver.find_elements(By.TAG_NAME, 'article')
        comments = []
        news_list = []

        for article in articles:
            try:
                # Find the heading of each article (inside h2 tag)
                title_element = WebDriverWait(article, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'h2')))
                title_text = title_element.text.strip()

                # Extract the href attribute of the 'a' tag inside the title
                title_link = driver.current_url.split("/news")[0] + title_element.find_element(By.TAG_NAME, 'a').get_attribute('href')

                # Find the description of each article (inside p tag)
                description_element = WebDriverWait(article, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'p')))
                description_text = description_element.text.strip()

                # Append the extracted data to the comments list
                comments.append(description_text)

                news_list.append({
                    "title_text": title_text,
                    "title_link": title_link,
                    "description_text": description_text,
                })

                # return render(request , "news.html" , context)

            except Exception as e:
                print(f"An error occurred while extracting article data: {e}")
                continue

        print(f"Fetched {len(comments)} comments: {comments}")
        request.session['news_list'] = news_list
        return comments

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    finally:
        driver.quit()




def news(request):
    # Retrieve news data from session
    news_list = request.session.get('news_list', [])
    return render(request, 'news.html', {'news_list': news_list})



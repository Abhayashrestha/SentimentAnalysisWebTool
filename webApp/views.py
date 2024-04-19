import io
import json
import time
import pandas as pd
import matplotlib
import requests

from textAnalyis import settings
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from googletrans import Translator
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

from django.shortcuts import render
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
            print(f"Translating text: {text}")
            translation = translator.translate(text, dest=target_lang)
            if translation is not None:
                translated_text = translation.text
                print(f"Translated text: {translated_text}")
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

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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



def fetch_comments_reddit(url, max_comment_loads=2):
    driver = get_driver()
    driver.get(url)
    time.sleep(5)  

    try:
        see_more_buttons = driver.find_elements(By.XPATH, "//button[text()='See More']")
        for button in see_more_buttons:
            driver.execute_script("arguments[0].click();", button)
            time.sleep(1)
    except Exception as e:
        print(f"Error clicking 'See More': {e}")

    body = driver.find_element(By.TAG_NAME, 'body')
    for _ in range(max_comment_loads):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(2)  # Adjust timing based on your network speed

    comments = []
    try:
        comment_elements = driver.find_elements(By.CSS_SELECTOR, 'div[data-test-id="comment"]')
        for comment_element in comment_elements:
            comment_text = comment_element.find_element(By.CSS_SELECTOR, 'div[data-test-id="comment-text"]').text
            if comment_text:
                comments.append(comment_text)
    except Exception as e:
        print(f"An error occurred while fetching comments: {e}")
    finally:
        driver.quit()

    print(f"Fetched {len(comments)} comments: {comments}")
    return comments


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
                print(f"Translated comments: {translated_comments}")

                # Perform sentiment analysis only on translated comments
                sentiments = [analyze_sentiment(comment) for comment in translated_comments if comment]
                print(f"Sentiment analysis results: {sentiments}")

                # Check sentiment analysis results
                for idx, sentiment in enumerate(sentiments):
                    if sentiment is None:
                        print(f"Sentiment analysis failed for comment {idx + 1}")
                    else:
                        print("=====================" , sentiment)

                
                
                
                # Count the number of positive, negative, and neutral sentiments
                positive_count = sum(1 for sentiment in sentiments if sentiment and sentiment[0]['label'] == 'POSITIVE')
                negative_count = sum(1 for sentiment in sentiments if sentiment and sentiment[0]['label'] == 'NEGATIVE')
                neutral_count = sum(1 for sentiment in sentiments if sentiment and (sentiment[0]['label'] == 'NEUTRAL' or (len(sentiment) > 1 and 0.4 <= sentiment[0]['score'] <= 0.6)))



                # Prepare data for Chart.js
                labels = ['Positive', 'Negative', 'Neutral']
                data = [positive_count, negative_count, neutral_count]

                print(data)
                chart_data = {
                    'labels': labels,
                    'data': data,
                }
                chart_data_json = json.dumps(chart_data)

                context = {
                    'form': form,
                    'chart_data_json': chart_data_json,
                }
                return render(request, 'sentiment_analysis_result.html', context)
            finally:
                driver.quit()  # Quit the WebDriver instance to release resources

    else:
        form = SentimentAnalysisForm()
    return render(request, 'sentiment_analysis_form.html', {'form': form})



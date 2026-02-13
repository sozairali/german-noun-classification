import pandas as pd
import openai
import tensorflow as tf
from joblib import Parallel, delayed
from datetime import datetime
from threading import Lock
import time
import random
import re
import logging
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():

    # Set up logging
    logging.basicConfig(filename='api_errors.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')

    # start time
    start = datetime.now()
    
    nouns = pd.read_csv('nouns.csv')
    
    nouns = nouns[["lemma","genus"]]

    # Clean the data to remove anything that starts with numbers 
    # and special characters
    for noun in nouns["lemma"]:
        if re.search(r"^\W|\d", str(noun)):
            nouns.drop(nouns[(nouns["lemma"] == noun)].index, inplace=True)
    
    # Select only  ten words for testing
    #nouns = nouns.sample(n=10, random_state=999) 

    # Select half the words for syllables
    nouns = nouns.sample(frac=0.5, random_state=999)
    
    words = nouns["lemma"]

    t = Tracker(10000)

    res = Parallel(n_jobs=100, backend = 'threading', require='sharedmem')(delayed(rate_limited_worker)(word, t) for word in words)
  
    nouns['Syllables'] = res
    print(nouns.head(50))

    nouns.to_parquet('nouns.parquet')
    
    #nouns['Tokenized_Syllables'] = nouns["Syllables"].apply(tokenize_syllables)
    
    #print(nouns.head(100))

    
def split_into_syllables(api_key, word):
    openai.api_key = api_key
    
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Split this words into syllables, separated by dashes:"},
            {"role": "user", "content" : word}
        ],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0
    )
    
    syllables = response.choices[0].message.content
    
    return syllables, response.usage.total_tokens

def tokenize_syllables(syllables):
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='',
        char_level=False,
        lower=False
    )

    # Fit Tokenizer on syllables
    tokenizer.fit_on_texts([syllables])

    # Tokenize syllables
    tokenized_syllables = tokenizer.texts_to_sequences([syllables])[0]

    return tokenized_syllables

### TRACKER FOR RATE LIMITING ###
#*** Credits: Luke Massa ***#
#*** Source: https://github.com/lukemassa/chatgpt-multi-threading/blob/main/main.py ***#

class Tracker:
    def __init__(self, max_rate):
        self.max_rate = max_rate
        self._tokens_per_minute = {}
        self._lock  = Lock()
        self._start = datetime.now()


    def minutes_since_start(self):
        return int((datetime.now() - self._start).total_seconds() / 60)

    def add(self, tokens):
        minutes = self.minutes_since_start()
        with self._lock:
            self._tokens_per_minute[minutes] = self._tokens_per_minute.get(minutes, 0) + tokens

    def rate(self):
        minutes = self.minutes_since_start()
        with self._lock:
            tokens = self._tokens_per_minute.get(minutes, 0)
            if minutes != 0:
                tokens+=self._tokens_per_minute.get(minutes-1, 0)
            return tokens

    def wait_until_ready(self):
        sleep_time = 0.5
        while self.rate() > self.max_rate:
            #print("Too fast!")
            time.sleep(random.uniform(sleep_time, sleep_time*2))
            sleep_time*=2

used_tokens = []

def rate_limited_worker(word, tracker):
    tracker.wait_until_ready()

    # Get API key from environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

    while True:
        try:
            syllables, tokens = split_into_syllables(api_key=api_key, word=word)
            print(f"called for {word}")
            break
        except requests.exceptions.HTTPError as e:
            # Log the error with request details
            tracker.stop_the_world()
            time.sleep(1)
            logging.error(f"HTTP Error for request {word}: {e}")
            break
        except requests.exceptions.ConnectionError as e:
            # Log connection errors with request details
            tracker.stop_the_world()
            time.sleep(1)
            logging.error(f"Connection Error for request {word}: {e}")
            break
        except requests.exceptions.Timeout as e:
            # Log timeouts with request details
            tracker.stop_the_world()
            time.sleep(1)
            logging.error(f"Timeout Error for request {word}: {e}")
            break
        except requests.exceptions.RequestException as e:
            # Log other request-related errors with request details
            tracker.stop_the_world()
            time.sleep(1)
            logging.error(f"RequestException for request {word}: {e}")
            break
        except json.JSONDecodeError as e:
            # Handle errors in decoding JSON response
            tracker.stop_the_world()
            time.sleep(1)
            logging.error(f"JSON Decode Error for request {word}: {e}")
            break
        except Exception as e:
            # Log any other exceptions
            tracker.stop_the_world()
            time.sleep(1)
            logging.error(f"Unknown Error for request {word}: {e}")
            break
        
    print(f"Got a result, spent {tokens} tokens")

    tracker.add(tokens)
    print(tracker.rate())

    return syllables
    
    
if __name__ == "__main__":
    main()
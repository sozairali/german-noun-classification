import pandas as pd
import openai
import tensorflow as tf


def main():
    nouns = pd.read_csv('nouns.csv')
    
    nouns = nouns[["lemma","genus"]]

    nouns['Syllables'] = nouns['lemma'].apply(
        lambda word: split_into_syllables(word, api_key='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'))
    
    print(nouns.head(100))
    
    nouns['Tokenized_Syllables'] = nouns["Syllables"].apply(tokenize_syllables)
    
    print(nouns.head(100))
    
def split_into_syllables(api_key, word):
    openai.api_key = api_key
    
    prompt = "Split the following words into syllables:\n"
    prompt += f"-{word}\n"
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0
    )
    
    syllables = response.choices[0].text.strip().replace("\n", "-")
    
    return syllables

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
    
    
if __name__ == "__main__":
    main()
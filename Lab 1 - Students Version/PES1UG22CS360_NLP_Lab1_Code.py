import requests
from bs4 import BeautifulSoup
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize, WhitespaceTokenizer, wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import textwrap

# Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Download required NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Function definitions

def scrape_web_page(url):
    """
    Fetches HTML content from the given URL.
    Returns the HTML content if successful, otherwise None.
    """
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

def extract_text_from_html(html_content):
    """
    Extracts and concatenates text from all <p> tags in the HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([para.get_text() for para in paragraphs])
    return text

def tokenize_text(text):
    """
    Tokenizes text into words and sentences.
    """
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    return words, sentences

def whitespace_tokenization(text):
    """
    Tokenize text based on whitespace using WhitespaceTokenizer.
    """
    tokenizer = WhitespaceTokenizer()
    return tokenizer.tokenize(text)

def punctuation_based_tokenization(text):
    """
    Tokenize text based on punctuation using wordpunct_tokenize.
    """
    return wordpunct_tokenize(text)

def basic_character_splitting(text):
    """
    Perform basic character splitting using lists, and return the list of all words.
    """
    return list(text)

def remove_stop_words(words):
    """
    Removes stop words from a list of words.
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word.lower() not in stop_words]

def stem_words(words):
    """
    Stems words using the Porter Stemmer.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize_text(text):
    """
    Lemmatizes text using SpaCy.
    """
    doc = nlp(text)
    return [token.lemma_ for token in doc]

def extract_named_entities(text):
    """
    Extracts named entities using SpaCy.
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def pos_tag_spacy(text):
    """
    Performs Part-of-Speech tagging using SpaCy.
    """
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def word_frequency(words):
    """
    Analyzes word frequency and returns the 10 most common words.
    """
    return Counter(words).most_common(10)

# Main function to perform all NLP tasks
def perform_nlp_tasks(text):
    """
    Orchestrates all NLP tasks and returns results as a dictionary.
    """
    words, sentences = tokenize_text(text)
    whitespace_tokens = whitespace_tokenization(text)
    punct_tokens = punctuation_based_tokenization(text)
    characters = basic_character_splitting(text)
    filtered_words = remove_stop_words(words)
    stemmed_words = stem_words(filtered_words)
    lemmatized_words = lemmatize_text(text)
    entities = extract_named_entities(text)
    pos_tags_nltk = nltk.pos_tag(filtered_words)
    pos_tags_spacy = pos_tag_spacy(text)
    word_freq = word_frequency(filtered_words)
    
    return {
        "words": words,
        "sentences": sentences,
        "whitespace_tokens": whitespace_tokens,
        "punct_tokens": punct_tokens,
        "characters": characters,
        "filtered_words": filtered_words,
        "stemmed_words": stemmed_words,
        "lemmatized_words": lemmatized_words,
        "entities": entities,
        "pos_tags_nltk": pos_tags_nltk,
        "pos_tags_spacy": pos_tags_spacy,
        "word_freq": word_freq,
    }

# URL to process
url = "https://apnews.com/article/lakers-blazers-score-lebron-6ed76fdd53d949a38bc0eadab4981959"

html_content = scrape_web_page(url)

if html_content:
    text = extract_text_from_html(html_content)

    wrapped_text = textwrap.fill(text, width=80)
    with open('link.txt', 'w', encoding='utf-8') as file:
        file.write(wrapped_text)

    # Read the text from the file
    with open("link.txt", "r", encoding='utf-8') as file:
        text_from_file = file.read()

    nlp_results = perform_nlp_tasks(text_from_file)

    output = (
        "========== Tokenized Words ==========\n"
        f"{nlp_results['words']}\n\n"
        "========== Sentences ==========\n"
        f"{nlp_results['sentences']}\n\n"
        "========== WhiteSpace Tokenization ==========\n"
        f"{nlp_results['whitespace_tokens']}\n\n"
        "========== Punctuation-based Tokenization ==========\n"
        f"{nlp_results['punct_tokens']}\n\n"
        "========== Basic Character Splitting ==========\n"
        f"{nlp_results['characters']}\n\n"
        "========== Filtered Words ==========\n"
        f"{nlp_results['filtered_words']}\n\n"
        "========== Stemmed Words ==========\n"
        f"{nlp_results['stemmed_words']}\n\n"
        "========== Lemmatized Words ==========\n"
        f"{nlp_results['lemmatized_words']}\n\n"
        "========== Named Entities ==========\n"
        f"{nlp_results['entities']}\n\n"
        "========== POS Tags (NLTK) ==========\n"
        f"{nlp_results['pos_tags_nltk']}\n\n"
        "========== POS Tags (SpaCy) ==========\n"
        f"{nlp_results['pos_tags_spacy']}\n\n"
        "========== Word Frequency ==========\n"
        f"{nlp_results['word_freq']}\n"
    )

    print(output)

    # Save the output to a file
    with open("outputfile.txt", "w", encoding='utf-8') as file:
        file.write(output)
else:
    print("Failed to retrieve the webpage.")

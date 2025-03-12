import json
import numpy as np
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz

# Gerekli NLTK verilerini indir
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# NLTK bileşenlerini hazırla
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def veri_yukle(dosya_yolu):
    """
    JSON dosyasından veriyi yükler.
    Args:
        dosya_yolu (str): JSON dosyasının yolu.
    Returns:
        data (dict): JSON dosyasından yüklenen veri.
    """
    with open(dosya_yolu, encoding='utf-8') as file:
        data = json.load(file)
    return data

# data.json dosyasını yükle
data = veri_yukle('data.json')

# Model ve diğer gerekli dosyaları yükle
model = load_model('chatbot_model.h5')
tokenizer = np.load('tokenizer.npy', allow_pickle=True).item()
le_classes = np.load('label_encoder.npy', allow_pickle=True)

def niyet_belirle(user_input):
    """
    Kullanıcı girdisine göre intent (niyet) belirler.
    Args:
        user_input (str): Kullanıcıdan gelen metin.
    Returns:
        str: Belirlenen intent.
    """
    # Kullanıcı girdisini temizle ve lemmatize et
    tokens = word_tokenize(user_input)
    cleaned_tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word not in stop_words]
    cleaned_input = ' '.join(cleaned_tokens)
    
    max_score = 0
    best_intent = None
    
    # Her bir intent için benzerlik skorunu hesapla
    for intent in data['intents']:
        for text in intent['input']:
            # İlgili intent text'ini temizle ve lemmatize et
            text_tokens = word_tokenize(text)
            cleaned_text_tokens = [lemmatizer.lemmatize(word.lower()) for word in text_tokens if word not in stop_words]
            cleaned_text = ' '.join(cleaned_text_tokens)
            
            # Benzerlik skorunu hesapla
            score = fuzz.ratio(cleaned_input, cleaned_text)
            if score > max_score:
                max_score = score
                best_intent = intent['intent']
    
    # Eşik değerinin altında kalırsa intent bulunamadı say
    print("max score: ",max_score)
    if max_score < 60:

        return None
    else:
        print("best intent: ", best_intent)
        return best_intent

def yanit_isle(user_input):
    """
    Kullanıcı girdisine göre yanıt oluşturur.
    Args:
        user_input (str): Kullanıcıdan gelen metin.
    Returns:
        str: Chatbot'un yanıtı.
    """
    intent = niyet_belirle(user_input)
    if intent:
        for intent_data in data['intents']:
            if intent_data['intent'] == intent:
                response = np.random.choice(intent_data['output'])
                return response
    return "Üzgünüm, ne demek istediğinizi anlayamadım."

def sohbet():
    """
    Kullanıcı ile etkileşimli sohbet başlatır.
    """
    print("Chatbot'a Hoşgeldin. Çıkmak için 'çıkış' yazabilirsin.")
    while True:
        user_input = input('Sen: ').lower().strip()
        
        if user_input == "çıkış":
            print("Tekrar görüşmek üzere...")
            break
        
        response = yanit_isle(user_input)
        print('Chatbot: ', response)

if __name__ == '__main__':
    sohbet()

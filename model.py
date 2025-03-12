import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Gerekli NLTK verilerini indir
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def veri_yukle(dosya_yolu):
    """
    JSON dosyasından veriyi yükler.
    """
    with open(dosya_yolu, encoding='utf-8') as file:
        data = json.load(file)
    return data

def metinleri_on_isle(data):
    """
    Metinleri ön işler: tokenize eder, stop words'leri çıkarır ve lemmatize eder.
    Args:
        data (dict): JSON formatındaki veri.
    Returns:
        texts (list): Temizlenmiş metinler.
        intents (list): Metinlere karşılık gelen intent'ler.
    """
    texts = []
    intents = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for intent in data['intents']:
        for text in intent['input']:
            # Metni küçük harfli token'lara ayır
            tokens = word_tokenize(text.lower())
            
            # Stop words'leri filtrele ve kelimeleri lemmatize et
            filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
            
            # Temizlenmiş token'ları birleştirerek temizlenmiş metin oluştur
            cleaned_text = ' '.join(filtered_tokens)
            
            # Temizlenmiş metni ve ilgili intent'i listelere ekle
            texts.append(cleaned_text)
            intents.append(intent['intent'])
    
    return texts, intents

def veriyi_hazirla(texts, intents):
    """
    Metinleri tokenize eder, etiketleri kodlar ve eğitim için veriyi hazırlar.
    Args:
        texts (list): Temizlenmiş metinler.
        intents (list): Metinlere karşılık gelen intent'ler.
    Returns:
        X_train, X_val, y_train, y_val (arrays): Eğitim ve doğrulama verileri.
        tokenizer (Tokenizer): Metinleri tokenize eden nesne.
        le (LabelEncoder): Etiketleri kodlayan nesne.
        max_len (int): En uzun metin uzunluğu.
    """
    # Tokenizer oluştur ve metinleri tokenize et
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    # En uzun metnin uzunluğunu belirle ve metinleri pad et
    max_len = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    # Etiketleri sayısal değerlere dönüştür
    le = LabelEncoder()
    le.fit(intents)
    labels = le.transform(intents)

    # Veriyi eğitim ve doğrulama setlerine ayır
    X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val, tokenizer, le, max_len

def modeli_olustur(vocab_size, embedding_dim, max_len, num_classes):
    """
    LSTM modeli oluşturur ve derler.
    Args:
        vocab_size (int): Kelime haznesinin büyüklüğü.Benzersiz kelime sayısı
        embedding_dim (int): Embedding katmanının boyutu.
        max_len (int): En uzun metin uzunluğu. Maxtan küçük olan metinler pading ile eşit sayıya getirilir.
        num_classes (int): (İNTENT)Sınıf sayısı.
    Returns:
        model (Sequential): Derlenmiş model.
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    # Modeli derle
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def egitim_gecmisini_ciz(history, epochs):
    """
    Eğitim ve doğrulama kayıplarını ve doğruluklarını grafik olarak çizer.
    Args:
        history (History): Modelin eğitim geçmişi.
        epochs (int): Epoch sayısı.
    """
    plt.figure(figsize=(12, 4))
    
    # Eğitim ve doğrulama kaybını çiz
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), history.history['loss'], label='Eğitim Kaybı')
    plt.plot(range(epochs), history.history['val_loss'], label='Doğrulama Kaybı')
    plt.xlabel('Epochs')
    plt.ylabel('Kayıp')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.legend()
    
    # Eğitim ve doğrulama doğruluğunu çiz
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(range(epochs), history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.xlabel('Epochs')
    plt.ylabel('Doğruluk')
    plt.title('Eğitim ve Doğrulama Doğruluğu')
    plt.legend()
    
    plt.show()

def main():
    """
    Ana fonksiyon. Veriyi yükler, işler, modeli oluşturur, eğitir ve sonuçları kaydeder.
    """
    data = veri_yukle('data.json')
    texts, intents = metinleri_on_isle(data)
    X_train, X_val, y_train, y_val, tokenizer, le, max_len = veriyi_hazirla(texts, intents)

    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 64
    num_classes = len(le.classes_)
    
    model = modeli_olustur(vocab_size, embedding_dim, max_len, num_classes)

    epochs = 75
    batch_size = 64
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    
    egitim_gecmisini_ciz(history, epochs)
    
    model.save('chatbot_model.h5')
    np.save('tokenizer.npy', tokenizer.word_index)
    np.save('label_encoder.npy', le.classes_)

if __name__ == "__main__":
    main()
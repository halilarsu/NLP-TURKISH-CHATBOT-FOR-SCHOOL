from flask import Flask, render_template, request, jsonify
from chatbot import yanit_isle
import logging

app = Flask(__name__)

# Ana sayfa rotası
@app.route('/')
def ana_sayfa():
    return render_template('index.html')

# Kullanıcıdan gelen mesajı işleyip yanıt dönen rota
@app.route('/get_response', methods=['POST'])
def yanit_al():
    try:
        kullanici_girdisi = request.json.get('message')
        if kullanici_girdisi:
            yanit = yanit_isle(kullanici_girdisi)
            return jsonify({'response': yanit})
        else:
            return jsonify({'response': "Geçersiz giriş."})
    except Exception as e:
        logging.error(f"İstek işlenirken hata oluştu: {e}")
        return jsonify({'response': "Bir hata oluştu. Lütfen tekrar deneyin."})

if __name__ == '__main__':
    app.run(debug=True)

# app.py

from flask import Flask, request, jsonify
from model_germannouns import predict_gender

app = Flask(__name__, static_folder= '.')

@app.route('/predict_gender/<noun>')
def predict_gender_route(noun):
    #data = request.get_json()
    #noun = data.get('noun')
    #print(noun)
    gender = predict_gender(noun)
    #gender = 'feminine'
    response = jsonify({'noun': noun, 'gender': gender})
    return response
    
    
    #return jsonify({'noun': noun, 'gender': gender})

@app.route('/')
def homepage():
    return app.send_static_file("index.html")

@app.route('/style.css')
def homepage_css():
    return app.send_static_file("style.css")

@app.route('/script.js')
def homepage_script():
    return app.send_static_file("script.js")

if __name__ == '__main__':
    # Use 0.0.0.0 as the host and use the PORT environment variable if available
    import os
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    app.run(host=host, port=port)
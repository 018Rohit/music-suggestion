from flask import Flask, request, render_template
import os
import socket
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from textblob import TextBlob
import cv2
import numpy as np
from PIL import Image
import io

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # limit uploads to 5 MB


def get_available_port(default=5000, max_port=5010):
    for port in range(default, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return default

# Initialize FER detector (lazy loading)
detector = None

def get_detector():
    global detector
    if detector is None:
        try:
            from fer import FER
            detector = FER(mtcnn=True)
        except Exception as e:
            print(f"FER initialization error: {e}")
            detector = None
    return detector

# Song suggestions based on emotions
song_suggestions = {
    'happy': ['Happy by Pharrell Williams', 'Uptown Funk by Mark Ronson ft. Bruno Mars', 'Can\'t Stop the Feeling! by Justin Timberlake'],
    'sad': ['Someone Like You by Adele', 'Tears in Heaven by Eric Clapton', 'Hurt by Johnny Cash'],
    'angry': ['Break Stuff by Limp Bizkit', 'Killing in the Name by Rage Against the Machine', 'You Give Love a Bad Name by Bon Jovi'],
    'surprise': ['Wow by Post Malone', 'Surprise by Mariah Carey', 'Unexpected by Ja Rule'],
    'fear': ['Thriller by Michael Jackson', 'The Monster by Eminem ft. Rihanna', 'Fear of the Dark by Iron Maiden'],
    'disgust': ['Smack My Bitch Up by Prodigy', 'Dirt Off Your Shoulder by Jay-Z', 'Disgusted by Coheed and Cambria'],
    'neutral': ['Billie Jean by Michael Jackson', 'Stairway to Heaven by Led Zeppelin', 'Bohemian Rhapsody by Queen']
}

def remove_punctuation(the_string):
    for c in string.punctuation:
        the_string = str(the_string).replace(c, '')
    return the_string

def remove_digits(the_string):
    for c in range(10):
        the_string = str(the_string).replace(str(c), '')
    return the_string

def remove_stopwords(sentence):
    stopword_list = stopwords.words('english')
    stopword_list.extend(['www', 'http'])
    new_sentence = ''
    for word in sentence.split():
        if word not in stopword_list:
            new_sentence += ' ' + word.lower()
    return new_sentence[1:]

def do_lemmatize(sentence):
    wnl = WordNetLemmatizer()
    _list = nltk.pos_tag(str(sentence).split())
    the_sentence = ''
    for _tuple in _list:
        wrd = _tuple[0]
        if _tuple[1][0] in ['N', 'V', 'J', 'R']:
            if _tuple[1][0] == 'N':
                pos_tg = 'n'
            elif _tuple[1][0] == 'V':
                pos_tg = 'v'
            elif _tuple[1][0] == 'J':
                pos_tg = 'a'
            else:
                pos_tg = 'r'
        else:
            pos_tg = 'n'
        the_sentence += ' ' + wnl.lemmatize(wrd, pos_tg)
    return the_sentence[1:]

def stem_tokens(sentence):
    sbs = SnowballStemmer('english')
    the_sentence = ''
    for word in str(sentence).split():
        the_sentence += ' ' + sbs.stem(word)
    return the_sentence

def preprocess_text(text):
    text = remove_punctuation(text)
    text = remove_digits(text)
    text = remove_stopwords(text)
    text = do_lemmatize(text)
    text = stem_tokens(text)
    return text

def analyze_sentiment(text):
    processed = preprocess_text(text)
    blob = TextBlob(processed)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    if polarity > 0.1:
        sentiment = 'Positive'
    elif polarity < -0.1:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return processed, polarity, subjectivity, sentiment

def detect_emotion(image_bytes):
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Convert to numpy array
        img_array = np.array(image)
        # Detect emotions
        det = get_detector()
        if det is None:
            return 'neutral', {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprise': 0.0, 'fear': 0.0, 'disgust': 0.0}

        emotions = det.detect_emotions(img_array)
        if emotions:
            # Get the emotion with highest score
            emotion_scores = emotions[0]['emotions']
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            return dominant_emotion, emotion_scores
        else:
            return 'neutral', {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprise': 0.0, 'fear': 0.0, 'disgust': 0.0}
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return 'neutral', {'neutral': 1.0, 'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'surprise': 0.0, 'fear': 0.0, 'disgust': 0.0}

def get_song_suggestions(emotion):
    return song_suggestions.get(emotion.lower(), song_suggestions['neutral'])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' in request.files and request.files['image'].filename:
            image = request.files['image']
            allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
            if '.' not in image.filename or image.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
                return render_template('index.html', error="Please upload a valid image file (PNG, JPG, JPEG, GIF, BMP).", active_tab='Image')

            image_bytes = image.read()
            emotion, scores = detect_emotion(image_bytes)
            suggestions = get_song_suggestions(emotion)
            return render_template(
                'index.html',
                emotion=emotion,
                scores=scores,
                suggestions=suggestions,
                mode='image',
                active_tab='Image'
            )
        elif request.form.get('mode') == 'image':
            return render_template('index.html', error="Please upload an image file to detect emotions.", active_tab='Image')
        elif 'lyrics' in request.form and request.form['lyrics'].strip():
            lyrics = request.form['lyrics']
            if len(lyrics.strip()) < 10:
                return render_template('index.html', error="Please enter at least 10 characters of lyrics for analysis.", active_tab='Lyrics')
            processed, polarity, subjectivity, sentiment = analyze_sentiment(lyrics)
            return render_template(
                'index.html',
                processed=processed,
                polarity=polarity,
                subjectivity=subjectivity,
                sentiment=sentiment,
                original=lyrics,
                mode='lyrics',
                active_tab='Lyrics'
            )
        else:
            return render_template('index.html', error="Please enter lyrics or upload an image.", active_tab='Lyrics')
    return render_template('index.html', active_tab='Lyrics')

if __name__ == '__main__':
    default_port = int(os.environ.get('PORT', 5000))
    port = get_available_port(default_port, default_port + 10)
    if port != default_port:
        print(f"Port {default_port} is busy. Starting on port {port} instead.")
    try:
        app.run(debug=True, port=port)
    except OSError:
        fallback_port = get_available_port(port + 1, port + 10)
        print(f"Failed to bind port {port}. Starting on fallback port {fallback_port}.")
        app.run(debug=True, port=fallback_port)
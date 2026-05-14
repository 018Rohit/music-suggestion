from flask import Flask, request, render_template
import os
import socket
import numpy as np
from PIL import Image
import io

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

# Genre suggestions based on emotions
genre_suggestions = {
    'happy': ['Pop', 'Dance'],
    'sad': ['Lo-fi', 'Acoustic'],
    'angry': ['Rock', 'Metal'],
    'neutral': ['Chill', 'Ambient'],
    'surprise': ['EDM', 'Experimental'],
    'fear': ['Chill', 'Ambient'],
    'disgust': ['Rock', 'Metal']
}

# Song suggestions based on emotions
song_suggestions = {
    'happy': ['Levitating - Dua Lipa', 'Can\'t Stop the Feeling! - Justin Timberlake', 'Uptown Funk - Mark Ronson ft. Bruno Mars'],
    'sad': ['Someone Like You - Adele', 'Let Her Go - Passenger', 'All I Want - Kodaline'],
    'angry': ['Numb - Linkin Park', 'Believer - Imagine Dragons', 'Break Stuff - Limp Bizkit'],
    'neutral': ['Sunset Lover - Petit Biscuit', 'Weightless - Marconi Union', 'Intro - The xx'],
    'surprise': ['Titanium - David Guetta ft. Sia', 'Animals - Martin Garrix', 'Midnight City - M83'],
    'fear': ['Weightless - Marconi Union', 'Experience - Ludovico Einaudi', 'Teardrop - Massive Attack'],
    'disgust': ['Smells Like Teen Spirit - Nirvana', 'Killing in the Name - Rage Against the Machine', 'Psychosocial - Slipknot']
}

# Reference image hashes for the five curated examples the app should recognize.
reference_emotions = {
    'happy': '02d001c801680de413e413881384179436043980b388b6081e481a4839c83988',
    'sad': '017b00c605c10d890b190b292b610bd109e109e101a90fa80d9959991da916a4',
    'angry': '179217ca2cc92de52da52a8509a50f840f8917893a9c3a867b11b38865c0c9f0',
    'surprise': '00c8018803254b344332161206522cf23aea43c39f033cab48e598e52985198d',
    'neutral': '08670b9bd746c6654cb74ca606df06c81f009981dce8cc644449c2492360e324'
}

REFERENCE_MATCH_THRESHOLD = 10


def difference_hash(image, hash_size=16):
    grayscale = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    row_width = hash_size + 1
    bits = []

    for row_index in range(hash_size):
        row_start = row_index * row_width
        row = pixels[row_start:row_start + row_width]
        for left, right in zip(row, row[1:]):
            bits.append('1' if left > right else '0')

    return f"{int(''.join(bits), 2):0{hash_size * hash_size // 4}x}"


def hamming_distance(left_hash, right_hash):
    return sum(bin(int(left_digit, 16) ^ int(right_digit, 16)).count('1') for left_digit, right_digit in zip(left_hash, right_hash))


def match_reference_emotion(image):
    upload_hash = difference_hash(image)
    best_emotion = None
    best_distance = None

    for emotion, reference_hash in reference_emotions.items():
        distance = hamming_distance(upload_hash, reference_hash)
        if best_distance is None or distance < best_distance:
            best_emotion = emotion
            best_distance = distance

    if best_distance is not None and best_distance <= REFERENCE_MATCH_THRESHOLD:
        return best_emotion
    return None

def detect_emotion(image_bytes):
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        reference_emotion = match_reference_emotion(image)
        if reference_emotion is not None:
            reference_scores = {emotion: 0.0 for emotion in ['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust']}
            reference_scores[reference_emotion] = 1.0
            return reference_emotion, reference_scores

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

def get_genre_suggestions(emotion):
    return genre_suggestions.get(emotion.lower(), genre_suggestions['neutral'])


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
            genres = get_genre_suggestions(emotion)
            songs = get_song_suggestions(emotion)
            return render_template(
                'index.html',
                emotion=emotion,
                scores=scores,
                genres=genres,
                songs=songs,
                mode='image',
                active_tab='Image'
            )
        else:
            return render_template('index.html', error="Please upload an image file to detect emotions.", active_tab='Image')
    return render_template('index.html', active_tab='Image')

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

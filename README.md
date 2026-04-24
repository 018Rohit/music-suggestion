# Music Sentiment Analysis Web App

This is a machine learning project that analyzes song lyrics to classify emotions such as positive, negative, or neutral. It uses Natural Language Processing (NLP) techniques and predictive models to understand the mood conveyed in the lyrics. Additionally, it can detect facial expressions from uploaded images and suggest songs based on the detected emotion.

## Features

- **Lyrics Input**: Users can paste song lyrics into a text area.
- **Text Preprocessing**: Applies NLP preprocessing including removing punctuation, digits, stopwords, lemmatization, and stemming.
- **Sentiment Analysis**: Uses TextBlob to calculate polarity and subjectivity, then classifies the sentiment.
- **Facial Expression Detection**: Upload an image to detect emotions using computer vision (FER library).
- **Song Suggestions**: Based on detected emotion, suggests songs that match the mood.
- **Results Display**: Shows sentiment classification, polarity score, subjectivity score, original lyrics, processed text, emotion scores, and song recommendations.

## Technologies Used

- **Backend**: Flask (Python web framework)
- **NLP Libraries**: NLTK, TextBlob
- **Computer Vision**: OpenCV, FER (Facial Expression Recognition)
- **Frontend**: HTML, CSS, JavaScript (tabbed interface)
- **Machine Learning**: TextBlob for sentiment analysis, FER for emotion detection

## Installation

1. Clone or download the project.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Open your browser and go to `http://127.0.0.1:5000/`

## Usage

### Lyrics Analysis
1. Click on the "Lyrics Analysis" tab.
2. Paste song lyrics into the text area.
3. Click "Analyze Sentiment".
4. View the results including sentiment classification, polarity, subjectivity, and processed text.

### Facial Expression Analysis
1. Click on the "Facial Expression" tab.
2. Upload an image containing a face.
3. Click "Detect Emotion & Suggest Songs".
4. View the detected emotion, emotion scores, and suggested songs.

## Project Structure

- `app.py`: Main Flask application with preprocessing, sentiment analysis, emotion detection, and song suggestion functions.
- `templates/index.html`: HTML template for the web interface with tabs for lyrics and image analysis.
- `requirements.txt`: Python dependencies.
- `README.md`: This file.

## Song Suggestions

The app includes predefined song suggestions for each emotion:
- Happy: Upbeat and joyful songs
- Sad: Emotional and melancholic songs
- Angry: High-energy and intense songs
- Surprised: Unexpected and exciting songs
- Fear: Tense and suspenseful songs
- Disgust: Alternative and unconventional songs
- Neutral: Classic and balanced songs

## Future Enhancements

- Integrate audio feature analysis for complete music sentiment analysis.
- Train custom ML models on larger lyrics datasets for better accuracy.
- Add music recommendation APIs (e.g., Spotify) for dynamic suggestions.
- Implement user authentication and save analysis history.
- Deploy to a cloud platform for public access.
- Improve emotion detection with custom models.

## Dataset

The preprocessing techniques are adapted from a Kaggle notebook on Amazon music reviews sentiment analysis. Facial emotion detection uses the FER library with pre-trained models.

## License

This project is for educational purposes. Please check licenses of used libraries.
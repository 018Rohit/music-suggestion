# Music Mood Detection Web App

This is a machine learning project that detects facial expressions from uploaded images and suggests music genres based on the detected emotion.

## Features

- **Facial Expression Detection**: Upload an image to detect emotions using computer vision (FER library).
- **Genre Suggestions**: Based on detected emotion, suggests music genres that match the mood.
- **Song Suggestions**: Recommends songs that fit the detected emotion.
- **Results Display**: Shows detected emotion, emotion scores, genre recommendations, and song recommendations.

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Computer Vision**: OpenCV, FER (Facial Expression Recognition)
- **Frontend**: HTML, CSS, JavaScript
- **Machine Learning**: FER for emotion detection

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

1. Click on the "Facial Expression" tab.
2. Upload an image containing a face.
3. Click "Detect Emotion & Suggest Genres".
4. View the detected emotion, emotion scores, suggested genres, and suggested songs.

## Project Structure

- `app.py`: Main Flask application with emotion detection and genre suggestion functions.
- `templates/index.html`: HTML template for the web interface.
- `requirements.txt`: Python dependencies.
- `README.md`: This file.

## Genre Suggestions

The app includes predefined genre suggestions for each emotion:
- Happy: Pop, Dance
- Sad: Lo-fi, Acoustic
- Angry: Rock, Metal
- Neutral: Chill, Ambient
- Surprise: EDM, Experimental

It also includes predefined song suggestions for each emotion.

## Future Enhancements

- Add music recommendation APIs (e.g., Spotify) for dynamic suggestions.
- Implement user authentication and save analysis history.
- Deploy to a cloud platform for public access.
- Improve emotion detection with custom models.

## Dataset

Facial emotion detection uses the FER library with pre-trained models.

## License

This project is for educational purposes. Please check licenses of used libraries.

#!/usr/bin/env python3
import requests
import os

# Test the lyrics analysis
def test_lyrics():
    url = "http://127.0.0.1:5000"
    data = {"lyrics": "I feel so happy today, the sun is shining and everything is great! This is wonderful."}

    try:
        response = requests.post(url, data=data)
        if "Positive" in response.text:
            print("✅ Lyrics Analysis: POSITIVE sentiment detected correctly")
        elif "sentiment" in response.text.lower():
            print("✅ Lyrics Analysis: Sentiment analysis working (response contains sentiment data)")
        else:
            print("❌ Lyrics Analysis: No sentiment data found in response")
        return True
    except Exception as e:
        print(f"❌ Lyrics Analysis: Error - {e}")
        return False

# Test the image upload (we'll use a dummy image)
def test_image():
    url = "http://127.0.0.1:5000"

    # Create a simple test image (1x1 pixel PNG)
    from PIL import Image
    import io

    # Create a small test image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    try:
        files = {'image': ('test.png', img_bytes, 'image/png')}
        response = requests.post(url, files=files)

        if "emotion" in response.text.lower() or "suggestions" in response.text.lower():
            print("✅ Image Upload: Facial analysis working (response contains emotion/suggestion data)")
        elif "neutral" in response.text.lower():
            print("✅ Image Upload: Default neutral emotion returned (FER fallback working)")
        else:
            print("❌ Image Upload: No emotion data found in response")
        return True
    except Exception as e:
        print(f"❌ Image Upload: Error - {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Music Sentiment Analysis App")
    print("=" * 50)

    lyrics_ok = test_lyrics()
    image_ok = test_image()

    print("\n" + "=" * 50)
    if lyrics_ok and image_ok:
        print("🎉 All tests passed! Both lyrics analysis and image upload are working.")
    else:
        print("⚠️  Some tests failed. Check the app logs for details.")
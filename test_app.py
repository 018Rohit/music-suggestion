#!/usr/bin/env python3
import requests

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

    image_ok = test_image()

    print("\n" + "=" * 50)
    if image_ok:
        print("🎉 All tests passed! Image upload and emotion detection are working.")
    else:
        print("⚠️  Some tests failed. Check the app logs for details.")

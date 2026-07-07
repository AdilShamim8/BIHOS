import requests

endpoints = ['skin', 'dental', 'nail', 'eye', 'oral']
url = 'http://127.0.0.1:5000/predict/'

# Create a dummy image
from PIL import Image
import io

img = Image.new('RGB', (224, 224), color='red')
buf = io.BytesIO()
img.save(buf, format='JPEG')
img_bytes = buf.getvalue()

for ep in endpoints:
    print(f"Testing {ep}...")
    res = requests.post(url + ep, files={'image': ('test.jpg', img_bytes, 'image/jpeg')})
    if res.status_code == 200:
        print(f"SUCCESS: {ep}")
    else:
        print(f"ERROR: {ep} - {res.status_code} - {res.text}")

import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'age':5, 'sex':200, 'bmi':400})

print(r.json())

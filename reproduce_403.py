import requests

url = 'https://data.pyg.org/whl/torch-2.12.0+cpu.html'
headers = {'User-Agent': 'Mozilla/5.0'}
resp = requests.get(url, headers=headers)
print('Status:', resp.status_code)
print('Content-Type:', resp.headers.get('Content-Type'))
print(resp.text[:300])

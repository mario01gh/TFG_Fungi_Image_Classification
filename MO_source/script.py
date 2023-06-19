import pandas as pd
import os
import requests

# Write the name of the species you want to download images
search_name = "Morchella esculenta"

headers = {
            'Connection': 'keep-alive',
            'Expect': '100-continue',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
            'TE': 'compress, deflate, gzip, trailers',
            'Upgrade-Insecure-Requests': '1',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 12_0_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15',
            'Sec-Fetch-User': '?1',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-Mode': 'navigate',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9'
        }
s = requests.Session()
s.headers = headers

file_name = search_name.replace(" ", "_")

here = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(here, 'MO_data.csv')

df = pd.read_csv(data_file)

selection_df = df.loc[df["name"] == search_name]
my_list = selection_df.image.tolist()
print(len(my_list))

path = os.path.join(here, file_name)
os.mkdir(path)
errores = []
for i in range(len(my_list)):
    image_name = file_name + "_" + str(i) + ".jpg"
    image_path = os.path.join(path, image_name)
    try:
        img_data = s.get(my_list[i], timeout=0.5).content
        with open(image_path, 'wb') as handler:
            handler.write(img_data)
    except (KeyError, requests.exceptions.ConnectionError, requests.exceptions.JSONDecodeError, requests.exceptions.Timeout):
        errores.append(i)
print(errores)
from bs4 import BeautifulSoup

def extract_urls(url):
    '''Extract image url's from given page'''
 
    r = requests.get(url)
    html_doc = r.text
    urls = []
    soup = BeautifulSoup(html_doc, 'html.parser')
    for img in soup.find_all('img'):
        url = img.get('src')
        if url and url.startswith('http'):
            urls.append(url)
    return urls     

import requests
from bs4 import BeautifulSoup
import time
import numpy
import pandas as pd 

def emojiDecode(description):
    output = (description
      .encode('utf-16', 'surrogatepass')
      .decode('utf-16')
      .encode("raw_unicode_escape")
      .decode("latin_1")
    )
    return output

def get_page_repo_list(URL,page):
    page = str(page)
    url_to_call = URL.format(page)

    response = requests.get(url_to_call, headers = {'User-Agent': "Mozilla/5.0"})
    if response.status_code != 200:
        print('connection failed')
        return
    html_content = response.content    
    dom = BeautifulSoup(html_content, 'html.parser')
    repoList = dom.select("ul.repo-list li")
    return repoList



def is_int(temp):
    try:
        int(temp)
        return 1
    except:
        return 0

def is_Star(temp):
    if "." in temp or is_int(temp):
        return 1
    else:
        return 0

    
def is_lang(temp):
    langs = ['Jupyter Notebook','Python','R','HTML','JavaScript','Java','CSS','Rebol','TeX','Shell']
    return temp in langs

def is_license(temp):
    return "Updated" not in temp

def get_elements(section):
    al = section.find_all("div", {"class" : "mr-3"})
    if len(al) == 4:
        return [div.get_text().replace("\n", "").strip() for div in al]
    result = [0,0,0,0]
    for div in al:
        temp = div.get_text().replace("\n", "").strip()
        if is_Star(temp):
            result[0] = temp
            pass

        elif is_lang(temp):
            result[1] = temp
            pass

        elif is_license(temp):
            result[2] = temp
            pass
        else:
            result[3] = temp

    return result

def clean_stars(stars):
    stars = str(stars)
    if '.' in stars:
        stars = stars.replace(".", "").replace("k", "00")
    stars = stars.replace("k", "000")
    return stars

def clean_elements(elements):

    elements[0] = clean_stars(elements[0])

    elements[3] = elements[3].replace("Updated", "").strip()
    return elements

def extractData(repo):
    name = repo.a.attrs["href"]
    try:
        description = repo.p.get_text().replace("\n", "").strip()
    except:
        description = "NA"
    description = emojiDecode(description)
    section = repo.find("div", {"class" : ""})
    features = get_elements(section)
    features =  clean_elements(features)
    features.insert(0,name)
    features.insert(1,description)
    return features



def get_URLs():
    f = open("tempURLS.txt", "r")
    listURLs = [x for x in f]
    return listURLs

def save_data(data,URLCOUNT,count):
    a = numpy.asarray(data)
    cols = ["URLCODE","Name", "Descr","Stars","MajLang","License","LastUpdated"]
    pd.DataFrame(a).to_csv("GitHub_DS_Repos{}-{}.csv".format(URLCOUNT,count), header=cols)

    
def driver():
    URLCOUNT = 1
    
    for URL in get_URLs():
        data = []
        count = 0
        for page in range(1,101):
            time.sleep(20)
            repoList = get_page_repo_list(URL,page)
            for repo in repoList:
                temp = extractData(repo)
                temp.insert(0,URLCOUNT)
                data.append(temp)
                
            print(URLCOUNT,page)
            if page % 5 == 0:
                count += 1
                save_data(data,URLCOUNT,count)
        URLCOUNT += 1

if __name__ == "__main__":
    driver()

    

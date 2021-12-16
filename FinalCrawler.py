import requests
from bs4 import BeautifulSoup
import time
import numpy
import pandas as pd 
import helpers

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

def get_elements(section):
    al = section.find_all("div", {"class" : "mr-3"})
    if len(al) == 4:
        return [div.get_text().replace("\n", "").strip() for div in al]
    result = [0,0,0,0]
    for div in al:
        temp = div.get_text().replace("\n", "").strip()
        if helpers.is_Star(temp):
            result[0] = temp
            pass

        elif helpers.is_lang(temp):
            result[1] = temp
            pass

        elif helpers.is_license(temp):
            result[2] = temp
            pass
        else:
            result[3] = temp

    return result

def get_tags(repo):
    tags = [x.get('title').replace("Topic: ","") for x in repo.find_all("a", {'class' : 'topic-tag topic-tag-link f6 px-2 mx-0'})]
    return tags
    
    return final_taglist
def extractData(repo,URLCOUNT):
    name = repo.a.attrs["href"]
    try:
        description = repo.p.get_text().replace("\n", "").strip()
    except:
        description = "NA"
    description = helpers.emojiDecode(description)
    tags = get_tags(repo)
    section = repo.find("div", {"class" : ""})
    features = get_elements(section)
    features =  helpers.clean_elements(features)
    features.insert(0,name)
    features.insert(1,description)
    features.insert(2,tags)
    features.insert(0,URLCOUNT)
    return features

def get_complete_page_data(URL,page):   
    return [extractData(repo,URLCOUNT) for repo in get_page_repo_list(URL,page)]

if __name__ == "__main__":
    URLCOUNT = 0 #global var
    max_pages = 101 #extra one for range()
    for URL in helpers.get_URLs():
        data = []
        count = 0
        for page in range(1,max_pages):
            time.sleep(20)
            page_data = get_complete_page_data(URL,page)
            data.append(page_data)
            print(URLCOUNT,page)
            if page % 5 == 0:
                count += 1
                helpers.save_data(data,URLCOUNT,count)    
        URLCOUNT += 1

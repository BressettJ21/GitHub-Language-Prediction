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

def is_int(temp):
    try:
        int(temp)
        return 1
    except:
        return 0

def is_Star(temp):
    n = len(temp)
    if (((n<=5) and "." in temp) or is_int(temp.replace("k", ""))):
        return 1
    else:
        return 0

def is_lang(temp):
    langs = ['Jupyter Notebook','Python','R','HTML','JavaScript','Java','CSS','Rebol','TeX','Shell']
    return temp in langs

def is_license(temp):
    return "Updated" not in temp

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

def get_URLs():
    f = open("tempURLS.txt", "r")
    listURLs = [x for x in f]
    return listURLs

def save_data(data,URLCOUNT,count):
    a = numpy.asarray(data)
    cols = ["URLCODE","Name", "Descr","tags","Stars","MajLang","License","LastUpdated"]
    length = len(cols)
    a = a.reshape(-1, a.shape[-1])
    pd.DataFrame(a).to_csv("GitHub_DS_Repos{}-{}.csv".format(URLCOUNT,count), header=cols)



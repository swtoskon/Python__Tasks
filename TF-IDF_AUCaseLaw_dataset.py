import io
import glob
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from pprint import pprint
import nltk
from lxml.html import fromstring, tostring
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from gensim.utils import simple_preprocess
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn_extra.cluster import KMedoids
# the starting dataset had erroneous xml
# ie "id=c01" instead of id="c01" (on catchphrases)
# the solution is generalized


def fix_xml_id_error(file_location: Path) -> str:
    """Input file path, and fix a dataset problem where id is shown as ie "id=1" instead of id="1", as in xml standard
        Args:
            file_location: A relative or absolute path to the file to be fixed
        Returns:
            A string with the corrected lines on it
    """
    with open(file_location, "r",encoding='windows-1252') as file_r:
        lines = file_r.read()
    lines = lines.replace('"id=', 'id="')
    return lines


# changes the encoding to unicode and compiles the html entities
# so it can work with special characters/entities   
def csv_unicode_html_clean(text_lines: str) -> str:
    """Take the lines as a string and pass it from html reader. This is to remove html elements that the dataset
    used, returning the text as a unicode string
        Args:
            text_lines: The file written as a string, clean, but possibility of html entities
        Returns:
            A unicode coded string, clean from html elements
    """
    div = fromstring(text_lines)
    clean = tostring(div, encoding='unicode')
    return clean


# removes automatically generated html paragraph tag
# from fromstring - tostring solution
def remove_html_paragraph(text_lines) -> str:
    """Input the return of csv_unicode_html_clean and remove the automated <p></p> that appears on the output after
    returning it as a string
    Args:
        text_lines: A relative or absolute path to the file to be fixed
    Returns:
        A string with the corrected lines on it
        """
    remove_first = re.sub('^<p>', '', text_lines)
    removed_all = re.sub('</p>$', '', remove_first)
    remove_=re.sub('\n','',removed_all)
    return remove_


def xml_elem(xml_string: str, tag: str) -> str:
    """Select tag of the xml element to return its text
           Args:
               xml_string: The string from the file, clean
               tag: The selected tag on the xml string
           Returns:
               All of the text that the tag contains
       """
    tree = ET.ElementTree(ET.fromstring(xml_string))
    full_sentence_text = ""
    for elem in tree.iter(tag=tag):
        full_sentence_text += elem.text
    return full_sentence_text


def xml_file_clean(file_location) -> dict:
    """A full clean up using the created functions, for the syntax error and the html elements
        Args:
            file_location: The location of a single file in the corpus
        Returns:
            A dictionary with the name tag of the file and the sentence tag, which is the full text of the document
    """

    # returns fixed xml error with ids (string)
    read_file = fix_xml_id_error(file_location)
    # returns lines clear from html entities
    new_lines = csv_unicode_html_clean(read_file)
    # returns the text without the html paragraph automatically generated
    clean_text = remove_html_paragraph(new_lines)
    case_details = {'name': xml_elem(clean_text, "name"),
                    'text': xml_elem(clean_text, "sentence")}
    return case_details


# will take folder as relative or absolute path ending on /
def full_dataset_unicode(folder_location) -> list:
    """The conversion of the full corpus into a dataset list
            Args:
                folder_location: The location of the corpus folder, as relative or absolute path ending on /
            Returns:
                The cleaned files as a dictionary list, creating a dataset
        """
    count = 0
    case_detail_list = []
    # glob scans the folder for all available files,
    for file in glob.glob(str(folder_location / "*")):
        print(file)
        # generates the clean xml string from the file location
        case_detail = xml_file_clean(file)
        case_detail_list.append(case_detail)
        count += 1
    print("Number of files: ", count)
    return case_detail_list

def tokenizeContent(contentsRaw):
    tokenized = nltk.tokenize.word_tokenize(contentsRaw)
    return tokenized

def removeStopWordsFromTokenized(contentsTokenized):
    stop_word_set = set(nltk.corpus.stopwords.words("english"))
    filteredContents = [word for word in contentsTokenized if word not in stop_word_set]
    return filteredContents

def convertItemsToLower(contentsRaw):
    filteredContents = [term.lower() for term in contentsRaw]
    return filteredContents

def removePunctuationFromTokenized(contentsTokenized):
    excludePuncuation = set(string.punctuation)
    filteredContents = [word for word in contentsTokenized if word not in excludePuncuation]
    return filteredContents

def performPorterStemmingOnContents(contentsTokenized):
    porterStemmer = nltk.stem.PorterStemmer()
    filteredContents = [porterStemmer.stem(word) for word in contentsTokenized]
    return filteredContents

def processData(rawContents):
    cleaned = tokenizeContent(rawContents)
    cleaned = removeStopWordsFromTokenized(cleaned)
    cleaned = performPorterStemmingOnContents(cleaned)    
    cleaned = removePunctuationFromTokenized(cleaned)
    cleaned = convertItemsToLower(cleaned)
    return cleaned

def returnListOfFilePaths(folderPath):
    listOfFilePaths = [join(folderPath, fileName) for fileName in listdir(folderPath) if isfile(join(folderPath, fileName))]
    return listOfFilePaths

def create_docContentDict(filePaths):
    rawContentDict = {}
    for filePath in filePaths:
        with open(filePath, "r") as ifile:
            fileContent = ifile.read()
        rawContentDict[filePath] = fileContent
    return rawContentDict



BASE_INPUT_DIR = "/Users/swtoskon/Downloads/corpus/fulltext"
fileNames = returnListOfFilePaths(BASE_INPUT_DIR)

lst=[]
count=0
for file in fileNames:
    case_detail = xml_file_clean(file)
    doc = case_detail['text']
    lst.append(doc)
    count+=1
    
tfidf = TfidfVectorizer(tokenizer=processData,stop_words='english')
tfs = tfidf.fit_transform(lst)
#feature_names = tfidf.get_feature_names()
#dense = tfs.todense()
#denselist = dense.tolist()
#df = pd.DataFrame(denselist, columns=feature_names)
print(count)

similarity_list=[]

#for i in range(len(fileNames)):
 #        matrixValue = cosine_similarity(tfs[3795], tfs[i])
  #       numValue = matrixValue[0][0]
   #      similarity_list.append([numValue,i,fileNames[i]])
#calculate cosine similarity
for i in range(1500):
    emb=tfs[i]
    if(i==1000 or i==200):
        print(i)
    for j in range(1500):
        if(i!=j and i>j):
            emb1=tfs[j]
            csim= cosine_similarity(emb.reshape(1,-1), emb1.reshape(1,-1))
            csim1=csim[0][0]
            similarity_list.append([csim1,fileNames[i],fileNames[j]])        
#print("second")
similarity_df = pd.DataFrame(similarity_list, columns=['cos_sim', 'document1', 'document2'])
similarity_df.to_csv("tfidf.csv")
print(len(similarity_df))

df = pd.read_csv('tfidf.csv')
df1= pd.read_csv('doc2vec1.csv')
df2=pd.read_csv('bert.csv')

#kendall corellation with 3 methods
print(df['cos_sim'].corr(df1['cos_sim'],method='kendall'))
print(df['cos_sim'].corr(df2['cos_sim'],method='kendall'))
print(df1['cos_sim'].corr(df2['cos_sim'],method='kendall'))



#sf=similarity_df.sort_values(by=['cos_sim'],ascending=False)
#print(sf[:10])

#print(sf[1945:1947])

#print(sf.tail(10))








    


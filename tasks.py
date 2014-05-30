# encoding: utf-8
from __future__ import unicode_literals

'''\n
celery example
-

the following tools need to be installed to run this example:
- celery
- redis
- nathan-py
- numpy
- nltk

to start a worker with 10 parallel instances:
$ celery -A tasks worker --concurrency=10 --loglevel=info

to run a command on the queue:
>>> import tasks
>>> tasks.keywords_for_url.delay('http://www.ai-one.com').get(timeout=10)

this module could now be used to create a webservice etc.
'''

import nltk
import re
import hashlib
from celery import Celery
from bs4 import BeautifulSoup
from nathan.core import Dataspace
from urllib.request import urlopen
from itertools import chain, islice
from collections import defaultdict

app = Celery('tasks', backend='redis://localhost',
    broker='amqp://guest@localhost//')


_dataspace = Dataspace("./brown.hnn")
_corpus = "brown"

def decode(page):
    data = page.read()
    
    # TODO: handle compressed files
    #encoding = page.info().get("Content-Encoding") 
    #if encoding in ('gzip', 'x-gzip', 'deflate'):
    #    if encoding == 'deflate':
    #        data = StringIO.StringIO(zlib.decompress(data))
    #    else:
    #        data = gzip.GzipFile('', 'rb', 9, StringIO.StringIO(data))
        
    charset = 'utf-8'
    match = re.findall('charset=(.+)', page.info().get('Content-Type', ''))
    if (len(match) > 0):
        charset = match[0]
    
    data = data.decode(charset)

    return data

def _fetch_url(url):
    #html = urlopen(url).read()
    with urlopen(url) as page:
        html = decode(page)
    soup = BeautifulSoup(html)

    # kill all script and style elements
    for script in soup(['script', 'style']):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split('  '))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

# brown.tagged_sents(tagset='universal')
    
def _tokenize(text, preserve_entities=True):
    
    def ne_concat(node, result):
        if isinstance(node, nltk.Tree):
            if node.label() == 'NE':
                result.append(' '.join(word for word, tag in node))
            else:
                for child in node:
                    ne_concat(child, result)
        else:
            word, tag = node
            result.append(word)
    
    def word_tokenize(sent):
        if preserve_entities:
            tagged = nltk.pos_tag(nltk.word_tokenize(sent))
            chunks = nltk.ne_chunk(tagged, binary=True)
            words = []
            ne_concat(chunks, words)
            return words
        else:
            return nltk.word_tokenize(sent)
            
    return [word_tokenize(sent) for sent in nltk.sent_tokenize(text)]

def _normalize(words):
    return [w.lower() for w in words]

@app.task
def keywords_from_url(url, **kwargs):
    text = _fetch_url(url)
    return keywords_from_text(text, **kwargs)

"""
 TODO: ACTHUNG: crash when running http://192.168.2.3:5000/fingerprint/http://www.google.com
 
 => logfile rausschreiben und Manfred schicken
 
### KEYWORD: google
### KEYWORD: suche
### KEYWORD: bilder
### KEYWORD: maps
### KEYWORD: play
### KEYWORD: youtube
### KEYWORD: news
### KEYWORD: gmail
### KEYWORD: drive
### KEYWORD: mehr
### KEYWORD: »webprotokoll
### KEYWORD: einstellungen
### KEYWORD: anmelden
### KEYWORD: schweiz
### KEYWORD: erweiterte
### KEYWORD: suchesprachoptionengoogle.ch
*** glibc detected *** /home/tdiggelm/nathan-py/py3.4/bin/python: free(): invalid next size (fast): 0x00007f92c4030e10 ***
======= Backtrace: =========
/lib/x86_64-linux-gnu/libc.so.6(+0x7eb96)[0x7f92eaf56b96]
"""

def print_args():
    def decorator(function):
        def inner(*args, **kwargs):
            print(args, kwargs)
            return function(*args, **kwargs)
        return inner
    return decorator

@app.task
@print_args()
def keywords_for_query(
    query, 
    corpus="brown",
    limit=20,
    associations_per_keyword=3,
    preserve_entities=True):

    is_url = ('(?i)\\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+'
        '[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+'
        '|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)'
        '|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))')
    
    text = query
    
    # find urls
    matches = re.findall(is_url, text)
    
    # currently take a maximum of 5 urls
    matches = matches[:5]
    
    for match in matches:
        url = "".join(match)
        url_fetch = url
        if not url.startswith("http"):
            url_fetch = "http://" + url
        # replace url with actual content
        try:
            text = text.replace(url, _fetch_url(url_fetch))
        except BaseException as ex:
            print(ex)
            pass
            
    #print(matches, text)
        
    global _corpus
    global _dataspace
    if (_corpus != corpus):
        if corpus == "none":
            _dataspace = Dataspace()
        elif corpus == "brown":
            _dataspace = Dataspace("./brown.hnn")
        elif corpus == "reuters-flat":
            _dataspace = Dataspace("./reuters-flat.hnn")
        elif corpus == "reuters-deep":
            _dataspace = Dataspace("./reuters-deep.hnn")
        else:
            raise ValueError("invalid corpus %s" % corpus)
        _corpus = corpus
    
    # TODO: cache result of with text_hash as key
    text_hash = hashlib.sha1(text.encode('utf-8')).hexdigest()
    sents = _tokenize(text, preserve_entities=preserve_entities)
    
    # insert the document
    root_handle = _dataspace.insert(text_hash)
    for sent in sents:
        sent = _normalize(sent) # normalize words before inserting into ds
        sent_handle = _dataspace.insert(sent)
        _dataspace.link(root_handle, sent_handle)
    
    # calculate the keywords
    result = []
    for keyword, _, _ in _dataspace.keywords_of(root_handle):
        if len(keyword) < 2: 
            continue
        elif len(result) >= limit:
            break
            
        if associations_per_keyword == 0:
            result.append((keyword,[]))
        else:
            assos = _dataspace.associate(keyword)
            filtered = (quant for quant, _, _ in assos if (quant != keyword 
                and len(quant) >= 2))
            item = (keyword, list(islice(filtered, associations_per_keyword)))
            result.append(item)
    
    # delete document
    for sent_handle in _dataspace.children_of(root_handle):
        _dataspace.delete(sent_handle)
    _dataspace.delete(root_handle)
    
    return result
        
    

@app.task
def keywords_from_text(text, 
    corpus="brown", limit=20, associations_per_keyword=3,
    preserve_entities=True):
    
    global _corpus
    global _dataspace
    if (_corpus != corpus):
        if corpus == "none":
            _dataspace = Dataspace()
        elif corpus == "brown":
            _dataspace = Dataspace("./brown.hnn")
        elif corpus == "reuters-flat":
            _dataspace = Dataspace("./reuters-flat.hnn")
        elif corpus == "reuters-deep":
            _dataspace = Dataspace("./reuters-deep.hnn")
        else:
            raise ValueError("invalid corpus %s" % corpus)
        _corpus = corpus
    
    # TODO: cache result of with text_hash as key
    text_hash = hashlib.sha1(text.encode('utf-8')).hexdigest()
    sents = _tokenize(text, preserve_entities=preserve_entities)
    
    # insert the document
    root_handle = _dataspace.insert(text_hash)
    for sent in sents:
        sent = _normalize(sent) # normalize words before inserting into ds
        sent_handle = _dataspace.insert(sent)
        _dataspace.link(root_handle, sent_handle)
    

    # calculate the keywords
    result = []
    for keyword, _, _ in _dataspace.keywords_of(root_handle):
        if len(keyword) < 2: 
            continue
        elif len(result) >= limit:
            break
            
        if associations_per_keyword == 0:
            result.append((keyword,[]))
        else:
            assos = _dataspace.associate(keyword)
            filtered = (quant for quant, _, _ in assos if (quant != keyword 
                and len(quant) >= 2))
            item = (keyword, list(islice(filtered, associations_per_keyword)))
            result.append(item)
    
    # delete document
    for sent_handle in _dataspace.children_of(root_handle):
        _dataspace.delete(sent_handle)
    _dataspace.delete(root_handle)
    
    return result
    
from flask import Flask
from flask import render_template
from flask import jsonify
from flask import request
from werkzeug.exceptions import BadRequest
import hashlib
from itertools import chain, islice
from collections import OrderedDict
from tasks import Filter

app = Flask(__name__)

from tasks import keywords_from_url, keywords_from_text, keywords_for_query, bing_find

def str_to_bool(string):
    return string is not None and (string == '1' or string.lower() == 'true')

@app.route('/')
def index(url=None):
    return render_template('hello.html', url=url)

@app.route('/test')
def test():
    return render_template('test.html')

def format_result(fp):
    html = ''
    for keyword, tag, score in fp:
        html += '<li style="position: relative"><div style="position: absolute; top: 2px; font-size: 11px; font-weight: 100; right:5px; color: #d33682">%s, %s</div><label class="checkbox"><input type="checkbox" checked="checked" value="%s" />%s</label></li>'%(tag, score, keyword, keyword)
    return html
    
def format_result_json(fp):
    return jsonify(keywords=fp)

@app.route('/fingerprint/', defaults={'url': ''}, methods=['GET', 'POST'])
@app.route('/fingerprint/<path:url>', methods=['GET', 'POST'])
def fingerprint(url=None):
    if request.method == 'POST':
        args = request.json
    else:
        args = dict(request.args.items())
        args["preserve_entities"] = str_to_bool(
            args.get("preserve_entities", "false"))
        args["corpus"] = args.pop("corpus", "brown").lower().strip()
        args["limit"] = int(args.pop("limit", "10"))

    #if len(args) > 0:
    #    raise TypeError("unsupported parameters in querystring")
    
    if request.method == 'POST':
        fp = keywords_from_text.delay(associations_per_keyword=0, **args).get(timeout=120)
    else:
        fp = keywords_from_url.delay(url, associations_per_keyword=0, **args).get(timeout=120)

    return format_result(fp)
    
class LimitedSizeDict(OrderedDict):
  def __init__(self, *args, **kwds):
    self.size_limit = kwds.pop("size_limit", None)
    OrderedDict.__init__(self, *args, **kwds)
    self._check_size_limit()

  def __setitem__(self, key, value):
    OrderedDict.__setitem__(self, key, value)
    self._check_size_limit()

  def _check_size_limit(self):
    if self.size_limit is not None:
      while len(self) > self.size_limit:
        self.popitem(last=False)

fingerprint_cache = LimitedSizeDict(size_limit=10)
keywords_cache = LimitedSizeDict(size_limit=10)

import re
def icasereplace(string, find, replace):
    p = re.compile('\\b' + re.escape(find) + '\\b', re.IGNORECASE)
    return p.sub(replace, string)

import pickle
def hash_args(args):
    dump = pickle.dumps(args)
    return hashlib.sha1(dump).hexdigest()

@app.route('/search', methods=['POST'])
def search():
    """
    TODO: retrieve keywords and return scored result list (also fingerprint_cache it)
    """
    args = dict(request.json)
    sort = args.pop('sorted', True)
    
    print("SEARCH ARGS: %s" % args)
    
    hash_id = hash_args(args)
    result = keywords_cache.get(hash_id, None)
    if result is None:
        result = bing_find.delay(**args).get(timeout=120)
        keywords_cache[hash_id] = result
    
    if sort:
        result = sorted(result, key=lambda i: i['score'], reverse=True)
    
    keywords = [k[0] for k in args["keywords"]]
    
    html = ''
    if not result is None:
        for item in result:
            desc = item['desc']
            for keyw in keywords:
                desc = icasereplace(desc, keyw, '<strong>%s</strong>' % keyw)
                
            html = html + '<div>'
            html = html + '<div><a href="%s">%s</a></div>' % (item['url'],  
                item['title'])
            html = html + '<p>score: %s</p>' % item.get('score', 'n/a')
            html = html + '<p>position: %s</p>' % item.get('pos', 'n/a')
            html = html + '<p>%s</p>' % desc
            html = html + '</div>'
    return html
    
@app.route('/fingerprint2', methods=['POST'])
def fingerprint2(url=None):
    args = dict(request.json)
    
    hash_key = hashlib.sha1((
            args['query'] + str(args['preserve_entities']) 
            + str(args['fetch_urls']) + str(args['corpus'] 
            + str(args['analyse_pos']))
        ).encode('utf-8')).hexdigest()
    
    limit = args.pop('limit', 10)
    accepted_tags = args.pop('accepted_tags', None)
    reject_numbers = args.pop('reject_numbers', True);
    reject_punctuation = args.pop('reject_punctuation', True);
    reject_stopwords = args.pop('reject_stopwords', True)
    
    # set to maximum limit
    args['limit'] = 500
    
    keywords = fingerprint_cache.get(hash_key, None)
    if keywords is None:
        keywords = keywords_for_query.delay(**args).get(timeout=120)
        fingerprint_cache[hash_key] = keywords
        
    filtered = filter(Filter(reject_numbers, accepted_tags, reject_stopwords, reject_punctuation), keywords)
    result = list(islice(filtered, limit))
    
    return format_result(result)
    
@app.route('/fingerprint2.json', methods=['POST'])
def fingerprint2_json(url=None):
    job = request.json
    fp = keywords_for_query.delay(**job).get(timeout=120)
    return format_result_json(fp)

from flask import Response

_status_count = 5

@app.route("/fp", methods=['POST'])
def fp_test():
    global _status_count
    req_id = 1
    if _status_count == 0:
        resp = Response("created", status=201, mimetype='text/plain',
            headers={"Location": "/fp/%s" % req_id})
    else:
        resp = Response("accepted", status=202, mimetype='text/plain',
            headers={"Location": "/status/%s" % req_id})
    return resp
    
@app.route("/status/<req_id>", methods=['GET'])
def fp_status(req_id):
    global _status_count
    if _status_count > 0:
        json = jsonify(status="processing", message="processing %s -> %s ..." % (req_id, _status_count));
        _status_count -= 1
        return json
    else:
        return jsonify(status="done", url="/fp/%s" % req_id)
    return resp
    
@app.route("/fp/<req_id>", methods=['GET', 'DELETE'])
def fp_result(req_id):
    global _status_count
    
    if _status_count != 0:
        return Response("%s not found" % req_id, status=404, 
            mimetype='text/plain')
    
    if request.method == 'GET':
        return Response("result %s" % req_id, status=200, 
            mimetype='text/plain')
    elif request.method == 'DELETE':
        _status_count = 5
        return "done"
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
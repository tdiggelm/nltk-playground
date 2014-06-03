from flask import Flask
from flask import render_template
from flask import jsonify
from flask import request
from werkzeug.exceptions import BadRequest
import hashlib
from itertools import chain, islice
from collections import OrderedDict

app = Flask(__name__)

from tasks import keywords_from_url, keywords_from_text, keywords_for_query

def str_to_bool(string):
    return string is not None and (string == '1' or string.lower() == 'true')

@app.route('/')
def index(url=None):
    return render_template('hello.html', url=url)
    
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

class Filter:    
    def __init__(self, reject_numbers, accepted_tags):
        self.reject_numbers = reject_numbers
        self.accepted_tags = accepted_tags
    
    def __call__(self, item):
        def isnumeric(s):
            try:
                float(s)
                return True
            except:
                return False
        
        word, tag, score = item
        
        if isnumeric(word):
            if self.reject_numbers:
                return False
            else:
                return True
               
        if not self.accepted_tags is None and tag not in self.accepted_tags:
            return False
            
        return True

cache = LimitedSizeDict(size_limit=10)
    
@app.route('/fingerprint2', methods=['POST'])
def fingerprint2(url=None):
    args = dict(request.json)
    
    hash_key = hashlib.sha1((
            args['query'] + str(args['preserve_entities']) 
            + str(args['fetch_urls']) + str(args['corpus'])
        ).encode('utf-8')).hexdigest()
    
    limit = args.pop('limit', 10)
    accepted_tags = args.pop('accepted_tags', None)
    reject_numbers = args.pop('reject_numbers', True);
    
    # set to maximum limit
    args['limit'] = 500
    
    keywords = cache.get(hash_key, None)
    if keywords is None:
        keywords = keywords_for_query.delay(**args).get(timeout=120)
        cache[hash_key] = keywords
    
    filtered = filter(Filter(reject_numbers, accepted_tags), keywords)
    result = list(islice(filtered, limit))
    
    return format_result(result)
    
@app.route('/fingerprint2.json', methods=['POST'])
def fingerprint2_json(url=None):
    job = request.json
    fp = keywords_for_query.delay(**job).get(timeout=120)
    return format_result_json(fp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
from flask import Flask
from flask import render_template
from flask import jsonify
from flask import request
from werkzeug.exceptions import BadRequest
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
    
@app.route('/fingerprint2', methods=['POST'])
def fingerprint2(url=None):
    job = request.json
    fp = keywords_for_query.delay(**job).get(timeout=120)
    return format_result(fp)
    
@app.route('/fingerprint2.json', methods=['POST'])
def fingerprint2_json(url=None):
    job = request.json
    fp = keywords_for_query.delay(**job).get(timeout=120)
    return format_result_json(fp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
from flask import Flask
from flask import render_template
from flask import jsonify
from flask import request
from werkzeug.exceptions import BadRequest
app = Flask(__name__)

from tasks import keywords_from_url, keywords_from_text

def str_to_bool(string):
    return string is not None and (string == '1' or string.lower() == 'true')

@app.route('/')
def index(url=None):
    return render_template('hello.html', url=url)
    
def format_result(fp):
    html = ''
    for keyword, associations in fp:
        html += '<li><label class="checkbox"><input type="checkbox" checked="checked" value="%s" />%s</label></li>' % (keyword, keyword)
    return html

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
        fp = keywords_from_text(associations_per_keyword=0, **args)
    else:
        fp = keywords_from_url(url, associations_per_keyword=0, **args)

    return format_result(fp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
from flask import Flask
from flask import render_template
from flask import jsonify
from flask import request
app = Flask(__name__)

from tasks import keywords_from_url

def str_to_bool(string):
    return string is not None and (string == '1' or string.lower() == 'true')

@app.route('/')
def index(url=None):
    return render_template('hello.html', url=url)
    
def format_result(fp):
    html = ''
    for keyword, associations in fp:
        html += '<li><label class="checkbox"><input type="checkbox" value="%s" />%s</label></li>' % (keyword, keyword)
    return html

@app.route('/fingerprint/', defaults={'url': ''})
@app.route('/fingerprint/<path:url>')
def fingerprint(url=None):
    args = dict(request.args.items())
    preserve_entities = str_to_bool(args.pop("preserve_entities", "false"))
    corpus = args.pop("corpus", "brown").lower().strip()

    if len(args) > 0:
        raise TypeError("unsupported parameters in querystring")
    
    if request.method == 'POST':
        if request.headers.get('content-type').startswith('text/plain'):
            text = request.get_data().decode('utf-8')
            
            fp = keywords_from_text(text, 
                preserve_entities=preserve_entities, corpus=corpus, 
                associations_per_keyword=0)
            return format_result(fp)
        else:
            raise BadRequest('expected text/plain')
    else:
        fp = keywords_from_url(url, 
            preserve_entities=preserve_entities, corpus=corpus, 
            associations_per_keyword=0)
        return format_result(fp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
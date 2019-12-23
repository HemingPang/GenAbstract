from app import app, genAbstract, logger
from flask import render_template, request, jsonify, json, Response


@app.route('/')
@app.route('/index')
def index():
    summary_text = "test"
    return render_template('index.html', summary=summary_text)


@app.route('/summary', methods=['POST'])
def get_summary():
    data = json.loads(request.form.get('data'))
    title = data['title']
    content = data['content']
    if title is None or len(title) == 0 or content is None or len(content) == 0:
        return jsonify({'status': -1, 'text': '参数异常'});
    else:
        logger.debug('title:%s \n content:%s\n' % (title, content))
        text = genAbstract.summarize(content, title)
        return jsonify({'status': 'OK', 'text': text});


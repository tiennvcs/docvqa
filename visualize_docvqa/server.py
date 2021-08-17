from flask import Flask
from flask import render_template
from load_gt import load_gt
import os
import glob2


app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='static')

@app.route("/")
def hello_world():
    return render_template('home.html')

@app.route("/vis/<folder>", methods=['GET', 'POST'])
def display_images(folder):
    gt_path = glob2.glob(os.path.join('visualize_docvqa', 'static', folder, '*.json'))[0]
    gt = load_gt(path=gt_path)
    return render_template('vis.html', folder=folder, items=gt)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port='2345', debug=True)

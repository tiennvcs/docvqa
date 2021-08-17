from flask import Flask
from flask import render_template

app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='static')

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/show")
def display_images():
    return render_template('show.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='2345', debug=True)

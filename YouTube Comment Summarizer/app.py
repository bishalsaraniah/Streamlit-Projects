from flask import Flask, request, Response
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World"

if __name__ == "__main__":
    app.run(port=8000, debug=True)
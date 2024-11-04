from bottle import route, run
from classifier.predict import predict

@route('/<input_line>')
def index(input_line):
    return {
        'result': predict(input_line, 10)
    }

def serve():
    run(host='localhost', port=5533)
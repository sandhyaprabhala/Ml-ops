from flask import Flask
from werkzeug.utils import cached_property
from flask_restx import Resource, Api

app = Flask(__name__)
api = Api(app)

@api.route('/odd_or_even')
class HelloWorld(Resource):
    def get(self):      
        num = int(input("Enter a number: "))
        if (num % 2) == 0:
            return{"{0}".format(num): "Even"}
        else:
            return{"{0}".format(num): "Odd"}
if __name__ == '__main__':
    app.run(debug=True)
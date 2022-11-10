# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:38:11 2022

@author: A ARUN JOSEPHRAJ
"""
from flask import Flask, jsonify, request
from BERT import *

app = Flask(__name__)

# on the terminal type: curl http://127.0.0.1:5000/
# returns hello world when we use GET.
# returns the data that we send when we use POST.
@app.route('/', methods = ['GET', 'POST'])
def home():
    
    txt1 = request.args.get('text1')
    txt2 = request.args.get('text2')

    en_1 = model.encode(txt1)
    en_2 = model.encode(txt2)

    x = float(cosine_similarity([en_1],[en_2]))
    print(cosine_similarity([en_1],[en_2]))
    #return int(cosine_similarity([en_1],[en_2])[0])
    return {'similarity score': round(x,2)}


# A simple function to calculate the square of a number
# the number to be squared is sent in the URL when we use GET
# on the terminal type: curl http://127.0.0.1:5000 / home / 10
# this returns 100 (square of 10)
@app.route('/home/<int:num>', methods = ['GET'])
def disp(num):

	return jsonify({'data': num**2})


# driver function
if __name__ == '__main__':

	app.run()

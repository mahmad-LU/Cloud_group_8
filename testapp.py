# -*- coding: utf-8 -*-
"""
@author duytinvo
"""
import os
import argparse
import subprocess
import logging
from flask import Flask, request, jsonify, render_template, url_for, flash, redirect
from werkzeug.exceptions import abort
from flask_cors import CORS
from baselines import load
import signal
import datetime
import sys
import shlex

# argparser = argparse.ArgumentParser(sys.argv[0])
#
# argparser.add_argument('--train_file', help='Trained file', default="../data/smsspamcollection/train.csv", type=str)
#
# argparser.add_argument('--dev_file', help='Developed file', default="../data/smsspamcollection/dev.csv", type=str)
#
# argparser.add_argument('--test_file', help='Tested file', default="../data/smsspamcollection/dev.csv", type=str)
#
# argparser.add_argument("--tfidf", action='store_true', default=False, help="tfidf flag")
#
# argparser.add_argument("--use_hash", action='store_true', default=False, help="hashing flag")
#
# argparser.add_argument("--scaler", action='store_true', default=False, help="scale flag")
#
# argparser.add_argument('--ml_cls', help='Machine learning classifier', default="MLP", type=str)
#                                                                 # kNN LR DT SVM MLP AB GB RF NB
#                                                                 #DT P(1); SVM dontwork;
# argparser.add_argument('--model_dir', help='Model dir', default="../data/smsspamcollection/", type=str)
#
# args, unknown = argparser.parse_known_args()
#
# model_dir, _ = os.path.split(args.model_dir)
#
# if not os.path.exists(args.model_dir):
#     os.mkdir(args.model_dir)
# args.model_name = os.path.join(args.model_dir, args.ml_cls + ".pickle")
#
# define the app
DebuggingOn = bool(os.getenv('DEBUG', False))  # Whether the Flask app is run in debugging mode, or not.
app = Flask(__name__)
app.config['SECRET_KEY'] = 'comp4111'
CORS(app)  # needed for cross-domain requests, allow everything by default

# # TODO: add ENV for production
# app.logger.info("Downloading a trained model ...")
# cmd = shlex.split("./download_model.sh staging")
# process = subprocess.run(cmd,
#                          stdout=subprocess.PIPE,
#                          stderr=subprocess.PIPE,
#                          universal_newlines=True)

# model_api = load(args.model_name)


def sigterm_handler(_signo, _stack_frame):
    print(str(datetime.datetime.now()) + ': Received SIGTERM')


def sigint_handler(_signo, _stack_frame):
    print(str(datetime.datetime.now()) + ': Received SIGINT')
    sys.exit(0)


signal.signal(signal.SIGTERM, sigterm_handler)
signal.signal(signal.SIGINT, sigint_handler)


# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


@app.route('/health')
def check_health():
    response = app.response_class(
        response="",
        status=200,
        mimetype='application/json')
    return response


@app.route('/')
def index():
    # return "<h1 style='color:blue'>Baseline Main Page</h1>"
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/getapi', methods=['GET'])
def getapi():
    """
    GET request at a sentence level
    http://127.0.0.1:5000/getapi?text=hi%20there,%20how%20are%20you
    """
    text = request.args.get('text', default='the best hotel i stayed so far', type=str)
    app.logger.info("Input: " + text)
    label = model_api.predict([text]).tolist()[0]
    prob = model_api.predict_proba([text]).min()
    result = dict()
    result["input"] = "i am getapi"
    result["Output"] = label
    result["Probability"] = prob
    app.logger.info("model_output: " + str(result))
    result = jsonify(result)
    return result


@app.route('/postapi', methods=['POST'])
def postapi():
    """
    POST request at a sentence level
    """
    json_data = request.json
    text = json_data['rv_text']
    app.logger.info("Input: " + text)
    label = model_api.predict([text]).tolist()[0]
    prob = model_api.predict_proba([text]).max()
    result = dict()
    result["input"] = "Im post api"
    result["Output"] = label
    result["Probability"] = prob
    app.logger.info("model_output: " + str(result))
    result = jsonify(result)
    return result


@app.route('/inference', methods=('GET', 'POST'))
def inference():
    if request.method == 'POST':
        text = request.form['input']
        # content = request.form['content']

        if not text:
            flash('text is required!')
        else:
            label = model_api.predict([text]).tolist()[0]
            prob = model_api.predict_proba([text]).max()

            result = dict()
            result["input"] = text
            result["output"] = label
            result["probability"] = prob
            app.logger.info("model_output: " + str(result))
            return render_template('inference.html', label=label, prob=prob)
    return render_template('inference.html', label="NA", prob="NA")


if __name__ == '__main__':
    """
    kill -9 $(lsof -i:5000 -t) 2> /dev/null
    http://127.0.0.1:5000/getapi_np5?nl=What%20is%20the%20average%20number%20of%20employees%20of%20the%20departments%20whose%20rank%20is%20between%2010%20and%2015?&dbid=department_management
    reply to win £100 weekly!
    """
    app.run(debug=True)
    # app.run(host='0.0.0.0', debug=True)


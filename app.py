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
import activity_clf

argparser = argparse.ArgumentParser(sys.argv[0])

argparser.add_argument('--train_file', help='Trained file', default="../data/smsspamcollection/train.csv", type=str)

argparser.add_argument('--dev_file', help='Developed file', default="../data/smsspamcollection/dev.csv", type=str)

argparser.add_argument('--test_file', help='Tested file', default="../data/smsspamcollection/dev.csv", type=str)

argparser.add_argument("--tfidf", action='store_true', default=False, help="tfidf flag")

argparser.add_argument("--use_hash", action='store_true', default=False, help="hashing flag")

argparser.add_argument("--scaler", action='store_true', default=False, help="scale flag")

argparser.add_argument('--ml_cls', help='Machine learning classifier', default="MLP", type=str)
                                                                # kNN LR DT SVM MLP AB GB RF NB
                                                                #DT P(1); SVM dontwork;
argparser.add_argument('--model_dir', help='Model dir', default="../data/smsspamcollection/", type=str)

args, unknown = argparser.parse_known_args()

model_dir, _ = os.path.split(args.model_dir)

if not os.path.exists(args.model_dir):
    os.mkdir(args.model_dir)
args.model_name = os.path.join(args.model_dir, args.ml_cls + ".pickle")

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

model_api = load(args.model_name)


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
    http://127.0.0.1:5000/getapi?text=0.3009048,-0.023610414,-0.096890735,-0.98681212,-0.974874,-0.9869338,-0.98857872,-0.97683914,-0.98907609,-0.9247045,-0.56511707,-0.79785615,0.83707914,0.67095619,0.84614302,-0.97973823,-0.99974749,-0.9997082,-0.99955126,-0.99078766,-0.98338863,-0.99174568,-0.15476172,-0.74450739,-0.34722364,-0.14045083,0.19130335,-0.33257658,0.29619996,0.12847083,-0.11331774,-0.011946188,0.47044518,0.31707349,-0.24776335,0.24558955,-0.10673192,-0.26626389,0.034102031,-0.59270372,-0.72762928,0.71743485,0.68079188,-0.97539016,-0.98538598,-0.97649265,-0.97526712,-0.98493576,-0.97633273,-0.78326747,0.6752476,0.6707921,-0.67747582,0.73033867,0.67633179,0.5204175,-0.9412147,0.043631744,-0.098430755,-0.9743359,-0.98371759,-0.97647074,-1,-0.58028217,-0.51622287,-0.9043131,0.90696148,-0.90927425,0.91143613,-0.50591816,0.50861951,-0.54613032,0.59753769,-0.33609639,0.36096455,-0.3861169,0.40907647,-0.44654098,0.34680999,-0.99267795,0.084172767,0.013429448,-0.0088533945,-0.98356356,-0.97996292,-0.98975629,-0.98365275,-0.97875556,-0.9884076,-0.98832744,-0.97948611,-0.9907028,0.97436263,0.9868926,0.98721567,-0.98533092,-0.99975351,-0.99960776,-0.99979674,-0.98417218,-0.98188433,-0.98983943,-0.6127758,-0.6824236,-0.77548193,-0.13235271,0.38769952,-0.21232081,0.19758167,-0.028425036,-0.015932283,-0.21282076,0.41315805,0.25865968,-0.10485472,0.088423296,0.37125105,-0.47650063,0.18616998,-0.15965415,-0.031533371,-0.24528334,0.15656219,-0.98720614,-0.95112948,-0.94152353,-0.98844673,-0.95049391,-0.94101743,-0.8677637,-0.96301549,-0.68445398,0.83888191,0.85979932,0.81695888,-0.91820759,-0.99989418,-0.99367864,-0.99628789,-0.98905275,-0.94634956,-0.93831281,-0.53331111,-1,0.24729088,-0.080629772,0.15945507,-0.22285051,0.33379446,-0.063304449,0.10981664,-0.28752973,0.2340657,-0.0031861408,-0.013391925,-0.076630657,0.15139456,-0.62216081,0.6616023,-0.96623296,-0.098648269,-0.0088691152,-0.098258239,-0.98611208,-0.98672806,-0.98367508,-0.98591854,-0.98653507,-0.98416005,-0.9828623,-0.98852835,-0.98560784,0.98547341,0.99076649,0.98701543,-0.98650784,-0.99984859,-0.99987818,-0.99976801,-0.98318237,-0.98654724,-0.98520168,-0.44295349,-0.25234171,-0.49045997,0.061594773,0.094865665,0.073068704,-0.23474652,0.006527973,0.34064123,0.0027975011,0.08370324,0.1106783,0.16635042,-0.080125235,0.31926347,0.51269279,-0.082498417,-0.3302296,-0.98122621,-0.97908161,-0.98367782,-0.97026762,-0.99886342,-0.98122621,-0.99956335,-0.98494142,-0.50265437,-0.0040795164,-0.048447939,0.12351173,-0.06859629,-0.98122621,-0.97908161,-0.98367782,-0.97026762,-0.99886342,-0.98122621,-0.99956335,-0.98494142,-0.50265437,-0.0040795164,-0.048447939,0.12351173,-0.06859629,-0.98586535,-0.98640829,-0.98817166,-0.9803263,-0.96944185,-0.98586535,-0.99971157,-0.99023134,-0.75281944,0.12187208,-0.090212915,-0.17094421,0.055911206,-0.91425567,-0.91926422,-0.90830393,-0.93478513,-0.92075221,-0.91425567,-0.99567926,-0.89757821,0.45725696,-0.16769694,0.23134379,-0.46678129,0.39924745,-0.98654648,-0.9883955,-0.98918745,-0.98878291,-0.98707537,-0.98654648,-0.99985973,-0.99054926,-0.38986875,0.07231011,-0.26985748,0.44609977,-0.34592583,-0.98350603,-0.97597165,-0.98590039,-0.98834378,-0.97476722,-0.98780159,-0.98518044,-0.97420588,-0.9857492,-0.99027246,-0.98477613,-0.99024208,-0.98458474,-0.99782666,-0.99035786,-0.9826049,-0.99985609,-0.9994351,-0.99969982,-0.97976045,-0.9856305,-0.98541841,-0.87389237,-0.7868864,-0.80988792,-1,-0.66666667,-1,0.035258617,-0.033292597,0.25595446,-0.40197375,-0.75682953,-0.48395228,-0.84541118,-0.72520462,-0.91294545,-0.99990423,-0.99982708,-0.99975192,-0.99958538,-0.99948056,-0.99980153,-0.99985679,-0.99983364,-0.99987497,-0.9996694,-0.99961594,-0.99984904,-0.99986797,-0.99952805,-0.9994658,-0.99960561,-0.99994176,-0.99960239,-0.99939796,-0.99941553,-0.99990806,-0.99999824,-0.99940436,-0.99983636,-0.99936829,-0.99994349,-0.99944684,-0.99953073,-0.99972605,-0.99969526,-0.99981185,-0.99982673,-0.99974676,-0.99959436,-0.99948248,-0.99982302,-0.99971401,-0.99985061,-0.99973492,-0.99958934,-0.99970495,-0.99981731,-0.98286485,-0.9797115,-0.98779743,-0.98593862,-0.9817359,-0.99028732,-0.98159665,-0.97995524,-0.98946177,-0.98872796,-0.98708303,-0.98988584,-0.97969309,-0.98384438,-0.98868827,-0.98289007,-0.99975325,-0.99960773,-0.9997968,-0.9810689,-0.97953396,-0.98535618,-1,-1,-1,-0.32,-0.04,-0.32,0.1059467,-0.010313722,0.08628947,-0.44406385,-0.82213348,-0.60116551,-0.95437261,-0.65018903,-0.89060754,-0.99993282,-0.99987421,-0.99978302,-0.99959255,-0.99953538,-0.99982959,-0.99987231,-0.99994328,-0.99989073,-0.99965907,-0.99963157,-0.99986359,-0.99983388,-0.99946365,-0.99957563,-0.99971144,-0.99991777,-0.99961024,-0.99936372,-0.99918754,-0.99986783,-0.99987527,-0.99964522,-0.99976533,-0.99918271,-0.99987278,-0.99970779,-0.99946205,-0.99975305,-0.99967283,-0.99981244,-0.99985022,-0.99975062,-0.99953841,-0.99925916,-0.9999206,-0.99966571,-0.99986041,-0.99967777,-0.99928861,-0.9997645,-0.99980857,-0.98404221,-0.95693298,-0.93642651,-0.98814147,-0.94804454,-0.94851088,-0.98449986,-0.96186908,-0.95332055,-0.98899427,-0.94533076,-0.9495264,-0.99991607,-0.96513733,-0.88647402,-0.96016011,-0.99989341,-0.99868066,-0.99803382,-0.98518391,-0.97973427,-0.9729082,-0.6341064,-0.32543592,-0.41209834,-1,-1,-1,-0.12132554,-0.069338384,0.25617829,-0.38966513,-0.72774944,0.19971835,-0.13045456,0.23633051,-0.069730069,-0.99990884,-0.99990628,-0.99994642,-0.99981726,-0.99976348,-0.99992011,-0.99995963,-1,-0.99990077,-0.99988442,-0.99980488,-0.9999775,-0.99989984,-0.99981264,-0.99799526,-0.99980065,-0.99994892,-0.9997385,-0.99965536,-0.99931772,-0.99868323,-0.99824306,-0.99852129,-0.99987334,-0.99958291,-0.9983133,-0.99860337,-0.99966991,-0.99826343,-0.9995102,-0.99954381,-0.99941196,-0.9988648,-0.99764969,-0.99662129,-0.99611552,-0.99818576,-0.99928299,-0.99853913,-0.99640161,-0.99813483,-0.99914304,-0.9798187,-0.98075362,-0.97480229,-0.98780834,-0.99452676,-0.9798187,-0.99956992,-0.98654173,-0.74806523,-1,0.06306111,-0.61236269,-0.88334067,-0.9863168,-0.98497016,-0.98205687,-0.98907128,-0.99873554,-0.9863168,-0.99975362,-0.98426996,-1,-0.96825397,0.11206121,-0.48552005,-0.82207319,-0.9398944,-0.91990834,-0.93027435,-0.90961635,-0.97776931,-0.9398944,-0.99646733,-0.96452493,-0.31941021,-1,-0.13540755,0.21075502,-0.091277268,-0.98854277,-0.98847859,-0.98592981,-0.99149865,-0.99653523,-0.98854277,-0.99989709,-0.98647635,-0.80997573,-0.96825397,-0.049977157,-0.52091696,-0.85619433,0.084151299,0.2317745,0.2785723,-0.081150432,0.77405349,-0.4777379,-0.5071373,"2","LAYING"

    """
    text = request.args.get('text', default='the best hotel i stayed so far', type=str)
    app.logger.info("Input: " + text)

    label, prob = activity_clf.pred(text)
    # label = model_api.predict([text]).tolist()[0]
    # prob = model_api.predict_proba([text]).max()
    result = dict()
    result["input"] = text
    result["Output"] = label
    result["Probability"] = prob
    app.logger.info("model_output: " + str(result))
    # result = jsonify(result)
    return result


@app.route('/postapi', methods=['POST'])
def postapi():
    """
    POST request at a sentence level
    """
    json_data = request.json
    text = json_data['rv_text']
    app.logger.info("Input: " + text)
    label, prob = activity_clf.pred(text)
    # label = model_api.predict([text]).tolist()[0]
    # prob = model_api.predict_proba([text]).max()
    result = dict()
    result["input"] = text
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
            label, prob = activity_clf.pred(text)
            # label = model_api.predict([text]).tolist()[0]
            # prob = model_api.predict_proba([text]).max()
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
    #app.run(host='0.0.0.0', debug=True)


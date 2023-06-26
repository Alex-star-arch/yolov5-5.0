import threading

from flask import Flask, render_template, request

from timeset import *

app = Flask(__name__)




@app.route('/clock', methods=['GET', 'POST'])
def index():
    global detect_hour, detect_minute


    if request.method == 'POST':

        detect_hour = int(request.form['hour'])
        detect_minute = int(request.form['minute'])
        offLineList=request.form['offLineList']
        getStreams(offLineList)
        set_detect_time(detect_hour, detect_minute)

    return render_template('index.html',detect_hour=detect_hour, detect_minute=detect_minute)


if __name__ == '__main__':
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()
    app.run(debug=True,port=5001)

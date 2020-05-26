import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from flask import Flask, flash, request, redirect, render_template

PEOPLE_FOLDER = os.path.join('static')
ALLOWED_EXTENSIONS = {'excel', 'csv', 'txt'}

warnings.filterwarnings('ignore')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def initial():
    return render_template('initial')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global dataframe
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            dataframe = pd.read_csv(request.files['file'], header=0, index_col=0)
            file = request.files['file']
            filename = file.filename
            print('{} isimli dosya yuklendi'.format(filename))
            print(dataframe)
            dataframe = dataframe.dropna()
            if 'submit-button' in request.form:
                user_answer = request.form['month']
                if user_answer == 'one':
                    data = dataframe
                    model = pm.auto_arima(data.values, start_p=1, start_q=1,
                                          test='adf',  # use adftest to find optimal 'd'
                                          max_p=5, max_q=3,  # maximum p and q
                                          m=1,  # frequency of series
                                          d=None,  # let model determine 'd'
                                          seasonal=False,  # No Seasonality
                                          start_P=0,
                                          D=0,
                                          trace=True,
                                          error_action='ignore',
                                          suppress_warnings=True,
                                          stepwise=True)

                    print(model.summary())
                    n_periods = 30
                    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
                    index_of_fc = np.arange(len(data.values), len(data.values) + n_periods)

                    # make series for plotting purposer
                    fc_series = pd.Series(fc, index=index_of_fc)
                    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
                    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

                    # Plotr
                    plt.figure(figsize=(12, 5), dpi=100)
                    plt.plot(data.values, label='Gerçek Veri')
                    plt.plot(fc_series, color='red', label="Tahmin")
                    plt.title("Gelecek 1 Ay Tahminlemesi", size=20)
                    plt.xlabel("Günler", size=15)
                    plt.ylabel("Inbound", size=15)
                    plt.legend(loc='upper right', fontsize=11)
                    plt.savefig('static/biraylik.png')
                    full_filename = os.path.join('static/biraylik.png')
                    return render_template('index.html', result=fc_series, user_image=full_filename)

                if user_answer == 'three':
                    data = dataframe
                    model = pm.auto_arima(data.values, start_p=1, start_q=1,
                                          test='adf',  # use adftest to find optimal 'd'
                                          max_p=5, max_q=3,  # maximum p and q
                                          m=1,  # frequency of series
                                          d=None,  # let model determine 'd'
                                          seasonal=False,  # No Seasonality
                                          start_P=0,
                                          D=0,
                                          trace=True,
                                          error_action='ignore',
                                          suppress_warnings=True,
                                          stepwise=True)

                    print(model.summary())
                    n_periods = 90
                    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
                    index_of_fc = np.arange(len(data.values), len(data.values) + n_periods)

                    # make series for plotting purposer
                    fc_series = pd.Series(fc, index=index_of_fc)
                    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
                    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

                    # Plotr
                    plt.figure(figsize=(12, 5), dpi=100)
                    plt.plot(data.values, label='Gerçek Veri')
                    plt.plot(fc_series, color='red', label="Tahmin")
                    plt.title("Gelecek 3 Ay Tahminlemesi", size=20)
                    plt.xlabel("Günler", size=15)
                    plt.ylabel("Inbound", size=15)
                    plt.legend(loc='upper right', fontsize=11)
                    plt.savefig('static/ucaylik.png')
                    full_filename = os.path.join('static/ucaylik.png')
                    return render_template('index.html', result=fc_series, user_image=full_filename)

                if user_answer == 'six':
                    data = dataframe
                    model = pm.auto_arima(data.values, start_p=1, start_q=1,
                                          test='adf',  # use adftest to find optimal 'd'
                                          max_p=5, max_q=3,  # maximum p and q
                                          m=1,  # frequency of series
                                          d=None,  # let model determine 'd'
                                          seasonal=False,  # No Seasonality
                                          start_P=0,
                                          D=0,
                                          trace=True,
                                          error_action='ignore',
                                          suppress_warnings=True,
                                          stepwise=True)

                    print(model.summary())

                    n_periods = 180
                    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
                    index_of_fc = np.arange(len(data.values), len(data.values) + n_periods)

                    # make series for plotting purposer
                    fc_series = pd.Series(fc, index=index_of_fc)
                    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
                    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

                    # Plotr
                    plt.figure(figsize=(12, 5), dpi=100)
                    plt.plot(data.values, label='Gerçek Veri')
                    plt.plot(fc_series, color='red', label="Tahmin")
                    plt.title("Gelecek 6 Ay Tahminlemesi", size=20)
                    plt.xlabel("Günler", size=15)
                    plt.ylabel("Inbound", size=15)
                    plt.legend(loc='upper right', fontsize=11)

                    plt.savefig('static/altiaylik.png')
                    full_filename = os.path.join('static/altiaylik.png')
                    return render_template('index.html', result=fc_series, user_image=full_filename)

                    # FOR PDF
                    # rendered = render_template('index.html', result=fc_series, user_image=full_filename)
                    # pdf = pdfkit.from_string(rendered,False)
                    # response = make_response(pdf)
                    # response.headers['Content-Type'] = 'application/pdf'
                    # response.headers['Content-Disposition'] = 'attachment; filename=output.pdf'
                    # return response

            if 'delete-button' in request.form:
                if os.path.exists("static/biraylik.png"):
                    os.remove("static/biraylik.png")
                else:
                    print("The file does not exist")
                if os.path.exists("static/ucaylik.png"):
                    os.remove("static/ucaylik.png")
                else:
                    print("The file does not exist")
                if os.path.exists("static/altiaylik.png"):
                    os.remove("static/altiaylik.png")
                else:
                    print("The file does not exist")
                return render_template('initial')


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()

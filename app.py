import itertools
import os
import warnings

import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pandas as pd
import pdfkit as pdfkit
import pmdarima as pm
import statsmodels.api as sm
from flask import Flask, flash, request, redirect, render_template, make_response
from flask_weasyprint import HTML

PEOPLE_FOLDER = os.path.join('static')
ALLOWED_EXTENSIONS = {'excel', 'csv', 'txt'}

warnings.filterwarnings('ignore')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_pickle(df):
    traindata = df
    traindata = traindata.set_index('Tarih')
    traindata.index = pd.to_datetime(traindata.index, unit='ns')
    print(traindata.index)

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 176) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

    for pr in pdq:
        for pr_s in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=pr,
                                                seasonal_order=pr_s,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                print('ARIMA{}x{}12 - AIC:{}'.format(pr, pr_s, results.aic))
            except:
                continue

    mod = sm.tsa.statespace.SARIMAX(traindata,
                                    order=(1, 1, 1),
                                    seasonal_order=(1, 1, 0, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit()
    print(results)
    #
    # dt_string = "2017/02/05 00:00:00.0"
    # dt_string2 = "2017/02/09 00:00:00.0"
    #
    # # Considering date is in dd/mm/yyyy format
    # dt_object1 = datetime.strptime(dt_string, '%Y/%m/%d %H:%M:%S.%f')
    # dt_object2 = datetime.strptime(dt_string2, '%Y/%m/%d %H:%M:%S.%f')
    # pred = results.get_prediction(start=50, end=55)
    pred = results.get_prediction(start=pd.to_datetime('2017/02/05'), end=pd.to_datetime('2017/07/09'), dynamic=False)

    pred_ci = pred.conf_int()
    print('ilk')
    print(pred_ci)
    # ax = traindata['2017-01':].plot(label='Öngörülen Veriler')
    # pred.predicted_mean.plot(ax=ax, label='Tahminlenen kısım', alpha=.7, figsize=(12, 4))
    # ax.fill_between(pred_ci.index,
    #                 pred_ci.iloc[:, 0],
    #                 pred_ci.iloc[:, 1], color='k', alpha=.2)
    # ax.set_xlabel('Tahmini Tarihler')
    # ax.set_ylabel('boyut')
    # plt.title('\nTahmin Sonuçları\n')
    # plt.legend()
    # plt.show()
    # Get forecast 500 steps ahead in future
    pred_uc = results.get_forecast(steps=500)

    # Get confidence intervals of forecasts
    pred_uc = pred_uc.conf_int()
    print('ikinci')

    print(pred_uc)
    ax = traindata['2017-06':].plot(label='Öngörülen Veriler')
    pred.predicted_mean.plot(ax=ax, label='Tahminlenen kısım', alpha=.7, figsize=(12, 4))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Tahmini Tarihler')
    ax.set_ylabel('boyut ')
    plt.title('\nTahmin Sonuçları\n')
    plt.legend()
    plt.show()

    # # render dataframe as html
    # html = pred_ci.to_html()
    #
    # # write html to file
    # text_file = open("pred.html", "w")
    # text_file.write(html)
    # text_file.close()

    conf_date = pred.conf_int().index
    conf_lower = pred.conf_int()['lower Inbound']
    conf_upper = pred.conf_int()['upper Inbound']
    conf_mean = (conf_upper + conf_lower) / 2

    results_df = pd.DataFrame({
        "conf_date": conf_date,
        "conf_lower": conf_lower,
        "conf_upper": conf_upper,
        "conf_mean": conf_mean
    })
    results_df.index = pd.to_datetime(results_df.index)
    print(results_df)
    dictionary = {
        'date': [],
        'lower': [],
        'upper': [],
        'mean': []
    }
    results_df.reset_index(drop=True, inplace=True)
    results_df.set_index('conf_date')

    dictionary = results_df.to_dict(orient='records')

    for key in dictionary:
        date_time = key['conf_date'].strftime("%m/%d/%Y, %H:%M:%S")
        key['conf_date'] = date_time
    dictionary = str(dictionary)
    return dictionary




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

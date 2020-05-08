import datetime
import itertools
import pickle
import warnings
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from flask import Flask, flash, request, redirect, render_template
from datetime import datetime

ALLOWED_EXTENSIONS = {'excel', 'csv', 'txt'}

warnings.filterwarnings('ignore')

app = Flask(__name__)


# logging_conf_path = os.path.normpath(os.path.join(os.path.dirname(__file__), 'logging.conf'))
# logging.config.fileConfig(logging_conf_path)
# log = logging.getLogger(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def create_model(df, traindata):
#     df.dropna(axis=1, how='all', inplace=True)
#     """
#
#     TEST DATASI DA KULLANILACAK
#
#     """
#
#
#     sample = traindata.resample('D').mean()
#
#     p = d = q = range(0, 2)
#     pdq = list(itertools.product(p, d, q))
#     seasonal_pdq = [(x[0], x[1], x[2], 754) for x in list(itertools.product(p, d, q))]
#     print('Examples of parameter combinations for Seasonal ARIMA...')
#     print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
#     print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
#     print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
#     print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
#
#     for pr in pdq:
#         for pr_s in seasonal_pdq:
#             try:
#                 mod = sm.tsa.statespace.SARIMAX(y,
#                                                 order=pr,
#                                                 seasonal_order=pr_s,
#                                                 enforce_stationarity=False,
#                                                 enforce_invertibility=False)
#                 results = mod.fit()
#                 print('ARIMA{}x{}12 - AIC:{}'.format(pr, pr_s, results.aic))
#             except:
#                 continue
#
#     mod = sm.tsa.statespace.SARIMAX(sample,
#                                     order=(1, 1, 1),
#                                     seasonal_order=(1, 1, 0, 12),
#                                     enforce_stationarity=False,
#                                     enforce_invertibility=False)
#     results = mod.fit()
#     pred = results.get_prediction(start=pd.to_datetime('{}'.format(start_date)),
#                                   end=pd.to_datetime('{}'.format(end_date)), dynamic=False)
#     conf_date = pred.conf_int().index
#     conf_lower = pred.conf_int()['lower size']
#     conf_upper = pred.conf_int()['upper size']
#     conf_mean = (conf_upper + conf_lower) / 2
#     results_df = pd.DataFrame({
#         "conf_date": conf_date,
#         "conf_lower": conf_lower,
#         "conf_upper": conf_upper,
#         "conf_mean": conf_mean
#     })
#     print(results_df)


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
    pred = results.get_prediction(start=pd.to_datetime('2017/02/05'),end=pd.to_datetime('2017/07/09'), dynamic=False)

    pred_ci = pred.conf_int()
    print('ilk')
    print(pred_ci)
    ax = traindata['2017-01':].plot(label='Öngörülen Veriler')
    pred.predicted_mean.plot(ax=ax, label='Tahminlenen kısım', alpha=.7, figsize=(12, 4))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Tahmini Tarihler')
    ax.set_ylabel('Disk Boyutu')
    plt.title('\nTahmin Sonuçları\n')
    plt.legend()
    plt.show()
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
    ax.set_ylabel('Disk ')
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


# def predict_model_(dataframe, content, results):
#     start_date = content['start_date']
#     end_date = content['end_date']
#     predicted_df = pd.DataFrame(dataframe)
#     predicted_df['date'] = pd.to_datetime(predicted_df['date'])
#
#     pred = results.get_prediction(start=pd.to_datetime('{}'.format(start_date)),
#                                   end=pd.to_datetime('{}'.format(end_date)), dynamic=False)
#     conf_date = pred.conf_int().index
#     conf_lower = pred.conf_int()['lower size']
#     conf_upper = pred.conf_int()['upper size']
#     conf_mean = (conf_upper + conf_lower) / 2
#
#     results_df = pd.DataFrame({
#         "conf_date": conf_date,
#         "conf_lower": conf_lower,
#         "conf_upper": conf_upper,
#         "conf_mean": conf_mean
#     })
#     dictionary = {
#         'date': [],
#         'lower': [],
#         'upper': [],
#         'mean': []
#     }
#     results_df.reset_index(drop=True, inplace=True)
#     results_df.set_index('conf_date')
#
#     dictionary = results_df.to_dict(orient='records')
#     for key in dictionary:
#         date_time = key['conf_date'].strftime("%m/%d/%Y, %H:%M:%S")
#         key['conf_date'] = date_time
#     dictionary2 = str(dictionary)
#     print(dictionary2)

# def jupyter(df):
#     df.dropna(axis=1, how='all', inplace=True)
#
#     size = df.copy()
#     train = size.sample(frac=0.7)
#     test = size.drop(train.index)
#     train = train.set_index('Tarih')
#     sample = train.resample('D').mean()
#     p = d = q = range(0, 2)
#     pdq = list(itertools.product(p, d, q))
#     seasonal_pdq = [(x[0], x[1], x[2], 754) for x in list(itertools.product(p, d, q))]
#     print('Examples of parameter combinations for Seasonal ARIMA...')
#     print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
#     print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
#     print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
#     print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
#     for pr in pdq:
#         for pr_s in seasonal_pdq:
#             try:
#                 mod = sm.tsa.statespace.SARIMAX(y,
#                                                 order=pr,
#                                                 seasonal_order=pr_s,
#                                                 enforce_stationarity=False,
#                                                 enforce_invertibility=False)
#                 results = mod.fit()
#                 print('ARIMA{}x{}12 - AIC:{}'.format(pr, pr_s, results.aic))
#             except:
#                 continue
#     mod = sm.tsa.statespace.SARIMAX(sample,
#                                     order=(1, 1, 1),
#                                     seasonal_order=(1, 1, 0, 12),
#                                     enforce_stationarity=False,
#                                     enforce_invertibility=False)
#     results = mod.fit()
#     pred = results.get_prediction(start=pd.to_datetime('2018-01-01'), end=pd.to_datetime('2020-04-05'), dynamic=False)
#     pred_ci = pred.conf_int()
#     print(pred_ci)


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
            dataframe = pd.read_csv(request.files['file'])
            file = request.files['file']
            filename = file.filename
            print('{} isimli dosya yuklendi'.format(filename))
            print(dataframe)
            if 'submit-button' in request.form:
                user_answer = request.form['month']
                if user_answer == 'one':
                    lastdate = dataframe["Tarih"].iloc[-1]
                    # content = {
                    #     "start_date": "2020-05-07 00:00:00.0",
                    #     "end_date": "2020-05-07 00:00:00.0"
                    # }

                    dict = create_pickle(dataframe)
                    return dict
                if user_answer == 'three':
                    dataframe['Tarih'] = pd.to_datetime(dataframe['Tarih'])
                    lastdate = dataframe["Tarih"].iloc[-1]
                    use_date = lastdate + relativedelta(months=+1)
                    print(dataframe.dtypes)
                    print(dataframe)
                    return use_date
                if user_answer == 'six':
                    lastdate = dataframe["Tarih"].iloc[-1]
                    return lastdate

        # predict_model_(dataframe,)


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()

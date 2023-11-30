from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
import zipfile
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from models import ECG_CNN_TransformerBIN, ECG_CNN_Transformer
from utils import inference_ovBIN, inference_ov, sigmoid, plot_data
from sklearn.preprocessing import StandardScaler
import numpy as np
from joblib import load
import pathlib
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json



scaler = load('static/models/scaler_main.joblib')

checkpointBIN = torch.load('static/models/BinMODELwithNOISEFULLSETT01loss.pt')
modelBIN  = ECG_CNN_TransformerBIN().cuda()
modelBIN.load_state_dict(checkpointBIN)

checkpoint = torch.load('static/models/WITHNOISEFULLnewf10-56loss0-15classes6.pt')
model  = ECG_CNN_Transformer().cuda()
model.load_state_dict(checkpoint)




app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/users')
def users():
    return render_template('users.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file.save(f'static/temp/{file.filename}')
    with zipfile.ZipFile(f'static/temp/{file.filename}', 'r') as zip_ref:
        zip_ref.extractall('static/temp/unzipped')

    files = [f for f in os.listdir('static/temp/unzipped') if f.endswith('.npy')]

    for file in files:
        data = np.load('static/temp/unzipped/' + file)[:,::10]
        plot_data(data, file)

    test_X = []
    record_names = [f for f in os.listdir('static/temp/unzipped') if f.endswith('.npy')]
    for key in record_names:
      test_X.append(np.load('static/temp/unzipped/'+key))


    normalized_data_listtest = [scaler.transform(data) for data in test_X]
    test_X = normalized_data_listtest

    eval_res = []
    for x in range(len(test_X)):
       eval_res.append(inference_ovBIN(modelBIN, torch.from_numpy(test_X[x]).cuda()))
    eval_res = [sigmoid(i) for i in eval_res]


    answers = []
    record_names = files
    for i in range(len(record_names)):
      test_record = np.load('static/temp/unzipped/'+record_names[i])
      test_record = scaler.transform(test_record)
      answers.append(inference_ov(model, torch.from_numpy(test_record).cuda()))

    reses = [1 if float(j) >= 0.275 else 0 for j in list(eval_res)]

    new_arr = []
    for k in range(len(answers)):
        for j in range(len(answers[k])):
            new_arr.append(answers[k][j].detach().cpu().numpy())
    for i in range(len(new_arr)):
        for j in range(len(new_arr[i])):
            try:
                new_arr[i][j] = sigmoid(new_arr[i][j])
            except:
                new_arr[i][j] = 0

    ress = []
    for kk in range(len(reses)):
        if reses[kk] == 0:
            ress.append([0,0,0,0,0,0,1])
        else:
            ress.append([1 if float(j) >= 0.1 else 0 for j in new_arr[kk]]+[0])
    with open('static/users.json', 'r') as f:
        data = json.load(f)
    
    new_users = [
            {
                'name': files[i].replace('.npy', ''),
                'image_url': url_for('static', filename='/plots/' + files[i].replace('.npy', '.jpg')),
                'illnesses': {
                    'перегородочный': 'НАЙДЕНО' if ress[i][0] else 'НЕ НАЙДЕНО',
                    'передний': 'НАЙДЕНО' if ress[i][1] else 'НЕ НАЙДЕНО',
                    'боковой': 'НАЙДЕНО' if ress[i][2] else 'НЕ НАЙДЕНО',
                    'передне-боковой': 'НАЙДЕНО' if ress[i][3] else 'НЕ НАЙДЕНО',
                    'передне-перегородочный': 'НАЙДЕНО' if ress[i][4] else 'НЕ НАЙДЕНО',
                    'нижний': 'НАЙДЕНО' if ress[i][5] else 'НЕ НАЙДЕНО'
                },
                'raw_data': ress[i]
            } for i in range(len(ress))
        ]


    for new_user in new_users:
        data['users'] = [user for user in data['users'] if user['name'] != new_user['name']]

    data['total'] = data['total']+len(ress)
    data['users'].extend(new_users)

    submit = pd.read_csv('static/submit.csv')
    for i in range(len(submit)):
        name = submit.iloc[i]['record_name']
        for user in data['users']:
            if user['name'] == name:
                submit.loc[i, 'перегородочный'] = user['raw_data'][0]
                submit.loc[i, 'передний'] = user['raw_data'][1]
                submit.loc[i, 'боковой'] = user['raw_data'][2]
                submit.loc[i, 'передне-боковой'] = user['raw_data'][3]
                submit.loc[i, 'передне-перегородочный'] = user['raw_data'][4]
                submit.loc[i, 'нижний'] = user['raw_data'][5]
                submit.loc[i, 'норма'] = user['raw_data'][6]
    submit.to_csv('static/submit.csv', index=False)
      
    with open('static/users.json', 'w') as f:
        json.dump(data, f)

    folder = 'static/temp'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    
    return redirect('/')


@app.route('/upload_csv', methods=['POST'])
def upload_csv():   
    file = request.files['file']
    file.save(f'static/submit.csv')

    return redirect('/')


@app.route('/api/get_users')
def get_users():
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))
    
    with open('static/users.json', 'r') as f:
        data = json.load(f)

    data['users'] = data['users'][
                           min(len(data['users']), offset):min(len(data['users']), offset + limit)]

    return jsonify(data)


@app.route('/api/get_user/<name>')
def get_user(name):
    with open('static/users.json', 'r') as f:
        data = json.load(f)
        
    return_user = None
    for key, user in enumerate(data['users']):
        if name == user['name']:
            return_user = data['users'][key]
    if return_user:
        return jsonify(return_user)
    else:
        return jsonify(return_user), 404


if __name__ == '__main__':
    app.run()

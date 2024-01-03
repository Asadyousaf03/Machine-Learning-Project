from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

## route for home page

#@app.route('/')

def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method =='GET': 
        return render_template('home.html')
    else:
        data=CustomData(
            duration=request.form.get('duration'),
            protocol_type=request.form.get('protocol_type'),
            service=request.form.get('service'),
            flag=request.form.get('flag'),
            src_bytes=int(request.form.get('src_bytes')),
            dst_bytes=int(request.form.get('dst_bytes')),
            land=int(request.form.get('land')),
            wrong_fragment=int(request.form.get('wrong_fragment')),
            urgent=int(request.form.get('urgent')),
            hot=int(request.form.get('hot')),
            num_failed_logins=int(request.form.get('num_failed_logins')),
            logged_in=int(request.form.get('logged_in')),
            num_compromised=int(request.form.get('num_compromised')),
            root_shell=int(request.form.get('root_shell')),
            su_attempted=int(request.form.get('su_attempted')),
            num_file_creations=int(request.form.get('num_file_creations')),
            num_shells=int(request.form.get('num_shells')),
            num_access_files=int(request.form.get('num_access_files')),
            is_guest_login=int(request.form.get('is_guest_login')),
            count=int(request.form.get('count')),
            srv_count=int(request.form.get('srv_count')),
            serror_rate=float(request.form.get('serror_rate')),
            rerror_rate=float(request.form.get('rerror_rate')),
            same_srv_rate=float(request.form.get('same_srv_rate')),
            diff_srv_rate=float(request.form.get('diff_srv_rate')),
            srv_diff_host_rate=float(request.form.get('srv_diff_host_rate')),
            dst_host_count=int(request.form.get('dst_host_count')),
            dst_host_diff_srv_rate=float(request.form.get('dst_host_diff_srv_rate')),
            dst_host_same_src_port_rate=float(request.form.get('dst_host_same_src_port_rate')),
            dst_host_srv_count=int(request.form.get('dst_host_srv_count')),
            dst_host_srv_diff_host_rate=float(request.form.get('dst_host_srv_diff_host_rate')),
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)   



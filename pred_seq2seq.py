# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 09:19:48 2022

@author: Administrator
"""
from numpy import array,reshape
import pandas as pd
from pandas.tseries.offsets import Day
import multiprocessing as mp
import sys
path = sys.path
sys.path.append('C:/Users/Administrator/static') #需更改模組路徑
import joblib
import tensorflow as tf



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def pred_process(tar,num,look_back,today):
    pred_dataframe = pd.DataFrame() 
    scaler = joblib.load(f'model/{num}.gz')
    model = tf.keras.models.load_model(f'model/{num}.h5')
    
    want_test = array(tar[look_back*num:look_back*(num+1)])
    want_test = scaler.transform(want_test.reshape(-1,1))

    for i in range(3):
        want_test = reshape(want_test, (1,look_back,1))
        Pred_value = model.predict(want_test)
        Pred = scaler.inverse_transform(Pred_value.reshape(-1, 1))
        for i in range(len(Pred)):
            pred_dataframe.loc[today.strftime('%Y-%m-%d'),(num*look_back+i)] = Pred[i,0]
        today = today+1*Day()
        want_test = Pred_value 
        
    return pred_dataframe

def collect_results(result):
    """Uses apply_async's callback to setup up |a separate Queue for each process"""
    results.append(result)


    
if __name__ == "__main__":
    sta = time.time()
    config ={
    'look_back' : 12,
    }
    nowtime = datetime.now()
    nowtime = nowtime-timedelta(210)
    
    # main 
    path  = 'C:/Users/Administrator/static_seq2seq'
    ts_list = pd.read_csv("dataset/weekdays-5min.csv")
    ts_list = ts_list.set_index('time').drop(columns=['Unnamed: 0'])
    ts_list = ts_list.fillna(method ='ffill')
    ts_collect = ts_list.loc[:,nowtime.strftime('%Y-%m-%d'):].T
    
    start = nowtime.strftime('%Y-%m-%d') #90天前開始
    
    tar = ts_list[start]
    
    results = []
    
    '''
    for i in range (24):#(int(288/config['look_back'])):
        pred_data = pred_process(path,tar,i,config,nowtime)
        collect_results(pred_data)
    
    
    #cores =mp.cpu_count()
    
    '''
    print (f'Parent process {os.getpid()}')

    p = mp.Pool()
    processes = int(288/config['look_back'])
    for i in range(processes):
        
        result = p.apply_async(pred_process, args=(path,tar,i,config['look_back'],nowtime),callback=collect_results)
    print ('Waiting for all subprocesses done...')
    p.close()
    p.join()
    
    for i in range(1,len(results)):
        results[0] = results[0].join(results[i])
    result = results[0].T.sort_index()
    end = time.time()
    print(end-sta)
    #wandb.finish()
    for i in range(3):
        new_day = (nowtime+1*Day()) #預測的天
        print(new_day)
        ev.Observation_accuracy(ts_collect.loc[new_day.strftime('%Y-%m-%d')].T,result.T.loc[new_day.strftime('%Y-%m-%d')].fillna(method ='ffill').T)
        
        nowtime = new_day
    
    print("完成")
    
    
    
    



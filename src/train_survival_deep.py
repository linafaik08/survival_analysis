import itertools
import pandas as pd
import numpy as np
import datetime
import json
import os


import torch # For building the networks 
import torchtuples as tt # Some useful functions
from torchtuples.callbacks import Callback
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ReduceLROnPlateau
from pycox.evaluation import EvalSurv

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

import itertools
    
class Concordance(tt.cb.MonitorMetrics):
    def __init__(self, x, durations, events, per_epoch=1, discrete=False, verbose=True):
        super().__init__(per_epoch)
        self.x = x
        self.durations = durations
        self.events = events
        self.verbose = verbose
        self.discrete = discrete
    
    def on_epoch_end(self):
        super().on_epoch_end()
        if self.epoch % self.per_epoch == 0:
            if not(self.discrete):
                _ = self.model.compute_baseline_hazards()
                surv = self.model.predict_surv_df(self.x)
            else:
                surv = self.model.interpolate(10).predict_surv_df(self.x)
                
            ev = EvalSurv(surv, self.durations, self.events)
            concordance = ev.concordance_td()
            self.append_score('concordance', concordance)
            
            if self.verbose:
                print('concordance:', round(concordance, 5))

def score_model(model, data, durations, events, discrete=False):
    if not(discrete):
        surv = model.predict_surv_df(data)
    else:
        surv = model.interpolate(10).predict_surv_df(data)
    return EvalSurv(surv, durations, events, censor_surv='km').concordance_td()
    
def train_deep_surv(train, val, test, model_obj, out_features,
                    n_nodes, n_layers, dropout , lr =0.01, 
                    batch_size = 16, epochs = 500, output_bias=False,  
                    tolerance=10, 
                    model_params = {}, discrete= False,
                    print_lr=True, print_logs=True, verbose = True):
    
    in_features = train[0].shape[1]
    num_nodes = [n_nodes]*(n_layers)
    batch_norm = True
    
    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features, 
        batch_norm, dropout, output_bias=output_bias)

    opt = torch.optim.Adam
    model = model_obj(net, opt, **model_params)
    model.optimizer.set_lr(lr)

    callbacks = [
        tt.callbacks.EarlyStopping(patience=15),
        Concordance(val[0], val[1][0], val[1][1], per_epoch=5, discrete=discrete)
    ]

    log = model.fit(train[0], train[1], batch_size, epochs, callbacks, verbose,
                val_data=val, val_batch_size=batch_size)

    logs_df = log.to_pandas().reset_index().melt(
        id_vars="index", value_name="loss", var_name="dataset").reset_index()
    
    #print("Last lr", lr_scheduler.get_last_lr())
        
    if print_logs:
        fig = px.line(logs_df, y="loss", x="index", color="dataset", width=800, height = 400)
        fig.show()
        
    # scoring the model
    scores = {
        'train': score_model(model, train[0], train[1][0], train[1][1]),
        'val': score_model(model, val[0], val[1][0], val[1][1]),
        'test': score_model(model, test[0], test[1][0], test[1][1])
    }
        
    return logs_df, model, scores

def grid_search_deep(train, val, test, out_features, grid_params, model_obj):
    best_score = -100
    
    n = 1
    for k, v in grid_params.items():
        n*=len(v)
        
    print(f'{n} total scenario to run')
    
    result = {}
    
    try: 
        for i, combi in enumerate(itertools.product(*grid_params.values())):
            params = {k:v for k,v in zip(grid_params.keys(), combi)}

            params_ = params.copy()
            if 'model_params' in params_.keys():
                params_['model_params'] = {k:v for k,v in params['model_params'].items() if k!='duration_index'}

            print(f'{i+1}/{n}: params: {params_}')

            logs_df, model, scores = train_deep_surv(train, val, test, model_obj,out_features,
                                      print_lr=False, print_logs=False, verbose = True, **params)

            result[i] = {}
            for k, v in params_.items():
                result[i][k] = v
            result[i]['lr'] = model.optimizer.param_groups[0]['lr']
            for k, score in scores.items():
                result[i]['score_'+k] = score

            score = scores['test']
            print('Current score: {} vs. best score: {}'.format(score, best_score))

            if best_score < score:
                best_score = score
                best_model = model
    
    except KeyboardInterrupt:
        pass
        
    table = pd.DataFrame.from_dict(result, orient='index')
    
    return best_model, table.sort_values(by="score_test", ascending=False).reset_index(drop=True)


def load_model(filename, path, model_obj, in_features, out_features, params):
    num_nodes = [int(params["n_nodes"])] * (int(params["n_layers"]))
    del params["n_nodes"]
    del params["n_layers"]

    if 'model_params' in params.keys():
        model_params = json.loads(params['model_params'].replace('\'', '\"'))
        del params['model_params']
        net = tt.practical.MLPVanilla(
            in_features=in_features, out_features=out_features, num_nodes=num_nodes, **params)
        model = model_obj(net, **model_params)
    else:
        net = tt.practical.MLPVanilla(
            in_features=in_features, out_features=out_features, num_nodes=num_nodes, **params)
        model = model_obj(net)
    model.load_net(os.path.join(path, filename))

    return model
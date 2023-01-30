import itertools
import pandas as pd
import numpy as np
import datetime


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


class LRScheduler(Callback):
    '''Wrapper for pytorch.optim.lr_scheduler objects.
    Parameters:
        scheduler: A pytorch.optim.lr_scheduler object.
       get_score: Function that returns a score when called.
    '''
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self):
        score = self.get_last_score()
        self.scheduler.step(score)
        stop_signal = False
        return stop_signal
    
    def get_last_score(self):
        return self.model.val_metrics.scores['loss']['score'][-1]
    
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
                print('concordance:', concordance)

    
    
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
    
    # finding the best learning rate from this model
    #lrfinder = model.lr_finder(train[0], train[1], batch_size, tolerance=tolerance)
    #lr = lrfinder.get_best_lr()
    
    # Decay LR by a factor of 0.1 every 7 epochs
    model.optimizer.set_lr(lr)
    
    #if print_lr:
        #lrfinder_df = lrfinder.to_pandas()
        #fig = px.line(x=lrfinder_df.index, y=lrfinder_df.train_loss, 
                      #log_x=True, width=700, height=400)

        #fig.update_layout(dict(xaxis={'title':'lr'}, yaxis={'title':'batch_loss'}))
        #fig.show()

    #print("Best learning rate: ", lr)
    
    
    #lambda1 = lambda epoch: 0.9 ** (epoch // 10)
    #lr_scheduler = LambdaLR(opt(model.net.parameters(), lr=0.01), lr_lambda=[lambda1])
    #lr_scheduler = StepLR(opt(model.net.parameters(), lr=0.01), step_size=100, gamma=0.5)
    #lr_scheduler = ReduceLROnPlateau(opt(model.net.parameters(), lr=0.01), mode='min', factor=0.8, patience=5, verbose=True)

    callbacks = [
        tt.callbacks.EarlyStopping(patience=15), 
        #LRScheduler(lr_scheduler), 
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
    if not(discrete):
        #_ = model.compute_baseline_hazards()
        surv = model.predict_surv_df(test[0])
    else:
        surv = model.interpolate(10).predict_surv_df(test[0])
        
    ev = EvalSurv(surv, test[1][0], test[1][1])
    score = ev.concordance_td()
        
    return logs_df, model, score

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

            logs_df, model, score = train_deep_surv(train, val, test, model_obj,out_features, 
                                      print_lr=False, print_logs=False, verbose = True, **params)

            result[i] = {}
            for k, v in params_.items():
                result[i][k] = v
            result[i]['lr'] = model.optimizer.param_groups[0]['lr']
            result[i]['score'] = score

            print('Current score: {} vs. best score: {}'.format(score, best_score))

            if best_score < score:
                best_score = score
                best_model = model
    
    except KeyboardInterrupt:
        pass
        
    table = pd.DataFrame.from_dict(result, orient='index')
    
    return best_model, table.sort_values(by="score", ascending=False)
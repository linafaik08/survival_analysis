import itertools
import pandas as pd
import numpy as np
import datetime
import time

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

from sksurv.metrics import concordance_index_censored
from sksurv.metrics import cumulative_dynamic_auc

def compute_score(censored, target, prediction, sign):
    return concordance_index_censored(list(censored.astype(bool)), target, sign*prediction)[0]

def compute_score_model(preds_df, col_var, col_pred, col_target, sign, metric="cindex", times = None):
    
    scores = {}
    
    for k in preds_df[col_var].unique():
    
        tmp = preds_df[preds_df[col_var]==k]
        scores[k] = compute_score(tmp.censored, tmp[col_target], tmp[col_pred], sign)
        
    scores = pd.DataFrame.from_dict(scores, orient='index').reset_index().rename(
        columns={'index':col_var, 0:col_pred})
        
    return scores

def get_distrib(data, col_var, name):
    
    cols_x = [c for c in data.columns if c !=col_var]
    
    distrib = data.groupby(col_var,as_index=False)[cols_x[0]].count()
    
    distrib[f'perc_{name}'] = distrib[cols_x[0]]/data.shape[0]*100
    distrib.drop(cols_x[0], axis=1, inplace=True)
    
    return distrib
    
def plot_score(scores_df, col_var, models_name):

    scores_graph = pd.melt(
        scores_df[[col_var]+models_name], 
        id_vars=[col_var], value_name='score', var_name='model')

    scores_graph[col_var] = scores_graph[col_var].astype(str)
    scores_graph['score_round'] = scores_graph.score.round(3).astype('str')
    
    fig = px.bar(
        scores_graph, x='model', y='score', 
        color = col_var, barmode='group',
        text = 'score_round',
        color_discrete_sequence = ['royalblue','lightgrey']
    )

    fig.update_traces(textposition='outside')

    fig.update_layout(
        dict(
            title = "{} - {}% of positive classes".format(
                col_var.capitalize(), 
                round(scores_df[scores_df[col_var]==1]['perc_train'].iloc[0])
            ),
            xaxis={'title' : 'Model'}, 
            yaxis={'title' : 'Concordance index', 'range': [0,1]},
        )
    )
    
    return fig
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib
from method import *
import copy
def get_one_data_set(D, i_exp, get_po: bool = True):
    """ Get data for one experiment. Adapted from https://github.com/clinicalml/cfrnet"""
    D_exp = {}
    D_exp['X'] = D['X'][:, :, i_exp]
    D_exp['w'] = D['w'][:, i_exp]
    D_exp['y'] = D['y'][:, i_exp]
    if D['HAVE_TRUTH']:
        D_exp['ycf'] = D['ycf'][:, i_exp]
    else:
        D_exp['ycf'] = None

    if get_po:
        D_exp['mu0'] = D['mu0'][:, i_exp]
        D_exp['mu1'] = D['mu1'][:, i_exp]

    return D_exp

def load_data_npz(fname, get_po: bool = True):
    """ Load data set (adapted from https://github.com/clinicalml/cfrnet)"""
    if fname[-3:] == 'npz':
        data_in = np.load(fname)
        data = {'X': data_in['x'], 'w': data_in['t'], 'y': data_in['yf']}
        try:
            data['ycf'] = data_in['ycf']
        except:
            data['ycf'] = None
    else:
        raise ValueError('This loading function is only for npz files.')

    if get_po:
        data['mu0'] = data_in['mu0']
        data['mu1'] = data_in['mu1']

    data['HAVE_TRUTH'] = not data['ycf'] is None
    data['dim'] = data['X'].shape[1]
    data['n'] = data['X'].shape[0]

    return data
def prepare_ihdp_data(data_train, data_test, setting='original', return_ytest=True):
    if setting == 'original':
        X, y, w, mu0, mu1 = data_train['X'], data_train['y'], data_train['w'], data_train['mu0'], \
                            data_train['mu1']

        X_t, y_t, w_t, mu0_t, mu1_t = data_test['X'], data_test['y'], data_test['w'], \
                                      data_test['mu0'], data_test['mu1']

    elif setting == 'modified':
        X, y, w, mu0, mu1 = data_train['X'], data_train['y'], data_train['w'], data_train['mu0'], \
                            data_train['mu1']

        X_t, y_t, w_t, mu0_t, mu1_t = data_test['X'], data_test['y'], data_test['w'], \
                                      data_test['mu0'], data_test['mu1']
        y[w == 1] = y[w == 1] + mu0[w == 1]
        mu1 = mu0 + mu1
        mu1_t = mu0_t + mu1_t
    else:
        raise ValueError('Setting should in [original, modified]')

    cate = mu1 - mu0
    cate_t = mu1_t - mu0_t

    if return_ytest:
        return X, y, w, mu0, mu1, cate, X_t, y_t, w_t, mu0_t, mu1_t, cate_t

    return X, y, w, mu0, mu1, cate, X_t, mu0_t, mu1_t, cate_t

def IHDPDataGenerator(train_file, test_file, n_exp, train_ratio, output_dir):

        data_train_full = load_data_npz(train_file, get_po=True)
        data_test_full = load_data_npz(test_file, get_po=True)
        for i_exp in range(0, n_exp):
            data_exp = get_one_data_set(data_train_full, i_exp=i_exp, get_po=True)
            data_exp_test = get_one_data_set(data_test_full, i_exp=i_exp, get_po=True)
            X, y, w, mu0, mu1, cate, X_test, y_test, w_test, mu0_test, mu1_test, cate_test = \
                prepare_ihdp_data(data_exp, data_exp_test, setting='original')

            X_train, X_val, w_train, w_val, y_train, y_val, mu0_train, mu0_val, mu1_train, \
            mu1_val, cate_train, cate_val = train_test_split(X, w, y, mu0, mu1, cate, train_size=train_ratio)
            data_train = {}
            data_val = {}
            data_test = {}
            data_train['x']=X_train; data_train['y']=y_train.squeeze(); data_train['t']=w_train.squeeze(); 
            data_train['mu0']=mu0_train.squeeze(); data_train['mu1']=mu1_train.squeeze(); data_train['cate']=cate_train.squeeze();
            
            data_val['x']=X_val; data_val['y']=y_val.squeeze(); data_val['t']=w_val.squeeze(); 
            data_val['mu0']=mu0_val.squeeze(); data_val['mu1']=mu1_val.squeeze(); data_val['cate']=cate_val.squeeze();
            
            data_test['x']=X_test; data_test['y']=y_test.squeeze(); data_test['t']=w_test.squeeze(); 
            data_test['mu0']=mu0_test.squeeze(); data_test['mu1']=mu1_test.squeeze(); data_test['cate']=cate_test.squeeze();
            
            
            # data_train = X_train, y_train.squeeze(), w_train.squeeze(), mu0_train.squeeze(), \
            #              mu1_train.squeeze(), cate_train.squeeze(), None
            # data_val = X_val, y_val.squeeze(), w_val.squeeze(), mu0_val.squeeze(), \
            #            mu1_val.squeeze(), cate_val.squeeze(), None
            # data_test = X_test, y_test.squeeze(), w_test.squeeze(), mu_0_test.squeeze(), \
            #             mu_1_test.squeeze(), cate_test.squeeze(), None
            joblib.dump([data_train, data_val, data_test], output_dir+'/'+'ihdp_'+str(i_exp))
        # return data_train, data_val, data_test

def prepare_for_learner(basemodel_name, data):
    treat_idx = data['t']==1
    control_idx = data['t']==0
    mu1_hat = get_model(basemodel_name, 'reg', data['x'][treat_idx, :], data['y'][treat_idx])
    mu0_hat = get_model(basemodel_name, 'reg', data['x'][control_idx, :], data['y'][control_idx])
    y1_pre = mu1_hat.predict(data['x'])
    y0_pre = mu0_hat.predict(data['x'])
    return mu1_hat, mu0_hat, y1_pre, y0_pre

def initial_necessary_dict(train_base_model_list, train_learner_list, val_base_model_list, val_learner_list, pseudo_estimator_list, other_scorer_list):
    hat_base_model = {}; hat_learner = {}; tilde_base_model = {}; tilde_learner = {}; pehe_all_learner = {}; scorer = {}
    for modelname in train_base_model_list:
        hat_base_model[modelname] = {}
        hat_learner[modelname] = {}
        for learnername in train_learner_list:
            hat_learner[modelname][learnername] = None
    for modelname in val_base_model_list:
        tilde_base_model[modelname] = {}
        tilde_learner[modelname] = {}
        # plug-in scorer
        for learnername in val_learner_list:
            tilde_learner[modelname][learnername] = None
            scorer['plug-'+modelname + '_' +learnername] = {}
        # pseudo scorer
        for estimatorname in pseudo_estimator_list:
            scorer['pseudo-' + modelname + '_' + estimatorname] = {}
    # other scorers
    for scorername in other_scorer_list:
        scorer[scorername] = {}

    # metric_dicts
    # pehe of all learners
    for modelname in train_base_model_list:
        for learnername in train_learner_list:
            pehe_all_learner[modelname + '_' +learnername] = None
    # scorer and scorer_metric
    scorer_metric = copy.deepcopy(scorer)
    for scorername in scorer.keys():
        for hatlearner in pehe_all_learner.keys():
            scorer[scorername][hatlearner] = None
    return hat_base_model, hat_learner, tilde_base_model, tilde_learner, pehe_all_learner, scorer, scorer_metric
def clip_propensity(pi_x):
    pi_x[pi_x>0.999]=0.999
    pi_x[pi_x<0.001] = 0.001
    return pi_x

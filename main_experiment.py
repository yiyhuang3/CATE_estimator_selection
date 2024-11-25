from util import *
import os
import joblib
from method import *
from scorer import *
from compute import *
from scipy.stats import spearmanr
import copy
def load_data(data_dir, i_exp):
    [data_train, data_val, data_test] = joblib.load(data_dir +'/' + 'acic_'+str(i_exp))
    return data_train, data_val, data_test
def run_experiment(M, data_dir, result_dir, train_base_model_list, train_learner_list, val_base_model_list, val_learner_list, pseudo_estimator_list, other_scorer_list):
    for i_exp in range(0, M):
        hat_base_model, hat_learner, tilde_base_model, tilde_learner, pehe_all_learner, scorer, scorer_metric =  \
            initial_necessary_dict(train_base_model_list, train_learner_list, val_base_model_list, val_learner_list, pseudo_estimator_list, other_scorer_list)
        data_train, data_val, data_test = load_data(data_dir=data_dir, i_exp=i_exp)
        '''train on training set'''
        for basemodel_name in train_base_model_list:
            '''get base models'''
            classifier_pi = get_model(model_name=basemodel_name, reg_or_cal='cla', x=data_train['x'], y=data_train['t'])
            pi_x_hat = classifier_pi.fit(data_train['x'], data_train['t'])
            hat_base_model[basemodel_name]['pi'] = pi_x_hat
            pi_x_pre = clip_propensity(hat_base_model[basemodel_name]['pi'].predict_proba(data_train['x'])[:, 1])
            '''get hat_learners'''
            mu1_hat, mu0_hat, y1_pre, y0_pre = prepare_for_learner(basemodel_name=basemodel_name, data=data_train)
            hat_base_model[basemodel_name]['mu1'] = mu1_hat
            hat_base_model[basemodel_name]['mu0'] = mu0_hat
            for learner_name in train_learner_list:
                hat_learner[basemodel_name][learner_name] = get_learner(hat_learner=hat_learner,learner_name=learner_name,
                                                                        hat_base_model=hat_base_model,basemodel_name=basemodel_name,
                                                                        mu1_hat=mu1_hat, mu0_hat=mu0_hat, y1_pre=y1_pre,
                                                                        y0_pre=y0_pre, pi_x_pre=pi_x_pre,
                                                                        data=data_train)


        '''evaluate on validation set'''
        '''train tilde_learner (plug-in scorer) on validation set'''
        for basemodel_name in val_base_model_list:
            classifier_pi = get_model(model_name=basemodel_name, reg_or_cal='cla', x=data_val['x'], y=data_val['t'])
            pi_x_tilde = classifier_pi.fit(data_val['x'], data_val['t'])
            tilde_base_model[basemodel_name]['pi'] = pi_x_tilde
            pi_x_pre = clip_propensity(tilde_base_model[basemodel_name]['pi'].predict_proba(data_val['x'])[:, 1])
            '''get tilde_learners'''
            mu1_tilde, mu0_tilde, y1_pre, y0_pre = prepare_for_learner(basemodel_name=basemodel_name, data=data_val)
            tilde_base_model[basemodel_name]['mu1'] = mu1_tilde
            tilde_base_model[basemodel_name]['mu0'] = mu0_tilde
            tilde_base_model[basemodel_name]['mu'] = get_model(basemodel_name, 'reg', data_val['x'], data_val['y'])
            for learner_name in val_learner_list:
                tilde_learner[basemodel_name][learner_name] = get_learner(hat_learner=tilde_learner, learner_name=learner_name,
                                                                        hat_base_model=tilde_base_model,basemodel_name=basemodel_name,
                                                                        mu1_hat=mu1_tilde, mu0_hat=mu0_tilde, y1_pre=y1_pre,
                                                                        y0_pre=y0_pre, pi_x_pre=pi_x_pre,
                                                                        data=data_val)

        # save tau_hat_x on validation set
        tau_hat_x = {}
        for modelname in train_base_model_list:
            for learnername in train_learner_list:
                hatlearner = hat_learner[modelname][learnername]
                tau_hat_x[modelname + '_' + learnername] = hatlearner.predict(data_val['x']).squeeze()



        '''get_scorer'''
        '''KL scorer'''
        if 'KL' in other_scorer_list:
            update_KL_scorer(scorer=scorer, tau_hat_x=tau_hat_x, data=data_val,
                             kl_paras={'lamda0':100, 'tol':0.00001, 'max_iter':2000, 'learning_rate':0.001})
        if 'fact' in other_scorer_list:
            for learner_name in train_learner_list:
                for basemodel_name in train_base_model_list:
                    if learner_name in ['S', 'T']:
                        y1_hat, y0_hat = hat_learner[basemodel_name][learner_name].predict_POs(data_val['x'])
                        y_f_hat = data_val['t'] * y1_hat.squeeze() + (1 - data_val['t']) * y0_hat.squeeze()
                        scorer['fact'][basemodel_name+'_'+learner_name] = np.sqrt(np.mean((y_f_hat - data_val['y']) ** 2))
                    else:
                        scorer['fact'][basemodel_name + '_' + learner_name] = np.inf
        if 'random' in other_scorer_list:
            for basemodel_name in train_base_model_list:
                for learner_name in train_learner_list:
                    scorer['random'][basemodel_name + '_' + learner_name] = 1
            selected_learner = np.random.choice(list(scorer['random'].keys()))
            scorer['random'][selected_learner] = 0
        if 'knn' in other_scorer_list:
            get_knn_score(scorer=scorer, tau_hat_x=tau_hat_x, data=data_val)
        '''plug-in scorer and pseudo scorer'''
        update_plug_scorer(scorer=scorer, tilde_learner=tilde_learner, val_base_model_list=val_base_model_list, val_learner_list=val_learner_list,
                           tau_hat_x=tau_hat_x, data=data_val)
        update_pseudo_scorer(scorer=scorer, pseudo_estimator_list=pseudo_estimator_list, val_base_model_list=val_base_model_list,
                           tilde_base_model=tilde_base_model, tau_hat_x=tau_hat_x, data=data_val)

        '''Compute test sample performance'''
        update_pehe_all_learner(pehe_all_learner=pehe_all_learner, hat_learner=hat_learner, data=data_test)
        joblib.dump([pehe_all_learner, scorer, scorer_metric], result_dir + '/scorer_metric_' + str(i_exp))

        label_selected_learner(scorer=scorer, scorer_metric=scorer_metric)
        compute_regret_all_scorer(pehe_all_learner=pehe_all_learner, scorer_metric=scorer_metric)
        compute_corrlation(pehe_all_learner=pehe_all_learner, scorer=scorer, scorer_metric=scorer_metric)


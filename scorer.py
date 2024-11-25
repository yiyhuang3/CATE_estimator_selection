import numpy as np
from util import *
from scipy.stats import kendalltau, spearmanr
from KL_scorer import *


def get_knn_score(scorer, tau_hat_x, data):
    x = data['x']
    t = data['t']
    y = data['y']
    tau_tilde = np.ones((x.shape[0],))
    x_treat = x[t == 1, :]
    y_treat = y[t == 1]
    x_control = x[t == 0, :]
    y_control = y[t == 0]
    for i in range(x.shape[0]):
        if t[i] == 1:
            dist = np.sum((x_control - x[i, :]) ** 2, axis=1)
            match_idx = np.argmin(dist)
            tau_tilde[i] = y[i] - y_control[match_idx]
        if t[i] == 0:
            dist = np.sum((x_treat - x[i, :]) ** 2, axis=1)
            match_idx = np.argmin(dist)
            tau_tilde[i] = y_treat[match_idx] - y[i]
    for hat_learner_name in tau_hat_x.keys():
        tau_hat = tau_hat_x[hat_learner_name]
        scorer['knn'][hat_learner_name] = np.sqrt(np.mean((tau_hat - tau_tilde) ** 2))
def update_plug_scorer(scorer, tilde_learner, val_base_model_list, val_learner_list, tau_hat_x, data):
    x = data['x']
    for scorer_basemodel_name in val_base_model_list:
        # scorer for plug-in method
        for scorer_learner_name in val_learner_list:
            tau_tilde = tilde_learner[scorer_basemodel_name][scorer_learner_name].predict(x).squeeze()
            for hat_learner_name in tau_hat_x.keys():
                tau_hat = tau_hat_x[hat_learner_name]
                scorer['plug-' + scorer_basemodel_name +'_'+scorer_learner_name][hat_learner_name] = np.sqrt(np.mean((tau_hat - tau_tilde) ** 2))


def update_pseudo_scorer(scorer, pseudo_estimator_list, val_base_model_list, tilde_base_model, tau_hat_x, data):
    x = data['x']
    t = data['t']
    y = data['y']
    for scorer_basemodel_name in val_base_model_list:
        pi_x_tilde = clip_propensity(tilde_base_model[scorer_basemodel_name]['pi'].predict_proba(x)[:, 1])
        for scorer_learner_name in pseudo_estimator_list:
            y1_tilde = tilde_base_model[scorer_basemodel_name]['mu1'].predict(x).squeeze()
            y0_tilde = tilde_base_model[scorer_basemodel_name]['mu0'].predict(x).squeeze()
            if scorer_learner_name == 'DR':
                y1_pseudo = y1_tilde + t / pi_x_tilde * (y - y1_tilde)
                y0_pseudo = y0_tilde + (1 - t) / (1 - pi_x_tilde) * (y - y0_tilde)
                y_pseudo = y1_pseudo - y0_pseudo
                for hat_learner_name in tau_hat_x.keys():
                    tau_hat = tau_hat_x[hat_learner_name]
                    scorer['pseudo-' + scorer_basemodel_name+'_'+scorer_learner_name][hat_learner_name] = np.sqrt(np.mean((tau_hat - y_pseudo) ** 2))
            elif scorer_learner_name == 'R':
                mu_x_tilde = tilde_base_model[scorer_basemodel_name]['mu'].predict(x).squeeze()
                residual_t_tilde = t - pi_x_tilde
                for hat_learner_name in tau_hat_x.keys():
                    tau_hat = tau_hat_x[hat_learner_name]
                    scorer['pseudo-' + scorer_basemodel_name+'_'+scorer_learner_name][hat_learner_name] = np.sqrt(np.mean((y-mu_x_tilde-tau_hat*residual_t_tilde)**2))
            elif scorer_learner_name == 'IF':
                A = t - pi_x_tilde
                C = pi_x_tilde * (1 - pi_x_tilde)
                B = 2 * t * A / C
                tau_tilde = y1_tilde - y0_tilde
                for hat_learner_name in tau_hat_x.keys():
                    tau_hat = tau_hat_x[hat_learner_name]
                    IF_score = np.mean((1 - B) * tau_tilde ** 2 + B * y * (tau_tilde - tau_hat) - A * (
                            tau_tilde - tau_hat) ** 2 + tau_hat ** 2)
                    scorer['pseudo-' + scorer_basemodel_name + '_' + scorer_learner_name][hat_learner_name] = IF_score

def update_KL_scorer(scorer, tau_hat_x, data, kl_paras):
    y = data['y']
    n = y.shape[0]
    treat_idx = data['t']==1
    control_idx = data['t']==0
    y_treat = y[treat_idx]
    y_control = y[control_idx]
    p_t = y_treat.shape[0]/y.shape[0]
    p_c = y_control.shape[0]/y.shape[0]
    x_t = data['x'][treat_idx]
    x_c = data['x'][control_idx]
    epsilon = kl_nn(s1=x_c, s2=x_t, k=3)
    for hat_learner_name in tau_hat_x.keys():
        tau_hat = tau_hat_x[hat_learner_name]
        tau_mu0 = tau_hat*data['mu0']
        tau_mu1 = tau_hat * data['mu1']
        V0_true = np.mean(tau_mu0[treat_idx])
        V1_true = np.mean(-tau_mu1[control_idx])
        z1 = -tau_hat * y
        z1 = z1[treat_idx]
        z0 = tau_hat * y
        z0 = z0[control_idx]
        lamda1_hat, V1_hat = estimate_V(z=z1, epsilon=epsilon, kl_paras=kl_paras)
        lamda0_hat, V0_hat = estimate_V(z=z0, epsilon=epsilon, kl_paras=kl_paras)
        KL_hat = np.mean(tau_hat**2) + 2/n * (np.sum(z1)+np.sum(z0)) + 2*(p_c*V1_hat + p_t*V0_hat)
        scorer['KL'][hat_learner_name] = KL_hat

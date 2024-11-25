import numpy as np
from util import *
from scipy.stats import kendalltau, spearmanr
def label_selected_learner(scorer, scorer_metric):
    for scorer_key in scorer.keys():
        target_scorer = scorer[scorer_key]
        scorer_metric[scorer_key]['best'] = min(target_scorer, key=target_scorer.get)
def update_pehe_all_learner(pehe_all_learner, hat_learner, data):
    x = data['x']
    true_cate = data['cate']
    for modelname in hat_learner.keys():
        for learnername in hat_learner[modelname]:
            hatlearner = hat_learner[modelname][learnername]
            estimate_cate = hatlearner.predict(x).squeeze()
            pehe = np.sqrt(np.mean((estimate_cate-true_cate)**2))
            pehe_all_learner[modelname + '_' + learnername] = pehe

def compute_regret_all_scorer(pehe_all_learner, scorer_metric):
    n_learner = len(pehe_all_learner)
    learner_rank = sorted(pehe_all_learner.items(), key=lambda x:x[1])
    best_learner = learner_rank[0][0]
    best_pehe = pehe_all_learner[best_learner]
    second_best_learner = learner_rank[1][0]
    second_best_pehe = pehe_all_learner[second_best_learner]
    worst_learner = learner_rank[-1][0]
    worst_pehe = pehe_all_learner[worst_learner]
    second_worst_learner = learner_rank[-2][0]
    second_worst_pehe = pehe_all_learner[second_worst_learner]
    for scorer_key in scorer_metric.keys():
        selected_estimator = scorer_metric[scorer_key]['best']
        selected_pehe = pehe_all_learner[selected_estimator]
        regret = (selected_pehe - best_pehe) / best_pehe
        scorer_metric[scorer_key]['pehe'] = selected_pehe
        scorer_metric[scorer_key]['abs_regret'] = selected_pehe - best_pehe
        scorer_metric[scorer_key]['regret'] = regret
    scorer_metric['oracle'] = {}
    scorer_metric['oracle']['pehe'] = best_pehe
    scorer_metric['oracle']['true_rank'] = learner_rank
    scorer_metric['oracle']['best'] = best_learner
    scorer_metric['oracle']['second_best'] = second_best_learner
    scorer_metric['oracle']['second_best_regret'] = (second_best_pehe - best_pehe)/best_pehe
    scorer_metric['oracle']['worst'] = worst_learner
    scorer_metric['oracle']['worst_regret'] = (worst_pehe - best_pehe)/best_pehe
    scorer_metric['oracle']['second_worst'] = second_worst_learner
    scorer_metric['oracle']['second_worst_regret'] = (second_worst_pehe - best_pehe)/best_pehe

    for scorer_key in scorer_metric.keys():
        if scorer_key=='oracle':
            continue
        for i in range(0, n_learner):
            scorer_metric[scorer_key][str(i)] = 0
        for i in range(0, n_learner):
            if scorer_metric[scorer_key]['best'] == learner_rank[i][0]:
                scorer_metric[scorer_key][str(i)] = 1


def compute_corrlation(pehe_all_learner, scorer, scorer_metric):
    true_rank_list = list(dict(sorted(pehe_all_learner.items(), key=lambda x: x[1])).keys())
    true_rank = {}
    for train_learner in pehe_all_learner.keys():
        true_rank[train_learner] = true_rank_list.index(train_learner)
    true_rank_value_list = list(true_rank.values())
    for scorer_key in scorer.keys():
        scorer_rank_dict = {}
        rank_list = list(dict(sorted(scorer[scorer_key].items(), key=lambda x: x[1])).keys())
        for train_learner in pehe_all_learner.keys():
            scorer_rank_dict[train_learner] = rank_list.index(train_learner)
        scorer_rank_list = list(scorer_rank_dict.values())
        s_cor, _ = spearmanr(true_rank_value_list, scorer_rank_list)
        k_cor, _ = kendalltau(true_rank_value_list, scorer_rank_list)
        scorer_metric[scorer_key]['s_cor'] = s_cor
        scorer_metric[scorer_key]['k_cor'] = k_cor

def label_winlose(pehe_all_learner, scorer_metric):
    best_learner = min(pehe_all_learner, key=pehe_all_learner.get)
    worst_learner = max(pehe_all_learner, key=pehe_all_learner.get)
    for scorer_key in scorer_metric.keys():
        if scorer_key=='oracle':
            continue
        scorer_metric[scorer_key]['win'] = 0
        scorer_metric[scorer_key]['lose'] = 0
        if scorer_metric[scorer_key]['best'] == best_learner:
            scorer_metric[scorer_key]['win'] = 1
        if scorer_metric[scorer_key]['best'] == worst_learner:
            scorer_metric[scorer_key]['lose'] = 1
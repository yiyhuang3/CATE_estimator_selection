from main_experiment import *
from ACIC_generate import *

def generate_datasets(data_dir, exp_setting, M):
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    for nonlinearity_y in exp_setting['nonlinearity_y_list']:
        for rho in exp_setting['rho_list']:
            for xi in exp_setting['xi_list']:
                for mis_ratio in exp_setting['mis_ratio_list']:
                    generation_dir = './data/'+str(nonlinearity_y) + '_' + str(rho)+ '_' + str(xi) +'_'+ str(mis_ratio)
                    if not os.path.isdir(generation_dir):
                        os.mkdir(generation_dir)
                    acic_generate(data_dir=data_dir, output_dir=generation_dir, xi=xi, rho=rho, n_exp=M,
                                  test_ratio=0.3, val_ratio=0.3, nonlinearity_y=nonlinearity_y, mis_ratio=mis_ratio)

def run_all_exps(M, exp_setting, train_base_model_list, train_learner_list, val_base_model_list, val_learner_list,
                 pseudo_estimator_list, other_scorer_list):
    if not os.path.isdir('./result'):
        os.mkdir('./result')
    for nonlinearity_y in exp_setting['nonlinearity_y_list']:
        for rho in exp_setting['rho_list']:
            for xi in exp_setting['xi_list']:
                for mis_ratio in exp_setting['mis_ratio_list']:
                    data_dir ='./data/'+str(nonlinearity_y)+ '_' + str(rho)+ '_' + str(xi) + '_'+str(mis_ratio)
                    target_result_dir = './result/'+str(nonlinearity_y)+ '_' + str(rho)+ '_' + str(xi) + '_'+str(mis_ratio)
                    if not os.path.isdir(target_result_dir):
                        os.mkdir(target_result_dir)
                    run_experiment(M, data_dir, target_result_dir, train_base_model_list,
                                   train_learner_list, val_base_model_list, val_learner_list, pseudo_estimator_list, other_scorer_list)
if __name__ == '__main__':
    train_base_model_list = ['lr', 'rf', 'svm', 'net']
    train_learner_list = ['U', 'S', 'PS', 'T', 'X', 'IPW', 'DR', 'R', 'RA']

    val_base_model_list = ['xgb']
    val_learner_list = ['U', 'S', 'PS', 'T', 'X', 'IPW', 'DR', 'R', 'RA']

    pseudo_estimator_list = ['DR', 'R', 'IF']
    other_scorer_list = ['random', 'fact', 'knn', 'KL']

    exp_setting = {'nonlinearity_y_list': [3], 'rho_list':[0, 0.1, 0.3], 'xi_list':[1], 'mis_ratio_list':[0]}

    M = 100
    generate_datasets(data_dir=r'./ACIC2016_data', exp_setting=exp_setting, M=M)
    run_all_exps(M, exp_setting, train_base_model_list, train_learner_list, val_base_model_list, val_learner_list,
                 pseudo_estimator_list, other_scorer_list)
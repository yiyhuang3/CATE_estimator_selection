from result_analyze import *
import pandas as pd
import os
if __name__ == '__main__':
    for train_base_model_list in [['lr','rf', 'svm', 'net']]:
        train_learner_list = ['U', 'S', 'PS', 'T', 'X', 'IPW', 'DR', 'R', 'RA']

        M = 100
        settings = []

        nonlinearity_y_list = [3]
        for nonlinearity_y in nonlinearity_y_list:
            # for rho, xi, mis_ratio in [[0.1, 1, 0]]:
            for rho, xi, mis_ratio in [[0, 1, 0], [0.1, 1, 0], [0.3, 1, 0], [0.1, 0, 0], [0.1, 2, 0]]:
                settings.append((nonlinearity_y, rho, xi, mis_ratio))

        regret_sum_df = pd.DataFrame()
        cor_sum_df = pd.DataFrame()
        if not os.path.isdir('./result_new'):
            os.mkdir('./result_new')
        if not os.path.isdir('./result_summary'):
            os.mkdir('./result_summary')
        result_dir_new_0 = './result_new/' + '_'.join(train_base_model_list)
        result_summary_dir = './result_summary/' + '_'.join(train_base_model_list)
        if not os.path.isdir(result_dir_new_0):
            os.mkdir(result_dir_new_0)
        if not os.path.isdir(result_summary_dir):
            os.mkdir(result_summary_dir)
        for (nonlinearity_y, rho, xi, mis_ratio) in settings:
            result_dir = './result/'+ str(nonlinearity_y) + '_' + str(rho) + '_' + str(xi)+ '_' + str(mis_ratio)
            result_dir_new = result_dir_new_0 + '/'+ str(nonlinearity_y) + '_' + str(rho) + '_' + str(xi)+ '_' + str(mis_ratio)
            if not os.path.isdir(result_dir_new):
                os.mkdir(result_dir_new)
            learner_rank_df, result_df = analyze_one_result(result_dir, result_dir_new, train_base_model_list, train_learner_list, M)
            learner_rank_df.to_csv(result_summary_dir + '/learner_rank_'+ str(nonlinearity_y) + '_' + str(rho) + '_' + str(xi)+ '_' + str(mis_ratio) + '.csv')
            result_df.to_csv(result_summary_dir + '/result_' + str(nonlinearity_y) + '_' + str(rho) + '_' + str(xi)+ '_' + str(mis_ratio) + '.csv')
            if regret_sum_df.index.empty or cor_sum_df.index.empty:
                regret_sum_df.index = result_df.index
                cor_sum_df.index = result_df.index
            regret_sum_df[str(nonlinearity_y) + '_' + str(rho) + '_' + str(xi)+ '_' + str(mis_ratio)+'_abs_mean'] = result_df['abs_regret_mean']
            regret_sum_df[str(nonlinearity_y) + '_' + str(rho) + '_' + str(xi) + '_' + str(
                    mis_ratio) + '_abs_std'] = result_df['abs_regret_std']
            cor_sum_df[str(nonlinearity_y) + '_' + str(rho) + '_' + str(xi)+ '_' + str(mis_ratio)+'_scor_mean'] = result_df['s_cor_mean']
            cor_sum_df[str(nonlinearity_y) + '_' + str(rho) + '_' + str(xi)+ '_' + str(mis_ratio) + '_scor_std'] = result_df['s_cor_std']
            cor_sum_df[str(nonlinearity_y) + '_' + str(rho) + '_' + str(xi)+ '_' + str(mis_ratio) + '_kcor_mean'] = result_df['k_cor_mean']
            cor_sum_df[str(nonlinearity_y) + '_' + str(rho) + '_' + str(xi)+ '_' + str(mis_ratio) + '_kcor_std'] = result_df['k_cor_std']

        regret_sum_df = regret_sum_df.style.highlight_min(color='yellow', axis=0)
        regret_sum_df.to_excel(result_summary_dir + '/regret_summary.xlsx', engine='openpyxl')
        cor_sum_df = cor_sum_df.style.highlight_max(color='green', axis=0)
        cor_sum_df.to_excel(result_summary_dir + '/cor_summary.xlsx', engine='openpyxl')
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from KL_scorer import *
def sigmoid(x, beta_for_T, xi):
    return 1/(1+np.exp(-xi*(np.dot(x, beta_for_T) + 3)))

def get_acic_covariates(data_dir):
    X = pd.read_csv(data_dir + '/covariates.csv')
    NUMERIC_COLS = [0, 3, 4, 16, 17, 18, 20, 21, 22, 23, 24, 24, 25, 30, 31, 32, 33, 39, 40, 41, 53, 54]
    X = X.drop(columns=['x_2', 'x_21', 'x_24'])
    feature_list = []
    for cols_ in X.columns:
        if type(X.loc[X.index[0], cols_]) not in [np.int64, np.float64]:

            enc = OneHotEncoder(drop='first')

            enc.fit(np.array(X[[cols_]]).reshape((-1, 1)))

            for k in range(len(list(enc.get_feature_names()))):
                X[cols_ + list(enc.get_feature_names())[k]] = enc.transform(
                    np.array(X[[cols_]]).reshape((-1, 1))).toarray()[:, k]

            feature_list.append(cols_)

        X.drop(feature_list, axis=1, inplace=True)

    X = X.iloc[:, NUMERIC_COLS]
    scaler = StandardScaler()
    X_t = scaler.fit_transform(X)
    return X_t

def generate_inner(x, nonlinearity_y):
    d = x.shape[1]
    beta_2_d = 0
    beta_3_d = 0
    beta_4_d = 0
    inner_1 = x
    inner_2_list = []
    inner_3_list = []
    inner_4_list = []
    for i in range(0, d):
        for j in range(i, d):
            inner_2_list.append(x[:, i] * x[:, j])
            beta_2_d = beta_2_d + 1
    for i in range(0, d):
        for j in range(i, d):
            for k in range(j, d):
                inner_3_list.append(x[:, i] * x[:, j]*x[:,k])
                beta_3_d = beta_3_d + 1
    for i in range(0, d):
        for j in range(i, d):
            for k in range(j, d):
                for l in range(k, d):
                    inner_4_list.append(x[:, i] * x[:, j]*x[:,k])
                    beta_4_d = beta_4_d + 1
    inner_2 = np.array(inner_2_list).T
    inner_3 = np.array(inner_3_list).T
    inner_4 = np.array(inner_4_list).T
    if nonlinearity_y==1:
        X_for_Y = inner_1
    elif nonlinearity_y==2:
        X_for_Y = np.concatenate((inner_1, inner_2), axis=1)
    elif nonlinearity_y==3:
        X_for_Y = np.concatenate((inner_1, inner_2, inner_3), axis=1)
    elif nonlinearity_y==4:
        X_for_Y = np.concatenate((inner_1, inner_2, inner_3, inner_4), axis=1)
    return X_for_Y, inner_1, inner_2, inner_3, inner_4
def acic_simulate_one(data_dir, xi, rho, nonlinearity_y, mis_ratio,seed):
    np.random.seed(seed)
    # get data
    X = get_acic_covariates(data_dir=data_dir)
    n = X.shape[0]
    d = X.shape[1]
    X_for_Y, inner_1, inner_2, inner_3, inner_4 = generate_inner(X, nonlinearity_y)

    mis_cols = np.random.randint(0, d, int(d*mis_ratio)).tolist()
    obs_cols = list(set(list(range(0, d))).difference(set(mis_cols)))
    X_obs = X[:, obs_cols]
    beta_for_Y = np.random.binomial(1, 0.2, X_for_Y.shape[1]).reshape(-1, 1)
    beta_for_T = np.random.binomial(1, 0.2, X.shape[1])

    prob_t = sigmoid(x=X, beta_for_T=beta_for_T, xi=xi).squeeze()
    t = np.random.binomial(1, prob_t, n)

    # generate POs
    X_for_tau = X
    ind_rho = np.random.binomial(1, rho, X_for_tau.shape[1])
    mu0 = np.matmul(X_for_Y, beta_for_Y)
    tau_x = np.matmul(X_for_tau, ind_rho.reshape(-1, 1))

    mu1 = mu0 + tau_x
    mu0 = mu0.squeeze()
    mu1 = mu1.squeeze()
    cate = mu1 - mu0
    y0 = mu0 + np.random.normal(0, 0.1, n)
    y1 = mu1 + np.random.normal(0, 0.1, n)
    y = t * y1 + (1-t) * y0

    idx_t = np.where(t==1)
    idx_c = np.where(t==0)
    x_t = X[idx_t, :][0]
    x_c = X[idx_c, :][0]
    epsilon = kl_nn(x_t, x_c, k=3)
    epsilon_1 = kl_nn(np.concatenate((x_t, y1[idx_t].reshape(-1, 1)), axis=1), np.concatenate((x_c, y1[idx_c].reshape(-1, 1)), axis=1), k=3)
    epsilon_2 = kl_nn(np.concatenate((x_t, y0[idx_t].reshape(-1, 1)), axis=1), np.concatenate((x_c, y0[idx_c].reshape(-1, 1)), axis=1), k=3)
    print(epsilon, epsilon_1, epsilon_2)
    return X_obs, t, y, mu0, mu1, cate

def acic_generate(data_dir, output_dir, xi, rho, n_exp, test_ratio, val_ratio, nonlinearity_y, mis_ratio):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i_exp in range(0, n_exp):
        X, t, y, mu0, mu1, cate = acic_simulate_one(data_dir=data_dir, xi=xi, rho=rho, nonlinearity_y=nonlinearity_y, mis_ratio=mis_ratio,seed=i_exp)
        X_train, X_test, t_train, t_test, y_train, y_test, mu0_train, mu0_test, mu1_train, mu1_test, cate_train, cate_test \
            = train_test_split(X, t, y, mu0, mu1, cate, test_size=test_ratio, random_state=0)
        X_train, X_val, t_train, t_val, y_train, y_val, mu0_train, mu0_val, mu1_train, mu1_val, cate_train, cate_val \
            = train_test_split(X_train, t_train, y_train, mu0_train, mu1_train, cate_train, test_size=val_ratio, random_state=0)
        data_train = {}
        data_val = {}
        data_test = {}
        data_train['x'] = X_train;
        data_train['y'] = y_train.squeeze();
        data_train['t'] = t_train.squeeze();
        data_train['mu0'] = mu0_train.squeeze();
        data_train['mu1'] = mu1_train.squeeze();
        data_train['cate'] = cate_train.squeeze();

        data_val['x'] = X_val;
        data_val['y'] = y_val.squeeze();
        data_val['t'] = t_val.squeeze();
        data_val['mu0'] = mu0_val.squeeze();
        data_val['mu1'] = mu1_val.squeeze();
        data_val['cate'] = cate_val.squeeze();

        data_test['x'] = X_test;
        data_test['y'] = y_test.squeeze();
        data_test['t'] = t_test.squeeze();
        data_test['mu0'] = mu0_test.squeeze();
        data_test['mu1'] = mu1_test.squeeze();
        data_test['cate'] = cate_test.squeeze();
        joblib.dump([data_train, data_val, data_test], output_dir + '/' + 'acic_' + str(i_exp))
# if __name__ == '__main__':
#     acic_generate(data_dir='D:\Experiments of all projects\Causal Estimator Selection\my exp\data\ACIC2016',
#                   output_dir='./acic_data/', xi=1, rho=1, n_exp=100, test_ratio=0.3, val_ratio=0.5)
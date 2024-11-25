import abc
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from flaml import AutoML, tune
from flaml.automl.model import SKLearnEstimator
from flaml.automl.task.task import CLASSIFICATION
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
class base_learner(BaseEstimator, RegressorMixin, abc.ABC):
    def __init__(self):
        pass
    def fit(self, x, y, t, pi=None):
        pass
    def predict(self, x):
        pass

class T_learner(BaseEstimator):
    def __init__(self, mu1_model, mu0_model):
        self.mu1_model = mu1_model
        self.mu0_model = mu0_model
    def predict(self, x):
        return self.mu1_model.predict(x) - self.mu0_model.predict(x)
    def predict_POs(self, x):
        return (self.mu1_model.predict(x), self.mu0_model.predict(x))
class S_learner(BaseEstimator):
    def __init__(self, mu_tx_model):
        self.mu_tx_model = mu_tx_model
    def predict(self, x):
        t0 = np.zeros((x.shape[0], 1))
        t1 = np.ones((x.shape[0], 1))
        predictor_0x = np.concatenate((t0, x), axis=1)
        predictor_1x = np.concatenate((t1, x), axis=1)
        return self.mu_tx_model.predict(predictor_1x) - self.mu_tx_model.predict(predictor_0x)
    def predict_POs(self, x):
        t0 = np.zeros((x.shape[0], 1))
        t1 = np.ones((x.shape[0], 1))
        predictor_0x = np.concatenate((t0, x), axis=1)
        predictor_1x = np.concatenate((t1, x), axis=1)
        return (self.mu_tx_model.predict(predictor_1x), self.mu_tx_model.predict(predictor_0x))
class X_learner(BaseEstimator):
    def __init__(self, pi, tau0, tau1):
        self.pi = pi
        self.tau0 = tau0
        self.tau1 = tau1
    def predict(self, x):
        pi_x = self.pi.predict(x)
        return pi_x * self.tau0.predict(x).squeeze() + (1-pi_x) * self.tau1.predict(x).squeeze()

class mySVM(SKLearnEstimator):
    def __init__(self, n_jobs=None, task='classification', **config):
        super().__init__(task, **config)

        if task in CLASSIFICATION:
            self.estimator_class = SVC
        else:
            self.estimator_class = SVR

    @classmethod
    def search_space(cls, data_size, task):
        space = {
            "C": {
                "domain": tune.uniform(lower=1, upper=5),
            },
            "kernel": {
                "domain": tune.choice(['sigmoid', 'rbf', 'linear']),
                "low_cost_init_value": 'linear'
            },
            "max_iter": {
                "domain": tune.randint(lower=100, upper=1000),
                "low_cost_init_value": 100
            }
        }
        if task in CLASSIFICATION:
            space["probability"]={"domain": True}
        return space

def get_learner(hat_learner, learner_name, hat_base_model, basemodel_name, mu1_hat, mu0_hat, y1_pre, y0_pre, pi_x_pre, data):
    x = data['x']
    t = data['t']
    y = data['y']
    learner = None
    if learner_name == 'S':
        tx = np.concatenate((t.reshape(-1, 1), x), axis=1)
        mu_tx = get_model(basemodel_name, 'reg', tx, y)
        learner = S_learner(mu_tx)
    elif learner_name == 'PS':
        if not hat_learner[basemodel_name]['S']:
            tx = np.concatenate((t.reshape(-1, 1), x), axis=1)
            mu_tx = get_model(basemodel_name, 'reg', tx, y)
            hat_learner[basemodel_name]['S'] = S_learner(mu_tx)
        response_for_PS = hat_learner[basemodel_name]['S'].predict(x)
        learner = get_model(basemodel_name, 'reg', x, response_for_PS)
    elif learner_name == 'X':
        treat_idx = t==1
        control_idx = t==0
        response_for_tau0 = y1_pre[control_idx] - y[control_idx]
        response_for_tau1 = y[treat_idx] - y0_pre[treat_idx]
        tau0 = get_model(basemodel_name, 'reg', x[control_idx], response_for_tau0)
        tau1 = get_model(basemodel_name, 'reg', x[treat_idx], response_for_tau1)
        pi = hat_base_model[basemodel_name]['pi']
        learner = X_learner(pi, tau0, tau1)
    elif learner_name == 'T':
        learner = T_learner(mu1_hat, mu0_hat)

    elif learner_name == 'IPW':
        y1_pseudo = t / pi_x_pre * y
        y0_pseudo = (1-t) / (1-pi_x_pre) * y
        tau_pseudo = y1_pseudo-y0_pseudo
        learner = get_model(basemodel_name, 'reg', x, tau_pseudo)
    elif learner_name == 'DR':
        y1_pseudo = y1_pre + t / pi_x_pre * (y-y1_pre)
        y0_pseudo = y0_pre + (1-t) / (1-pi_x_pre) * (y - y0_pre)
        tau_pseudo = y1_pseudo-y0_pseudo
        learner = get_model(basemodel_name, 'reg', x, tau_pseudo)

    elif learner_name == 'R':
        residual_t = (t - pi_x_pre)
        residual_y = y - (t*y1_pre + (1-t)*y0_pre)
        tau_pseudo = residual_y/residual_t
        learner = get_model(basemodel_name, 'reg', x, tau_pseudo, sample_weight=residual_t**2)

    elif learner_name == 'RA':
        tau_pseudo = t*(y-y0_pre) + (1-t)*(y1_pre-y)
        learner = get_model(basemodel_name, 'reg', x, tau_pseudo)

    elif learner_name == 'U':
        residual_t = (t - pi_x_pre)
        residual_y = y - (t*y1_pre + (1-t)*y0_pre)
        tau_pseudo = residual_y / residual_t
        learner = get_model(basemodel_name, 'reg', x, tau_pseudo)
    return learner

def get_model(model_name, reg_or_cal, x, y, sample_weight=None, n_cv=3):

    if model_name == 'lr':
        if reg_or_cal == 'reg':
            model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=n_cv)
        if reg_or_cal == 'cla':
            model = LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=n_cv, max_iter=1000)
        model = model.fit(x, y, sample_weight=sample_weight)
    if model_name == 'xgb':
        if reg_or_cal == 'reg':
            automodel = AutoML()
            automl_settings = {
                "time_budget": 5,
                "metric": "r2",
                "task": "regression",
                "eval_method": 'cv',
                "n_splits": 3,
                "verbose": 0,
                "estimator_list": ['xgboost']
            }
            if sample_weight is None:
                automodel.fit(x, y, **automl_settings)
            else:
                automodel.fit(x, y, sample_weight=sample_weight, **automl_settings)

            model = automodel.model.estimator
        if reg_or_cal == 'cla':
            automodel = AutoML()
            automl_settings = {
                "time_budget": 5,
                "metric": "accuracy",
                "task": "classification",
                "eval_method": 'cv',
                "n_splits": 3,
                "verbose": 0,
                "estimator_list": ['xgboost']
            }
            automodel.fit(x, y, **automl_settings)
            model = automodel.model.estimator
    if model_name == 'net':
        if reg_or_cal == 'reg':
            model = MLP_Regressor(x, y, sample_weight)
        if reg_or_cal == 'cla':
            model = MLPClassifier(hidden_layer_sizes=(200, 200, 200, 100, 100), solver='adam',alpha=1e-5, random_state=0)
            model = model.fit(x, y)
    if model_name == 'rf':
        if reg_or_cal == 'reg':
            automodel = AutoML()
            automl_settings = {
                "time_budget": 5,
                "metric": "r2",
                "task": "regression",
                "eval_method": 'cv',
                "n_splits": 3,
                "verbose": 0,
                "estimator_list": ['rf']
            }
            if sample_weight is None:
                automodel.fit(x, y, **automl_settings)
            else:
                automodel.fit(x, y, sample_weight=sample_weight, **automl_settings)

            model = automodel.model.estimator
        if reg_or_cal == 'cla':
            automodel = AutoML()
            automl_settings = {
                "time_budget": 5,
                "metric": "accuracy",
                "task": "classification",
                "eval_method": 'cv',
                "n_splits": 3,
                "verbose": 0,
                "estimator_list": ['rf']
            }
            automodel.fit(x, y, **automl_settings)
            model = automodel.model.estimator
    if model_name == 'svm':
        if reg_or_cal == 'reg':
            automodel = AutoML()
            automodel.add_learner('svm', mySVM)
            automl_settings = {
                "time_budget": 2,
                "metric": "r2",
                "task": "regression",
                "eval_method": 'cv',
                "n_splits": 3,
                "verbose": 0,
                "estimator_list": ['svm']
            }
            if sample_weight is None:
                automodel.fit(x, y, **automl_settings)
            else:
                automodel.fit(x, y, sample_weight=sample_weight, **automl_settings)

            model = automodel.model.estimator
        if reg_or_cal == 'cla':
            automodel = AutoML()
            automodel.add_learner('svm', mySVM)
            automl_settings = {
                "time_budget": 2,
                "metric": "r2",
                "task": "classification",
                "eval_method": 'cv',
                "n_splits": 3,
                "verbose": 0,
                "estimator_list": ['svm']
            }
            automodel.fit(x, y, **automl_settings)
            model = automodel.model.estimator
    else:
        ValueError('Invalid regressor name')
    return model


def net_loss(concat_true, concat_pred):
    '''concat_true is 2-dim [yf,t_true]; concat_pred is 4-dim [y0,y1,pi_x]'''
    y_true = tf.reshape(concat_true[:, 0], [-1, 1])
    t_true = tf.reshape(concat_true[:, 1], [-1, 1])
    y0_pre = tf.reshape(concat_pred[:, 0], [-1, 1])
    y1_pre = tf.reshape(concat_pred[:, 1], [-1, 1])
    # t_pre = tf.reshape(concat_pred[:, 2], [-1, 1])
    y_pre = t_true*y1_pre + (1-t_true)*y0_pre
    '''factual loss of g and m'''
    y_factual_loss = tf.reduce_mean(tf.square(y_true - y_pre))
    # t_loss = tf.reduce_mean(K.binary_crossentropy(t_true, t_pre))
    '''final loss'''
    loss = y_factual_loss
    return loss



def MLP_Regressor(predictors, response, sample_weight=None):
    response = response.reshape(-1, 1)
    def my_loss(y_true, y_pre):
        return tf.reduce_mean(tf.square(y_true - y_pre))
    layer_reg = 0.01
    input_dim = predictors.shape[1]
    inputs = Input(shape=(input_dim,))
    # representation
    z = Dense(units=200, activation='relu', kernel_initializer='RandomNormal')(inputs)
    z = Dense(units=200, activation='relu', kernel_initializer='RandomNormal')(z)
    z = Dense(units=200, activation='relu', kernel_initializer='RandomNormal')(z)
    z = Dense(units=100, activation='relu', kernel_initializer='RandomNormal')(z)
    z = Dense(units=100, activation='relu', kernel_initializer='RandomNormal')(z)

    y_pre = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(layer_reg))(z)
    model = Model(inputs=inputs, outputs=y_pre)

    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor='val_loss', patience=5, min_delta=0.),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=True, mode='auto',
                          min_delta=0., cooldown=0, min_lr=0)
    ]
    '''initial optimizer'''
    opt = Adam(lr=0.001)
    # opt = SGD(lr=1e-5, momentum=0.9, nesterov=True)
    '''compile model_Y'''
    model.compile(loss=my_loss, optimizer=opt, metrics=my_loss)
    model.summary()
    # fit model
    model.fit(predictors, response,
                epochs=300, batch_size=int(64), validation_split=0.2, shuffle=False, callbacks=callbacks, sample_weight=sample_weight)
    return model
import tensorflow as tf

import json
import os

import progressbar

from tensorflow.keras import layers

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt
import pickle
import time

import copy

BATCH_SIZE = 16
BUFFER_SIZE = 2**20

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dev', dest='dev', action='store_true')
parser.add_argument('--config', dest='config_path', type=str, required=True)
parser.add_argument('--newpop', action='store_true')
args = parser.parse_args()

with open(args.config_path,'r') as ifile:
    CONFIG_JSON = json.load(ifile)['development' if args.dev else 'production']

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

IN_FILENAME = os.path.join(CONFIG_JSON['INDATA_FOLDER'],CONFIG_JSON['Xinfile'])
MODEL_PATH = os.path.join(CONFIG_JSON['INDATA_FOLDER'],CONFIG_JSON['Model_infile'])

df = pd.read_csv(IN_FILENAME)

######
# Variables selection
######
VAR_SEL = ["RTVANIRV1", "IVI0RV1", "IASPVr1", "Vco_new", "Vco_new2", "VCONV_RAT_3", "V21CONVER_3", "V21_prom_pres", "V21TM0002v3", "VTCP0082", "VTCPTI0077", "VTCPTI0076", "VTCPTI0084", "BMCI", "Car+Col%_dilu", "Hol%_dilu", "RVB"]
PLS_SEL = [f'PLS_{i}' for i in [0,1,4,7]]

VAR_SEL = VAR_SEL + PLS_SEL + ['Resultado_FO']

MSK = df.Resultado_FO != -10
scaler = StandardScaler()
scaler.fit(df.loc[MSK,VAR_SEL])
X = scaler.transform(df.loc[MSK,VAR_SEL]).astype(np.float32)
cv = df.cv_split[MSK].to_numpy()

##
class MEnsamble:
    def __init__(self, Model, n, ws, bins, **kwparams):
        self.n = kwparams.get('n', n)
        self.Model = kwparams.get('Model', Model)
        self.ms = None
        self.idxs = None
        self.__updateN__(**kwparams)
        self.ws = kwparams.get('ws', ws)
        self.bins = kwparams.get('bins', bins)

    def __updateN__(self, **kwparams):
        self.ms = [self.Model(**kwparams) for i in range(self.n)]
        self.idxs = [[] for i in range(self.n)]

    def splitDataset(self, y):
        bres = [(y > (self.bins[i] - 1e-10)) & (y <= self.bins[i + 1]) for i in range(len(self.bins) - 1)]
        for i in range(self.n):
            res = []
            for j, w in enumerate(self.ws):
                res.append(np.random.choice(
                    np.where(bres[j])[0],
                    np.min([w, bres[j].sum()]),
                    replace=False))
            #                 print(bres[j].sum(),w,len(res[-1]))
            self.idxs[i] = np.hstack(res)

    def fit(self, X, y, splitDataset=True):
        self.splitDataset(y)
        for m, idx in zip(self.ms, self.idxs):
            m.fit(X[idx], y[idx])

    def predict(self, X):
        yp = [m.predict(X) for m in self.ms]
        return np.median(yp, axis=0)

    def set_params(self, **params):
        m_keys = {}
        for p in params:
            assert not (hasattr(self, p) and hasattr(self.ms[0], p))
            if hasattr(self, p):
                self.__setattr__(p, params[p])
                if p == 'n':
                    self.__updateN__()
            else:
                m_keys[p] = params[p]
        for m in self.ms:
            m.set_params(**m_keys)


class Model_Wrapper(object):
    def __init__(self, model, columns, X_scaler, Y_scaler):
        self.model = model
        self.columns = columns

        self.X_scaler = X_scaler

        self.Y_scaler = Y_scaler

    def predict(self, X):
        X = X[self.columns]

        Xs = self.X_scaler.transform(X)
        yp = self.model.predict(Xs)

        ys = self.Y_scaler.transform(yp)
        y = pd.DataFrame(ys)
        return y


class NoScale:
    def transform(self, x, *args, **kwargs):
        return x


with open(MODEL_PATH, 'rb') as ifile:
    mwrap = pickle.load(ifile)

mwrap.model.predict(X[:BATCH_SIZE, :-5])

##
with open(os.path.join(CONFIG_JSON["INDATA_FOLDER"],'predictedXy.pkl'),'rb') as ifile:
    x = pickle.load(ifile)

epochs = 1024
bsize = 4096
lrn_prog = []


tr_x = x
tr_y = x[:,-1]

tr_x = scaler.transform(x)
tr_x[:,-1] = tr_y

ts_size = 2**14







ts_x = tr_x[:-ts_size]
tr_x = tr_x[-ts_size:]

ds_tr = tf.data.Dataset.from_tensor_slices(tr_x).batch(bsize).shuffle(tr_x.shape[0])
ds_ts = tf.data.Dataset.from_tensor_slices(ts_x).batch(bsize).shuffle(ts_x.shape[0])

tr_mse = tf.keras.losses.MeanSquaredError()


def train(m, epochs):
    optimiz = tf.keras.optimizers.Adam(1e-1, amsgrad=True)
    ts_mae = tf.keras.metrics.MeanAbsoluteError()

    @tf.function
    def train_step(m, optimiz, btch):
        x = btch[:, :-5]
        y = btch[:, -1]
        with tf.GradientTape() as tape:
            yp = m(x, training=True)
            lss = tr_mse(y, yp)
        grad = tape.gradient(lss, m.trainable_variables)
        optimiz.apply_gradients(zip(grad, m.trainable_variables))
        return lss

    @tf.function
    def test_step(m):
        x = ts_x[:, :-5]
        y = ts_x[:, -1]
        yp = m(x)
        lss = ts_mae(tf.reshape(y, [-1]), tf.reshape(yp, [-1]))
        return lss

    # with progressbar.ProgressBar(max_value=epochs) as bar:
    PATIENCE = 10
    i_from_best = 0
    best_model_path = 'bmodel.h5'

    ts_mae.reset_states()
    tloss = test_step(m)

    best_l = tloss
    best_e = None
    lrn_prog = []
    for e in range(epochs):

        if i_from_best > PATIENCE:
            break

        for i, btch in enumerate(ds_tr):
            lss = train_step(m, optimiz, btch)

            # bar.update(e)

        ts_mae.reset_states()
        tloss = test_step(m)
        lrn_prog.append(tloss.numpy())

        i_from_best += 1

        if best_e is None or tloss < best_l:
            best_e = e
            i_from_best = 0
            best_w = m.save_weights(best_model_path)
            best_l = tloss

    if epochs > 0:
        m.load_weights(best_model_path)

    return best_l, lrn_prog


####
##
##  GENETIC AUX FUNCTIONS
##
####

def build_model(desc):
    res = []
    for l in desc:
        res.append(layers.Dense(l['out'],kernel_regularizer=l['kr'],bias_regularizer=l['br']))
        if l.get('bn',False):
            res.append(layers.BatchNormalization())
        if not l.get('final', False):
            res.append(layers.LeakyReLU(alpha=0.01))
    return tf.keras.models.Sequential(res)


def parse_regularizer(d):
    if d is None:
        return None
    kr = []
    if d['config']['l1'] != 0.0:
        kr.append('l1')
    if d['config']['l2'] != 0.0:
        kr.append('l2')
    kr = '_'.join(kr)
    return kr


def build_mdesc(m):
    dense_idxs = list(np.where([type(l) == layers.Dense for l in m.layers])[0])
    res = []

    def dict_build(out, kr, br, bn, final, idx):
        return {
            'out': out,
            'kr': kr,
            'br': br,
            'bn': bn,
            'final': final,
            'idx': idx
        }

    for ini, end in zip(dense_idxs[:], dense_idxs[1:] + [dense_idxs[-1] + 1]):
        block = m.layers[ini:end]
        dense = block[0]
        dense_config = dense.get_config()

        # Config
        out = dense_config['units']
        kr = parse_regularizer(dense_config['kernel_regularizer'])
        br = parse_regularizer(dense_config['bias_regularizer'])
        bn = type(block[1]) == layers.BatchNormalization if len(block) > 1 else False
        final = end == len(m.layers)

        res.append(dict_build(out, kr, br, bn, final, ini))
    return res


# glunif = tf.keras.initializers.glorot_uniform()


def reshape_weights(target, donor):
    res = []

    for target_w, donor_w in zip(target, donor):
        new_w = target_w
        assert new_w.ndim <= 2
        if new_w.ndim == 1:
            max_s = min(new_w.shape[0], donor_w.shape[0])
            # print(max_s)
            new_w[:max_s] = donor_w[:max_s]
        else:
            max_1 = min(new_w.shape[0], donor_w.shape[0])
            max_2 = min(new_w.shape[1], donor_w.shape[1])
            # print(max_1,max_2)
            new_w[:max_1, :max_2] = donor_w[:max_1, :max_2]
        res.append(new_w)
    return res


def transfer_weights(target, donor):
    target_idx = np.where([type(l) == layers.Dense for l in target.layers])[0]
    donor_idx = np.where([type(l) == layers.Dense for l in donor.layers])[0]
    assert (len(target_idx) == len(donor_idx))

    for t, d in zip(target_idx, donor_idx):
        target.layers[t].set_weights(
            reshape_weights(
                target.layers[t].get_weights(),
                donor.layers[d].get_weights()
            )
        )


def transfer_weights_w2m(target_m, donor_w):
    target_idx = np.where([type(l) == layers.Dense for l in target_m.layers])[0]
    # print(len(target_m.layers),len(donor_w))
    for t in target_idx:
        target_m.layers[t].set_weights(
            reshape_weights(
                target_m.layers[t].get_weights(),
                donor_w[t]
            )
        )


MAX_LEN = 8



def _cross(m1_desc, m1_cut, m1_w, m2_desc, m2_cut, m2_w):
    # Cross 1 desc and weights
    c1 = copy.deepcopy(m1_desc[:m1_cut] + m2_desc[m2_cut:])
    # print("!!",len(m1_desc),len(m2_desc),len(c1),m1_cut,m2_cut)
    w1 = copy.deepcopy(m1_w[:m1_desc[m1_cut]['idx']] + m2_w[m2_desc[m2_cut]['idx']:])
    # correct offset
    # print("#",m1_desc[m1_cut]['idx'],m2_desc[m2_cut]['idx'])
    c1_off = m1_desc[m1_cut]['idx'] - m2_desc[m2_cut]['idx']
    # print(m1_desc,m2_desc,c1_off,m1_cut,m2_cut)
    for d in c1[m1_cut:]:
        d['idx'] += c1_off
    # print(c1)
    return c1, w1


MAX_WIDTH = 512


def mutate_mod(m_desc, m_w):
    # change size
    l = np.random.randint(len(m_desc) - 1)
    p = np.random.rand()
    # print(p)
    if p < 0.5:
        m_desc[l]['out'] = np.random.randint(MAX_WIDTH)
    # remove bn
    else:
        # print(m_desc[l]['bn'])
        m_desc[l]['bn'] = not (m_desc[l]['bn'])
        if m_desc[l]['bn']:
            m_w.insert(m_desc[l]['idx'] + 1, [])
        else:
            m_w.pop(m_desc[l]['idx'] + 1)


def net_cross(m1, m2, mutate_prob=0.1):
    m1_desc = build_mdesc(m1)
    m2_desc = build_mdesc(m2)

    m1_cut = np.random.randint(1, len(m1_desc))
    m2_cut = np.random.randint(1, len(m2_desc))

    m1_w = [l.get_weights() for l in m1.layers]
    m2_w = [l.get_weights() for l in m2.layers]

    fake_input = tf.zeros((1, m1_w[0][0].shape[0]))

    p1, p2 = np.random.rand(2)

    c1_desc, w1 = _cross(m1_desc, m1_cut, m1_w, m2_desc, m2_cut, m2_w)
    if p1 < mutate_prob:
        mutate_mod(c1_desc, w1)
    c2_desc, w2 = _cross(m2_desc, m2_cut, m2_w, m1_desc, m1_cut, m1_w)
    if p2 < mutate_prob:
        mutate_mod(c2_desc, w2)

    c1 = build_model(c1_desc)
    c1(fake_input)
    w1 = transfer_weights_w2m(c1, w1)

    c2 = build_model(c2_desc)
    c2(fake_input)
    # a,b,c=(c2,c2_desc,w2)
    w2 = transfer_weights_w2m(c2, w2)

    # return c1,c2
    return c1, c2

def save_model(m, m_path):
    with open(m_path.format('pkl'), 'wb') as ofile:
        desc = build_mdesc(m)
        weights = [l.get_weights() for l in m.layers]
        pickle.dump((desc, weights), ofile)


def load_model(m_path):
    with open(m_path.format('pkl'), 'rb') as ifile:
        desc, weights = pickle.load(ifile)
        # print("Loading model {}".format(m_path),desc)
        m = build_model(desc)
        m(tf.zeros((1, 17)))
        for l, w in zip(m.layers, weights):
            l.set_weights(w)
        return m

####
##
##  GENETIC ROUTINE
##
####

NGENS = CONFIG_JSON['GEN_NGENS']
NPOP = CONFIG_JSON['GEN_POPSIZE']
INIT_EPOCH = CONFIG_JSON['GEN_INIT_EPOCH']
LS_EPOCHS = CONFIG_JSON['GEN_LS_EPOCH']

pop = []
losses = []

# Initialize pop
P_BN = 0.5
MAX_LEN = 8
print("Building population")
if args.newpop:
    for i in range(NPOP):
        # random desc
        p_bn = np.random.rand()
        out_n = np.random.randint(MAX_WIDTH)
        len_m = np.random.randint(MAX_LEN - 1)
        d = [
            {'out': np.random.randint(MAX_WIDTH), 'kr': 'l2', 'br': 'l2', 'bn': np.random.rand() < P_BN} for i in
            range(np.random.randint(1, MAX_LEN - 1))]
        d += [{'out': 1, 'kr': 'l2', 'br': 'l2', 'final': True}]
        print(d)
        # generate model
        m = build_model(d)
        # train for init_epoch
        loss, _ = train(m, INIT_EPOCH)
        # add to pop
        pop.append(m)
        losses.append(loss)
        print("Built pop {}".format(i))
else:
    pop = [load_model(os.path.join(CONFIG_JSON['OUTDATA_FOLDER'],"gen_model_" + str(i) + ".{}")) for i in range(40)]
    losses = [train(m,0)[0] for m in pop]


print("Starting genetic")
for i in range(NGENS):
    # take 2 parents
    i1, i2 = np.random.choice(range(NPOP), 2, replace=False)
    # create offspring
    o1, o2 = net_cross(pop[i1], pop[i2],mutate_prob=2/3)
    o1_loss, _ = train(o1, INIT_EPOCH)
    o2_loss, _ = train(o2, INIT_EPOCH)
    min_o = o1 if o1_loss < o2_loss else o2
    min_l = min(o1_loss, o2_loss)

    max_p_idx = np.argmax(losses)
    if min_l < losses[max_p_idx]:
        pop[max_p_idx] = min_o
        losses[max_p_idx] = min_l

    # random local search

    if np.random.rand() < 0.5:
        ils = np.random.choice(range(NPOP))
    else:
        ils = np.argmin(losses)
    ls_loss, _ = train(pop[ils], LS_EPOCHS)
    losses[ils] = ls_loss

    print("GEN {}: Min={}\tMean={}\tStd={}".format(i,np.min(losses),np.mean(losses),np.std(losses)))


for i, m in enumerate(pop):
    save_model(m, os.path.join(CONFIG_JSON['OUTDATA_FOLDER'],"gen_model_" + str(i) + ".{}"))

# pop2 = [load_model("/content/drive/My Drive/UGR/REPSOL/data/gen_model_" + str(i) + ".{}") for i in range(40)]
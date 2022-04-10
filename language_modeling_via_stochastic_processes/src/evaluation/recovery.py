import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
import os
from language_modeling_via_stochastic_processes.src import constants

np.random.seed(5)

NUM_TRIALS = 3
NUM_SAMPLES_PER_SEQ = 1000
MU=0.0
SIGMA=1.0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_DIR = constants.VISUALIZATION_DIR

def g(M, x):
    return np.dot(M, x)

def run_model(model, x):
    feats, pred = model.forward(torch.tensor([x], device=device).float())
    pred = pred.cpu().detach().numpy()[0]
    return pred

def get_values(model, dataset, M, Rot, x_t):
    """
    Run:
    * g(x_t) to get y_t
    * model(y_t) to get \tilde{x}
    * R^{-1}\tilde{x} to get recovery accounting rotation
    """
    g_x_t = g(M, x_t)
    if hasattr(dataset, 'noisy_sigma'):
        # TODO divide noise by data dim
        noise = np.random.normal(0, dataset.noisy_sigma, dataset.data_dim)
        # noise /= dataset.data_dim
        g_x_t += noise
    pred_x_t = run_model(model, g_x_t)
    r_pred_x_t = Rot.T.dot(pred_x_t)
    return g_x_t, pred_x_t, r_pred_x_t


def sample_trajectory(model, dataset, M, Rot, dt ):
    data_dim = model.data_dim
    # x_t = np.random.uniform(-5, 5, data_dim) # ou
    x_t = np.zeros(data_dim) # bridge
    g_x_t, pred_x_t, r_pred_x_t = get_values(model=model, dataset=dataset, M=M, Rot=Rot, x_t=x_t)

    x_ts = [x_t]
    g_x_ts = [g_x_t]
    pred_x_ts = [pred_x_t]
    r_pred_x_ts = [r_pred_x_t]
    for _ in range(NUM_SAMPLES_PER_SEQ-1):
        noise = np.sqrt(dt)*SIGMA*np.random.normal(MU, 1.0, data_dim)
        noise /= data_dim
        # bridge
        t = _/NUM_SAMPLES_PER_SEQ
        dt = dataset.dt
        x_tp1 = x_t * (1- dt/(1. - t)) + (dt/(1.-t))*dataset.B_T + noise

        ## OU
        # x_tp1 = x_t - x_t * dt + noise
        g_x_tp1, pred_x_tp1, r_pred_x_tp1 = get_values(
            model=model, dataset=dataset, M=M, Rot=Rot, x_t=x_tp1)

        # Track vals
        x_ts.append(x_tp1)
        g_x_ts.append(g_x_tp1)
        pred_x_ts.append(pred_x_tp1)
        r_pred_x_ts.append(r_pred_x_tp1)

        # Update
        x_t = x_tp1

    t = np.arange(NUM_SAMPLES_PER_SEQ)
    x_ts = np.array(x_ts)
    g_x_ts  = np.array(g_x_ts)
    pred_x_ts  = np.array(pred_x_ts)
    r_pred_x_ts  = np.array(r_pred_x_ts)
    return {'t': t,
            'x_ts': x_ts,
            'g_x_ts': g_x_ts,
            'pred_x_ts': pred_x_ts,
            'r_pred_x_ts': r_pred_x_ts}

def plot_trajectory(data, data_dim, example_num, fname):
    t = data['t']
    x_ts = data['x_ts']
    g_x_ts = data['g_x_ts']
    pred_x_ts = data['pred_x_ts']
    r_pred_x_ts = data['r_pred_x_ts']

    fig, axes = plt.subplots(data_dim, sharex=True, figsize=(10, 6))
    fig.suptitle('Example {}'.format(example_num))

    for _, ax in enumerate(axes):
        ax.plot(t, x_ts[:, _],label='dim {}: x_t'.format(_), )
        ax.plot(t, g_x_ts[:, _], label='dim {}: g(x_t)'.format(_))
        ax.plot(t, pred_x_ts[:, _],  label='dim {}: h(g(x_t))'.format(_),linestyle="--")
        ax.plot(t, r_pred_x_ts[:, _],  label='dim {}: R h(g(x_t))'.format(_),linestyle="--")
        ax.set_title("dim={}".format(_))
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(SAVE_DIR, fname))
    print(x_ts)
    plt.clf()

def track_data(data):
    recovery_error = np.abs(data['x_ts'] - data['r_pred_x_ts'])
    wandb.log({
               'recovery_error_mean': recovery_error.mean(),
               'recovery_error_std': recovery_error.std(),
               })


def get_recovery(model, dataset, M, dt, seed, objective_name, exp_name):
    """
    Runs an example trajectory

    Args:
        model: ou_model.OUModel
        M: true mixing matrix applied to x_t to produce observations y_t
        dt: sampling rate
        seed: seed used for training
        objective_name: objective used for training model.

    """
    data_dim = model.data_dim
    # \tilde{M^{-1}}
    predicted_M_inv = model.predictor.weight.cpu().detach().numpy()
    # Recovery up to rotation
    Rot = np.dot(predicted_M_inv, M)
    det = np.linalg.det(Rot)

    for _ in range(NUM_TRIALS):
        data = sample_trajectory(model=model, dataset=dataset, M=M, Rot=Rot, dt=dt)
        fname = "exp{}_d{}_dt{}_seed{}_trial{}.pdf".format(exp_name, data_dim, dt, seed, _)
        plot_trajectory(data, data_dim=model.data_dim, example_num=_,
                        fname=os.path.join(objective_name, fname))
        track_data(data)

    wandb.log({'final_det': det,})





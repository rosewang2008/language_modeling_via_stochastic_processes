
from src.datasets.ou import *
from src.datasets.ts import *

DATASETS = {
    'OUData': OUData,
    'OUData_LT': OUData_LT,
    'OUBridgeData_LT': OUBridgeData_LT,
    'Noisy_OUData_LT': Noisy_OUData_LT,
    'OUData_Bridge': OUData_Bridge,
    'BrownianBridgeData_LT':BrownianBridgeData_LT,
    'BrownianBridgeRandomT_LT':BrownianBridgeRandomT_LT,
    'BrownianBridgeRandom_LT':BrownianBridgeRandom_LT,
    'OUData_Classification': OUData_Classification,
    'TimeSeriesData': TimeSeriesData,
    'StocksData': StocksData,
    'IntervalStocksData': IntervalStocksData,

}


def get_datasets(config):
    dataset = DATASETS[config.data_params.name]
    ### For OU data ####
    if "BrownianBridge" in config.data_params.name:
        train_dataset = dataset(
            train=True,
            data_dim=config.data_params.data_dim,
            n_samples=config.data_params.n_samples,
            samples_per_seq=config.data_params.samples_per_seq,
            seed=config.data_params.data_seed,
            one_hot_labels=False,
            dt=config.data_params.dt,
            mu=config.data_params.mu,
            sigma=config.data_params.sigma,
            )
        test_dataset = dataset(
            train=False,
            data_dim=config.data_params.data_dim,
            n_samples=config.data_params.n_samples,
            samples_per_seq=config.data_params.samples_per_seq,
            seed=config.data_params.data_seed+1,
            one_hot_labels=False,
            dt=config.data_params.dt,
            mu=config.data_params.mu,
            sigma=config.data_params.sigma,
            )
    if "Noisy" in config.data_params.name:
        train_dataset = dataset(
            train=True,
            data_dim=config.data_params.data_dim,
            n_samples=config.data_params.n_samples,
            samples_per_seq=config.data_params.samples_per_seq,
            seq_len=config.data_params.seq_len,
            seed=config.data_params.data_seed,
            one_hot_labels=False,
            dt=config.data_params.dt,
            mu=config.data_params.mu,
            sigma=config.data_params.sigma,
            noisy_sigma=config.data_params.noisy_sigma,
            )

        if ("LT" in config.data_params.name) or ("Class" in config.data_params.name):
            test_dataset = dataset(
                train=False,
                data_dim=config.data_params.data_dim,
                n_samples=config.data_params.n_samples,
                seq_len=config.data_params.seq_len,
                samples_per_seq=config.data_params.samples_per_seq,
                seed=config.data_params.data_seed+1,
                one_hot_labels=False,
                dt=config.data_params.dt,
                mu=config.data_params.mu,
                sigma=config.data_params.sigma,
                noisy_sigma=config.data_params.noisy_sigma,
                A=train_dataset.A
                )

    elif "OU" in config.data_params.name:
        train_dataset = dataset(
            train=True,
            data_dim=config.data_params.data_dim,
            n_samples=config.data_params.n_samples,
            samples_per_seq=config.data_params.samples_per_seq,
            seed=config.data_params.data_seed,
            one_hot_labels=False,
            dt=config.data_params.dt,
            mu=config.data_params.mu,
            sigma=config.data_params.sigma,
            )

        if ("LT" in config.data_params.name) or ("Class" in config.data_params.name):
            test_dataset = dataset(
                train=False,
                data_dim=config.data_params.data_dim,
                n_samples=config.data_params.n_samples,
                samples_per_seq=config.data_params.samples_per_seq,
                seed=config.data_params.data_seed+1,
                one_hot_labels=False,
                dt=config.data_params.dt,
                mu=config.data_params.mu,
                sigma=config.data_params.sigma,
                A=train_dataset.A
                )
    elif "TimeSeries" in config.data_params.name:
        train_dataset = dataset(
                train=True,
                seed=config.data_params.data_seed,
                data_dim=config.data_params.data_dim,
                sequence_length=config.data_params.seq_len,
                one_hot_labels=False,
                filepath=config.data_params.train_path
        )
        test_dataset = dataset(
                train=False,
                seed=config.data_params.data_seed,
                data_dim=config.data_params.data_dim,
                sequence_length=config.data_params.seq_len,
                one_hot_labels=False,
                filepath=config.data_params.test_path
        )

    elif "Interval" in config.data_params.name:
        train_dataset = dataset(
                train=True,
                seed=config.data_params.data_seed,
                data_dim=config.data_params.data_dim,
                sequence_length=config.data_params.seq_len,
                one_hot_labels=False,
                directory=config.data_params.train_dir,
                sampling_interval=config.data_params.sampling_interval
        )
        test_dataset = dataset(
                train=False,
                seed=config.data_params.data_seed,
                data_dim=config.data_params.data_dim,
                sequence_length=config.data_params.seq_len,
                one_hot_labels=False,
                directory=config.data_params.test_dir,
                sampling_interval=config.data_params.sampling_interval
        )

    elif "Stocks" in config.data_params.name:
        train_dataset = dataset(
                train=True,
                seed=config.data_params.data_seed,
                data_dim=config.data_params.data_dim,
                sequence_length=config.data_params.seq_len,
                one_hot_labels=False,
                directory=config.data_params.train_dir
        )
        test_dataset = dataset(
                train=False,
                seed=config.data_params.data_seed,
                data_dim=config.data_params.data_dim,
                sequence_length=config.data_params.seq_len,
                one_hot_labels=False,
                directory=config.data_params.test_dir
        )

    return train_dataset, test_dataset


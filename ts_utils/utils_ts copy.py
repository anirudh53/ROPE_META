from ray.air import Checkpoint, session
from torch.utils.data import DataLoader
from typing import Dict, Union, List, Tuple
from string import ascii_lowercase
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import h5py
import json
import yaml
import time
import copy
import os


LABELS = [f"{letter})" for letter in ascii_lowercase]


def series_to_supervised(data: np.ndarray, n_in: int = 1, n_out: int = 1, dropnan: bool = True) -> pd.DataFrame:
    """converts raw data to supervised time series format; takes time x (data + inputs)

    :param data: input data (data/coefficients + inputs)
    :type data: np.ndarray
    :param n_in: number of input time steps, defaults to 1
    :type n_in: int, optional
    :param n_out: number of output time steps, defaults to 1
    :type n_out: int, optional
    :param dropnan: drop columns with nan values, defaults to True
    :type dropnan: bool, optional
    :return: structured time series
    :rtype: pd.DataFrame
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # shift input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def get_training_statistics(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """takes the training data and computes the standard normalization statistics

    :param data: training dataset
    :type data: torch.Tensor
    :return: mean and standard deviation about the first axis (should be samples or time)
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    mean = data.mean(0)
    std = data.std(0)
    return mean, std


def normalize(data: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """normalizes the input data using normalization statistics

    :param data: data to be normalized
    :type data: torch.Tensor
    :param mean: mean spatial grid
    :type mean: torch.Tensor
    :param std: spatial grid standard deviation
    :type std: torch.Tensor
    :return: normalized data
    :rtype: torch.Tensor
    """
    return (data - mean) / std


def denormalize(data: np.ndarray, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """denormalizes some input data based on standard normalization statistics

    :param data: input data
    :type data: np.ndarray
    :param stats: standard normalization statistics
    :type stats: Dict[str: torch.Tensor]
    :return: de-normalized data
    :rtype: torch.Tensor
    """
    data = torch.Tensor(data)
    return (data * stats["sigma"] + stats["mu"]).numpy()


def load_h5(filename: str) -> Dict[str, np.ndarray]:
    """loads a single hdf file and returns the variables

    :param filename: name of TIEGCM coefficient file
    :type filename: str
    :return: density array
    :rtype: Dict[str, np.ndarray]
    """
    keys = ["coeff", "F10", "Kp", "doy", "utc"]
    output = {}
    with h5py.File(filename, "r") as hf:
        for key in keys:
            output[key] = np.array(hf.get(key))
    return output


def normalize_time(array: np.ndarray, tau: int) -> Tuple[np.ndarray, np.ndarray]:
    """take an array of time information and create sinusoidal versions using a time constant (tau)

    :param array: input time array
    :type array: np.ndarray
    :param tau: time constant
    :type tau: int
    :return: sinusoidal transformations of the input array
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    output1 = np.sin(2 * np.pi * array / tau)
    output2 = np.cos(2 * np.pi * array / tau)
    return output1, output2


def combine_coefficients_and_inputs(
    coeffs: np.ndarray, F10: np.ndarray, Kp: np.ndarray, doy: np.ndarray, utc: np.ndarray
) -> np.ndarray:
    """take the coefficients and all inputs and combine them into a single array

    :param coeffs: latent space coefficients
    :type coeffs: np.ndarray
    :param F10: F_10.7 solar proxy
    :type F10: np.ndarray
    :param Kp: geomagnetic K-index
    :type Kp: np.ndarray
    :param doy: day of year
    :type doy: np.ndarray
    :param utc: universal time (hours)
    :type utc: np.ndarray
    :return: combined coefficient and input array (n_time x (n_coeff + n_inputs))
    :rtype: np.ndarray
    """
    doy1, doy2 = normalize_time(doy, 365.25)
    utc1, utc2 = normalize_time(utc, 24)
    output = np.concatenate(
        (
            coeffs,
            F10.reshape(len(F10), 1),
            Kp.reshape(len(Kp), 1),
            utc1.reshape(len(utc1), 1),
            utc2.reshape(len(utc2), 1),
            doy1.reshape(len(doy1), 1),
            doy2.reshape(len(doy2), 1),
        ),
        axis=1,
    )
    return output


def load_data(data_config: Dict[str, Union[str, List[int]]], test: bool) -> Tuple[np.ndarray, np.ndarray]:
    """loads data from all years for training and validation

    :param data_config: data section of the config
    :type data_config: Dict[str, Union[str, List[int]]]
    :param test: whether or not you are using the test set
    :type test: bool
    :return: training and validation data
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    directory = data_config.get("directory")
    suffix = data_config.get("suffix")
    training_years = data_config.get("training_years")
    if not test:
        validation_years = data_config.get("validation_years")
    else:
        validation_years = data_config.get("test_years")
    for year in training_years:
        if year == training_years[0]:
            loadedT = load_h5(f"{directory}pca_{year}.h5")
        else:
            loaded0 = load_h5(f"{directory}pca_{year}.h5")
            for key, value in loaded0.items():
                loadedT[key] = np.concatenate((loadedT[key], loaded0[key]), axis=0)
    for year in validation_years:
        if year == validation_years[0]:
            loadedV = load_h5(f"{directory}pca_{year}.h5")
        else:
            loaded0 = load_h5(f"{directory}pca_{year}.h5")
            for key, value in loaded0.items():
                loadedV[key] = np.concatenate((loadedV[key], loaded0[key]), axis=0)
    loadedT = combine_coefficients_and_inputs(
        loadedT["coeff"],
        loadedT["F10"],
        loadedT["Kp"],
        loadedT["doy"],
        loadedT["utc"],
    )
    loadedV = combine_coefficients_and_inputs(
        loadedV["coeff"],
        loadedV["F10"],
        loadedV["Kp"],
        loadedV["doy"],
        loadedV["utc"],
    )
    return loadedT, loadedV


def load_and_process_data(
    data_config: Dict[str, Union[str, List[int]]], test: bool = False
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """loads TIE-GCM data, converts to log10 space, splits, and normalizes the data

    :param data_config: data section of the config
    :type data_config: Dict[str, Union[str, List[int]]]
    :param test: whether or not you are using the test set, defaults to False
    :type test: bool, optional
    :return: normalized training and validation data
    :rtype: Dict[str, torch.Tensor]
    """
    # load TIEGCM coefficient data for training and validation
    outputT, outputV = load_data(data_config, test=test)
    mean, std = get_training_statistics(outputT)
    mean, std = np.array(mean), np.array(std)
    scaledT = normalize(outputT, mean, std)
    scaledV = normalize(outputV, mean, std)
    # frame as supervised learning
    reframedT = series_to_supervised(scaledT, data_config["notS"], 1)
    reframedV = series_to_supervised(scaledV, data_config["notS"], 1)
    # drop columns we do not want to predict (last 6 are the inputs for what we predict)
    reframedT.drop(reframedT.columns[-data_config["noI"] :], axis=1, inplace=True)
    reframedV.drop(reframedV.columns[-data_config["noI"] :], axis=1, inplace=True)
    # split into train and test sets
    valuesT = reframedT.values
    valuesV = reframedV.values
    n_train_hoursT = data_config["batch_size"] * int(np.round(valuesT.shape[0] / data_config["batch_size"]))
    n_train_hoursV = data_config["batch_size"] * int(np.round(valuesV.shape[0] / data_config["batch_size"]))
    train0 = valuesT[:n_train_hoursT, :]
    test0 = valuesV[:n_train_hoursV, :]
    # split train into input and outputs
    train_X, train_y = train0[:, : -data_config["nm"]], train0[:, -data_config["nm"] :]
    test_X, test_y = test0[:, : -data_config["nm"]], test0[:, -data_config["nm"] :]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape(
        int(train_X.shape[0]),
        data_config["notS"],
        int(train_X.shape[1] / float(data_config["notS"])),
    )
    test_X = test_X.reshape(
        (
            int(test_X.shape[0]),
            data_config["notS"],
            int(test_X.shape[1] / float(data_config["notS"])),
        )
    )
    # convert to tensor
    xT, xV = torch.Tensor(train_X), torch.Tensor(test_X)
    yT, yV = torch.Tensor(train_y), torch.Tensor(test_y)
    # get normalization statistics
    stats = {"mu": torch.Tensor(mean), "sigma": torch.Tensor(std)}
    if test:
        data = {"test": (xV, yV)}
    else:
        data = {"train": (xT, yT), "valid": (xV, yV)}
    output = (data, stats)
    return output


def stack_for_training(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """stack training examples, so there are 125-step chunks

    :param x: input
    :type x: torch.Tensor
    :param y: output
    :type y: torch.Tensor
    :return: stacked input / output
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    num_examples = x.shape[0]
    x1 = torch.zeros([num_examples // 125, 125, 3, 16], dtype=torch.float32)
    for i in range(num_examples // 125):
        x1[i] = x[i * 125 : (i+1) * 125]
    y1 = torch.zeros([num_examples // 125, 125, 10], dtype=torch.float32)
    for i in range(num_examples // 125):
        y1[i] = y[i * 125 : (i+1) * 125]
    return x1, y1


def get_Kp_weights(filename: str) -> np.ndarray:
    """looks at Kp distributions and gets weighting factors for different samples based on their Kp

    :param filename: path to file containing Kp for training period
    :type filename: str
    :return: weighting factors for each sample
    :rtype: np.ndarray
    """
    with h5py.File(filename, "r") as hf:
        Kp = np.array(hf.get("Kp"))
    Kp[Kp == 9.0] = 8.99
    weights = [[], [], [], [], [], [], [], [], []]
    for i in range(len(Kp)):
        weights[int(np.floor(Kp[i]))].append(i)
    sums = [len(weight) for weight in weights]
    weights_S = len(Kp) / np.array(sums)
    weights_S = 100 / np.sum(weights_S) * weights_S
    # Create sample weighting array
    sample = np.zeros(len(Kp), dtype="float32")
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            sample[weights[i][j]] = weights_S[int(np.floor(Kp[weights[i][j]]))]
    return sample


class LSTM_Training:
    """trainer class for an autoencoder; supports training for CVAEs, InfoVAEs, COAEs, or CVOAEs"""

    def __init__(
        self,
        config: Dict[str, Union[int, str, float]],
        model: torch.nn.Module,
        is_tuner: bool,
        patience: int,
    ):
        """initializes VAE_training object

        :param config: config dictionary for both model and training parameters
        :type config: Dict[str, Union[int, str, float]]
        :param model: initialized VAE object
        :type model: torch.nn.Module
        :param is_tuner: True if running a hyperparameter tuner, False if not
        :type is_tuner: bool
        :param patience: number of epochs without improvement to terminate training
        :type patience: int
        """
        if not is_tuner:
            self._build_output(config)
        self._config = config.get("training")
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._is_tuner = is_tuner
        self._patience = patience
        self.logs = {"loss": [], "val_loss": []}
        self._model = model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """gets the optimizer and moves model to the device for training"""
        self._optimizer = self._get_optimizer()(self._model.parameters(), lr=self._config.get("learning_rate"))
        self._model.to(self._device)
        if self._config.get("weight_path", None) is not None:
            weights = torch.load(self._config.get("weight_path"))
            self._model.load_state_dict(weights)
            print("Loaded pretrained weights")
        else:
            print("No pretrained weights; they have been randomly initialized")
        self._lr_scheduler = None
        if (self._config.get("step_size", None) is not None) and (self._config.get("gamma", None) is not None):
            step_size = self._config["step_size"]
            gamma = self._config["gamma"]
            self._lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self._optimizer,
                step_size=step_size,
                gamma=gamma,
                verbose=True,
            )

    def _build_output(self, config: Dict[str, Union[int, str, float]]) -> None:
        """sets the path, creates the output directory if it doesn't exist, and saves the config there

        :param config: config dictionary for both model and training parameters
        :type config: Dict[str, Union[int, str, float]]
        """
        self._path = "weights/" + config.get("training").get("model_name", "no_model_name_provided") + "/"
        if not os.path.exists(self._path):
            os.mkdir(self._path)
        with open(self._path + "config.yaml", "w+") as f:
            yaml.dump(config, f)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """converts the configured optimizer string to the optimizer object

        :return: desired optimizer
        :rtype: torch.optim.optimizer.Optimizer
        """
        opts = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "adagrad": torch.optim.Adagrad,
            "rmsprop": torch.optim.RMSprop,
            "nadam": torch.optim.NAdam,
        }
        return opts[self._config.get("optimizer").lower()]

    def _add_to_logs(
        self,
        loss: List[float] = [],
        val_loss: List[float] = [],
    ) -> None:
        """adds the current losses to log; set up to work for both training and validation with the
        use of lists, list-addition, and defaults

        :param loss: overall loss, defaults to []
        :type loss: List[float], optional
        :param val_loss: overall validation loss, defaults to []
        :type val_loss: List[float], optional
        """
        self.logs["loss"] += loss
        self.logs["val_loss"] += val_loss

    def _compile_losses(
        self,
        running_loss: float,
        dataset_size: int,
    ) -> Tuple[Union[torch.FloatTensor, str]]:
        """computes overall epoch loss for each of the individual losses; if a loss is zero, it will
        return a string

        :param running_loss: total summed loss over every sample in epoch
        :type running_loss: float
        :param dataset_size: size of dataset (either training or validation set)
        :type dataset_size: int
        :return: _description_
        :rtype: Tuple[Union[torch.FloatTensor, str]]
        """
        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def _compute_metric(self, x: torch.Tensor, xhat: torch.Tensor, latent: torch.Tensor) -> float:
        """computes the custom metric

        :param x: normalized density
        :type x: torch.Tensor
        :param xhat: normalized prediction
        :type xhat: torch.Tensor
        :param latent: latent representation
        :type latent: torch.Tensor
        """
        x = x.detach().cpu().numpy()
        xhat = xhat.detach().cpu().numpy()
        latent = latent.detach().cpu().numpy()
        metric = self._model.metric(x, xhat, latent)
        return metric

    def train(self, dataloaders: DataLoader, dataset_sizes: Dict[str, int]) -> None:
        """trains the model based on the configured training scheme

        :param dataloaders: dataloaders for both training and validation
        :type dataloaders: DataLoader
        :param dataset_sizes: number of samples in training and validation sets
        :type dataset_sizes: Dict[str, int]
        """
        # get initial variables
        since = time.time()
        best_model_wts = copy.deepcopy(self._model.state_dict())
        best_metric = np.inf
        training_flag = True
        counter = 0
        # loop through epochs
        for epoch in range(self._config.get("epochs", 1)):
            if not training_flag:
                break
            counter += 1
            print(f'Epoch {epoch+1}/{self._config.get("epochs", 1)}')
            print("-" * 10)
            phase_flag = True
            # iterate over two phases
            for phase in ["train", "valid"]:
                if not phase_flag:
                    training_flag = False
                    break
                # set model to training or evaluation mode
                if phase == "train":
                    self._model.train()
                else:
                    self._model.eval()
                # initialize losses
                running_loss = 0.0
                set_flag = True
                # iterate over all batches in the phase
                for x in tqdm(dataloaders[phase]):
                    sample_weights = None
                    if len(x) == 3:
                        x, y, sample_weights = x[0], x[1], x[2]
                        sample_weights = sample_weights.to(self._device)
                    elif len(x) == 2:
                        x, y = x[0], x[1]
                    else:
                        raise ValueError(f"Length of `x` expected to be 2 or 3. len(x) = {len(x)}")
                    # move input to CPU or GPU
                    x = x.to(self._device)
                    y = y.to(self._device)
                    if not set_flag:
                        phase_flag = False
                        break
                    self._model.reset_states()
                    # set gradients to zero before running batch through the model
                    for b in range(x.shape[1]): # shape is (1, num_samples_in_batch, 16)
                        self._optimizer.zero_grad()
                        # have gradients enabled only in training phase
                        with torch.set_grad_enabled(phase == "train"):
                            yhat = self._model(x[0, b:b+1])
                            # compute all losses
                            loss = self._model.loss(y[0, b:b+1], yhat)
                            if np.isnan(loss.detach().cpu().numpy()):
                                print("\n" * 10)
                                print(f"Loss is {loss.detach().cpu().numpy()}, terminating training")
                                print("\n" * 10)
                                set_flag = False
                                break
                            # metric = 0.0
                            # if phase == "valid":
                            #     metric = self._compute_metric(x, xhat, latent)
                            # if in training, perform backpropagation and step the optimizer
                            if phase == "train":
                                loss.backward()
                                self._optimizer.step()
                        # update running losses
                        running_loss += loss.item() * x.size(0)
                # compute overall losses for the epoch and print results
                epoch_loss = self._compile_losses(running_loss, dataset_sizes[phase])
                print(f"{phase} Loss: {epoch_loss:.4f}")
                # if validation phase
                if phase == "valid":
                    # add losses to log
                    self._add_to_logs(val_loss=[epoch_loss])
                    # if best validation loss, update it and save the weights locally
                    if epoch_loss < best_metric:
                        counter = 0
                        best_metric = epoch_loss
                        best_model_wts = copy.deepcopy(self._model.state_dict())
                        if not self._is_tuner:
                            torch.save(self._model.state_dict(), self._path + "best_weights.pth")
                else:
                    # add training losses to log
                    self._add_to_logs(loss=[epoch_loss])
                # add checkpoint data
                print()
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
            if self._is_tuner:
                checkpoint_data = {
                    "epoch": epoch,
                    "net_state_dict": self._model.state_dict(),
                    "optimizer_state_dict": self._optimizer.state_dict(),
                }
                checkpoint = Checkpoint.from_dict(checkpoint_data)
                # report
                session.report(
                    {"loss": epoch_loss},
                    checkpoint=checkpoint,
                )
            if counter == self._patience:
                print("\n" * 10)
                print(f"Patience has been met, terminating training.")
                print("\n" * 10)
                break
        # compute total training time and print metrics
        if not training_flag and self._is_tuner:
            checkpoint_data = {
                "epoch": epoch,
                "net_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
            }
            checkpoint = Checkpoint.from_dict(checkpoint_data)
            session.report(
                {"loss": 100000},
                checkpoint=checkpoint,
            )
            print("Training terminated due to nan loss")
            return
        time_elapsed = time.time() - since
        print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed%60:.0f}s")
        print(f"Best val metric: {best_metric:.4f}")
        if not self._is_tuner:
            # save final weights and save logs
            self._save_weights()
        # load back in best weights
        self._model.load_state_dict(best_model_wts)
        if not self._is_tuner:
            # save these weights
            self._save_weights(best=True)

    def _save_weights(self, best: bool = False) -> None:
        """saves model weights and logs"""
        if best:
            torch.save(self._model.state_dict(), self._path + "best_weights.pth")
            with open(self._path + "logs.json", "w+") as f:
                json.dump(self.logs, f)
        else:
            torch.save(self._model.state_dict(), self._path + "final_weights.pth")









# class LSTM_Training:
#     """trainer class for an autoencoder; supports training for CVAEs, InfoVAEs, COAEs, or CVOAEs"""

#     def __init__(
#         self,
#         config: Dict[str, Union[int, str, float]],
#         model: torch.nn.Module,
#         is_tuner: bool,
#         patience: int,
#     ):
#         """initializes VAE_training object

#         :param config: config dictionary for both model and training parameters
#         :type config: Dict[str, Union[int, str, float]]
#         :param model: initialized VAE object
#         :type model: torch.nn.Module
#         :param is_tuner: True if running a hyperparameter tuner, False if not
#         :type is_tuner: bool
#         :param patience: number of epochs without improvement to terminate training
#         :type patience: int
#         """
#         if not is_tuner:
#             self._build_output(config)
#         self._config = config.get("training")
#         self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self._is_tuner = is_tuner
#         self._patience = patience
#         self.logs = {"loss": [], "val_loss": []}
#         self._model = model
#         self._initialize_model()

#     def _initialize_model(self) -> None:
#         """gets the optimizer and moves model to the device for training"""
#         self._optimizer = self._get_optimizer()(self._model.parameters(), lr=self._config.get("learning_rate"))
#         self._model.to(self._device)
#         if self._config.get("weight_path", None) is not None:
#             weights = torch.load(self._config.get("weight_path"))
#             self._model.load_state_dict(weights)
#             print("Loaded pretrained weights")
#         else:
#             print("No pretrained weights; they have been randomly initialized")
#         self._lr_scheduler = None
#         if (self._config.get("step_size", None) is not None) and (self._config.get("gamma", None) is not None):
#             step_size = self._config["step_size"]
#             gamma = self._config["gamma"]
#             self._lr_scheduler = torch.optim.lr_scheduler.StepLR(
#                 self._optimizer,
#                 step_size=step_size,
#                 gamma=gamma,
#                 verbose=True,
#             )

#     def _build_output(self, config: Dict[str, Union[int, str, float]]) -> None:
#         """sets the path, creates the output directory if it doesn't exist, and saves the config there

#         :param config: config dictionary for both model and training parameters
#         :type config: Dict[str, Union[int, str, float]]
#         """
#         self._path = "weights/" + config.get("training").get("model_name", "no_model_name_provided") + "/"
#         if not os.path.exists(self._path):
#             os.mkdir(self._path)
#         with open(self._path + "config.yaml", "w+") as f:
#             yaml.dump(config, f)

#     def _get_optimizer(self) -> torch.optim.Optimizer:
#         """converts the configured optimizer string to the optimizer object

#         :return: desired optimizer
#         :rtype: torch.optim.optimizer.Optimizer
#         """
#         opts = {
#             "adam": torch.optim.Adam,
#             "sgd": torch.optim.SGD,
#             "adagrad": torch.optim.Adagrad,
#             "rmsprop": torch.optim.RMSprop,
#             "nadam": torch.optim.NAdam,
#         }
#         return opts[self._config.get("optimizer").lower()]

#     def _add_to_logs(
#         self,
#         loss: List[float] = [],
#         val_loss: List[float] = [],
#     ) -> None:
#         """adds the current losses to log; set up to work for both training and validation with the
#         use of lists, list-addition, and defaults

#         :param loss: overall loss, defaults to []
#         :type loss: List[float], optional
#         :param val_loss: overall validation loss, defaults to []
#         :type val_loss: List[float], optional
#         """
#         self.logs["loss"] += loss
#         self.logs["val_loss"] += val_loss

#     def _compile_losses(
#         self,
#         running_loss: float,
#         dataset_size: int,
#     ) -> Tuple[Union[torch.FloatTensor, str]]:
#         """computes overall epoch loss for each of the individual losses; if a loss is zero, it will
#         return a string

#         :param running_loss: total summed loss over every sample in epoch
#         :type running_loss: float
#         :param dataset_size: size of dataset (either training or validation set)
#         :type dataset_size: int
#         :return: _description_
#         :rtype: Tuple[Union[torch.FloatTensor, str]]
#         """
#         epoch_loss = running_loss / dataset_size
#         return epoch_loss

#     def _compute_metric(self, x: torch.Tensor, xhat: torch.Tensor, latent: torch.Tensor) -> float:
#         """computes the custom metric

#         :param x: normalized density
#         :type x: torch.Tensor
#         :param xhat: normalized prediction
#         :type xhat: torch.Tensor
#         :param latent: latent representation
#         :type latent: torch.Tensor
#         """
#         x = x.detach().cpu().numpy()
#         xhat = xhat.detach().cpu().numpy()
#         latent = latent.detach().cpu().numpy()
#         metric = self._model.metric(x, xhat, latent)
#         return metric

#     def train(self, dataloaders: DataLoader, dataset_sizes: Dict[str, int]) -> None:
#         """trains the model based on the configured training scheme

#         :param dataloaders: dataloaders for both training and validation
#         :type dataloaders: DataLoader
#         :param dataset_sizes: number of samples in training and validation sets
#         :type dataset_sizes: Dict[str, int]
#         """
#         # get initial variables
#         since = time.time()
#         best_model_wts = copy.deepcopy(self._model.state_dict())
#         best_metric = np.inf
#         training_flag = True
#         counter = 0
#         # loop through epochs
#         for epoch in tqdm(range(self._config.get("epochs", 1))):
#             if not training_flag:
#                 break
#             counter += 1
#             print(f'Epoch {epoch+1}/{self._config.get("epochs", 1)}')
#             print("-" * 10)
#             phase_flag = True
#             # iterate over two phases
#             for phase in ["train", "valid"]:
#                 if not phase_flag:
#                     training_flag = False
#                     break
#                 # set model to training or evaluation mode
#                 if phase == "train":
#                     self._model.train()
#                 else:
#                     self._model.eval()
#                 # initialize losses
#                 running_loss = 0.0
#                 set_flag = True
#                 # iterate over all batches in the phase
#                 for x in tqdm(dataloaders[phase]):
#                     sample_weights = None
#                     if len(x) == 3:
#                         x, y, sample_weights = x[0], x[1], x[2]
#                         sample_weights = sample_weights.to(self._device)
#                     elif len(x) == 2:
#                         x, y = x[0], x[1]
#                     else:
#                         raise ValueError(f"Length of `x` expected to be 2 or 3. len(x) = {len(x)}")
#                     # move input to CPU or GPU
#                     x = x.to(self._device)
#                     y = y.to(self._device)
#                     if not set_flag:
#                         phase_flag = False
#                         break
#                     # set gradients to zero before running batch through the model
#                     self._optimizer.zero_grad()
#                     # have gradients enabled only in training phase
#                     with torch.set_grad_enabled(phase == "train"):
#                         yhat = self._model(x)
#                         # compute all losses
#                         loss = self._model.loss(y, yhat)
#                         if np.isnan(loss.detach().cpu().numpy()):
#                             print("\n" * 10)
#                             print(f"Loss is {loss.detach().cpu().numpy()}, terminating training")
#                             print("\n" * 10)
#                             set_flag = False
#                             break
#                         # metric = 0.0
#                         # if phase == "valid":
#                         #     metric = self._compute_metric(x, xhat, latent)
#                         # if in training, perform backpropagation and step the optimizer
#                         if phase == "train":
#                             loss.backward()
#                             self._optimizer.step()
#                     # update running losses
#                     running_loss += loss.item() * x.size(0)
#                 # compute overall losses for the epoch and print results
#                 epoch_loss = self._compile_losses(running_loss, dataset_sizes[phase])
#                 print(f"{phase} Loss: {epoch_loss:.4f}")
#                 # if validation phase
#                 if phase == "valid":
#                     # add losses to log
#                     self._add_to_logs(val_loss=[epoch_loss])
#                     # if best validation loss, update it and save the weights locally
#                     if epoch_loss < best_metric:
#                         counter = 0
#                         best_metric = epoch_loss
#                         best_model_wts = copy.deepcopy(self._model.state_dict())
#                         if not self._is_tuner:
#                             torch.save(self._model.state_dict(), self._path + "best_weights.pth")
#                 else:
#                     # add training losses to log
#                     self._add_to_logs(loss=[epoch_loss])
#                 # add checkpoint data
#                 print()
#             self._lr_scheduler.step()
#             if self._is_tuner:
#                 checkpoint_data = {
#                     "epoch": epoch,
#                     "net_state_dict": self._model.state_dict(),
#                     "optimizer_state_dict": self._optimizer.state_dict(),
#                 }
#                 checkpoint = Checkpoint.from_dict(checkpoint_data)
#                 # report
#                 session.report(
#                     {"loss": epoch_loss},
#                     checkpoint=checkpoint,
#                 )
#             if counter == self._patience:
#                 print("\n" * 10)
#                 print(f"Patience has been met, terminating training.")
#                 print("\n" * 10)
#                 break
#         # compute total training time and print metrics
#         if not training_flag and self._is_tuner:
#             checkpoint_data = {
#                 "epoch": epoch,
#                 "net_state_dict": self._model.state_dict(),
#                 "optimizer_state_dict": self._optimizer.state_dict(),
#             }
#             checkpoint = Checkpoint.from_dict(checkpoint_data)
#             session.report(
#                 {"loss": 100000},
#                 checkpoint=checkpoint,
#             )
#             print("Training terminated due to nan loss")
#             return
#         time_elapsed = time.time() - since
#         print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed%60:.0f}s")
#         print(f"Best val metric: {best_metric:.4f}")
#         if not self._is_tuner:
#             # save final weights and save logs
#             self._save_weights()
#         # load back in best weights
#         self._model.load_state_dict(best_model_wts)
#         if not self._is_tuner:
#             # save these weights
#             self._save_weights(best=True)

#     def _save_weights(self, best: bool = False) -> None:
#         """saves model weights and logs"""
#         if best:
#             torch.save(self._model.state_dict(), self._path + "best_weights.pth")
#             with open(self._path + "logs.json", "w+") as f:
#                 json.dump(self.logs, f)
#         else:
#             torch.save(self._model.state_dict(), self._path + "final_weights.pth")

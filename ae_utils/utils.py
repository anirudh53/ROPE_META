from ray.train import Checkpoint, session
from torch.utils.data import DataLoader
from typing import Dict, Union, List, Tuple
from scipy.stats import pearsonr
from string import ascii_lowercase
from tqdm import tqdm
import numpy as np
import torch
import h5py
import json
import yaml
import time
import copy
import os


LABELS = [f"{letter})" for letter in ascii_lowercase]


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


def load_h5(filename: str, return_sw: bool) -> np.ndarray:
    """loads a single hdf file and return the density array

    :param filename: name of TIEGCM file
    :type filename: str
    :param return_sw: whether or not you want the space weather inputs
    :type return_sw: bool
    :return: density array
    :rtype: np.ndarray
    """
    keys = ["DEN", "f107d", "Kp"] if return_sw else ["DEN"]
    output = {}
    with h5py.File(filename, "r") as hf:
        for key in keys:
            output[key] = np.array(hf.get(key))
    return output


def load_data(
    data_config: Dict[str, Union[str, List[int]]], test: bool, return_sw: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """loads data from all years for training and validation

    :param data_config: data section of the config
    :type data_config: Dict[str, Union[str, List[int]]]
    :param test: whether or not you are using the test set
    :type test: bool
    :param return_sw: whether or not you want the space weather inputs
    :type return_sw: bool
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
    for year in tqdm(training_years, "Loading training data"):
        if year == training_years[0]:
            loadedT = load_h5(f"{directory}{year}{suffix}", return_sw=return_sw)
        else:
            loaded0 = load_h5(f"{directory}{year}{suffix}", return_sw=return_sw)
            for key, value in loaded0.items():
                loadedT[key] = np.concatenate((loadedT[key], loaded0[key]), axis=0)
    for year in tqdm(validation_years, "Loading validation/test data"):
        if year == validation_years[0]:
            loadedV = load_h5(f"{directory}{year}{suffix}", return_sw=return_sw)
        else:
            loaded0 = load_h5(f"{directory}{year}{suffix}", return_sw=return_sw)
            for key, value in loaded0.items():
                loadedV[key] = np.concatenate((loadedV[key], loaded0[key]), axis=0)
    if return_sw:
        return (
            loadedT["DEN"],
            loadedV["DEN"],
            loadedT["f107d"],
            loadedV["f107d"],
            loadedT["Kp"],
            loadedV["Kp"],
        )
    return loadedT["DEN"], loadedV["DEN"]


def load_and_process_data(
    data_config: Dict[str, Union[str, List[int]]], test: bool = False, return_sw: bool = False
) -> Dict[str, torch.Tensor]:
    """loads TIE-GCM data, converts to log10 space, splits, and normalizes the data

    :param data_config: data section of the config
    :type data_config: Dict[str, Union[str, List[int]]]
    :param test: whether or not you are using the test set, defaults to False
    :type test: bool, optional
    :param return_sw: whether or not you want the space weather inputs, defaults to False
    :type return_sw: bool, optional
    :return: normalized training and validation data
    :rtype: Dict[str, torch.Tensor]
    """
    # load TIEGCM data for training and validation
    output = load_data(data_config, test=test, return_sw=return_sw)
    if return_sw:
        dataT, dataV, F10T, F10V, KpT, KpV = output
    else:
        dataT, dataV = output
    # convert to tensor
    dataT, dataV = torch.Tensor(np.log10(dataT)), torch.Tensor(np.log10(dataV))
    # add dimension of size 1 to act as a single channel
    dataT, dataV = dataT.unsqueeze(1), dataV.unsqueeze(1)
    # get normalization statistics
    mean, std = get_training_statistics(dataT)
    stats = {"mu": mean, "sigma": std}
    if test:
        data = {"test": normalize(dataV, mean, std)}
    else:
        data = {"train": normalize(dataT, mean, std), "valid": normalize(dataV, mean, std)}
    output = (data, stats)
    if return_sw:
        output += (F10T, F10V, KpT, KpV)
    return output


def get_Kp_weights(filename: str = None, Kp: np.ndarray = None) -> np.ndarray:
    """looks at Kp distributions and gets weighting factors for different samples based on their Kp;
    either uses a file or the data itself; cannot take both

    :param filename: path to file containing Kp for training period, defaults to None
    :type filename: str, optional
    :param kp: Kp data, defaults to None
    :type kp: np.ndarray, optional
    :return: weighting factors for each sample
    :rtype: np.ndarray
    """
    assert (filename is not None) or (Kp is not None)
    if Kp is None:
        with h5py.File(filename, "r") as hf:
            Kp = np.array(hf.get("Kp"))
    Kp[Kp == 9.0] = 8.99
    weights = [[], [], [], [], [], [], [], [], []]
    for i in range(len(Kp)):
        weights[int(np.floor(Kp[i]))].append(i)
    sums = [len(weight) for weight in weights]
    sums = [np.max(sums[i:]) for i in range(len(sums))]
    weights_S = len(Kp) / np.array(sums)
    weights_S[weights_S == np.inf] = 0
    weights_S = 100 / np.sum(weights_S) * weights_S
    # Create sample weighting array
    sample = np.zeros(len(Kp), dtype="float32")
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            sample[weights[i][j]] = weights_S[int(np.floor(Kp[weights[i][j]]))]
    return sample


class AE_Training:
    """trainer class for an autoencoder; supports training for CVAEs, InfoVAEs, COAEs, or CVOAEs"""

    def __init__(
        self,
        config: Dict[str, Union[int, str, float]],
        model: torch.nn.Module,
        is_tuner: bool,
        patience: int,
        stats: Dict[str, np.ndarray],
        spatial_weights: torch.Tensor = None,
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
        :param stats: training statistics
        :type stats: Dict[str, np.ndarray]
        :param spatial_weights: multiplicative factor to put emphasis on certain spatial areas, defaults to None
        :type spatial_weights: torch.Tensor, optional
        """
        if not is_tuner:
            self._build_output(config)
        self._config = config["training"]
        self._model_config = config["model"]
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._is_tuner = is_tuner
        self._patience = patience
        self._alpha = self._model_config["alpha"]
        self._bottleneck_size = self._model_config["bottleneck_size"]
        self._spatial_weights = spatial_weights
        self._stats = stats
        self._normalization_term = 1.0
        if spatial_weights is not None:
            self._normalization_term = spatial_weights.mean()
            self._spatial_weights = self._spatial_weights.to(self._device)
        self.logs = {"loss": [], "val_loss": [], "MSE": [], "ORTH": [], "METRIC": []}
        self._model = model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """gets the optimizer and moves model to the device for training"""
        self._optimizer = self._get_optimizer()(
            self._model.parameters(), lr=self._config.get("learning_rate")
        )
        self._model.to(self._device)
        if self._config.get("weight_path", None) is not None:
            weights = torch.load(self._config.get("weight_path"))
            self._model.load_state_dict(weights)
            print("Loaded pretrained weights")
        else:
            print("No pretrained weights; they have been randomly initialized")
        self._lr_scheduler = None
        if (self._config.get("step_size", None) is not None) and (
            self._config.get("gamma", None) is not None
        ):
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
        self._path = (
            "weights/" + config.get("training").get("model_name", "no_model_name_provided") + "/"
        )
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
        mse: List[float] = [],
        orth: List[float] = [],
        metric: List[float] = [],
    ) -> None:
        """adds the current losses to log; set up to work for both training and validation with the
        use of lists, list-addition, and defaults

        :param loss: overall loss, defaults to []
        :type loss: List[float], optional
        :param val_loss: overall validation loss, defaults to []
        :type val_loss: List[float], optional
        :param mse: mean square error loss, defaults to []
        :type mse: List[float], optional
        :param orth: orthogonal loss, defaults to []
        :type orth: List[float], optional
        """
        self.logs["loss"] += loss
        self.logs["val_loss"] += val_loss
        self.logs["MSE"] += mse
        self.logs["ORTH"] += orth
        self.logs["METRIC"] += metric

    def _compile_losses(
        self,
        running_loss: float,
        running_MSE: float,
        running_ORTH: float,
        running_METRIC: float,
        dataset_size: int,
    ) -> Tuple[Union[torch.FloatTensor, str]]:
        """computes overall epoch loss for each of the individual losses; if a loss is zero, it will
        return a string

        :param running_loss: total summed loss over every sample in epoch
        :type running_loss: float
        :param running_MSE: total summed MSE loss over every sample in epoch
        :type running_MSE: float
        :param running_ORTH: total summed orthogonal loss over every sample in epoch
        :type running_ORTH: float
        :param running_METRIC: total summed metric over every sample in epoch
        :type running_METRIC: float
        :param dataset_size: size of dataset (either training or validation set)
        :type dataset_size: int
        :return: _description_
        :rtype: Tuple[Union[torch.FloatTensor, str]]
        """
        epoch_loss = running_loss / dataset_size
        epoch_MSE = running_MSE / dataset_size
        epoch_ORTH = running_ORTH / dataset_size
        epoch_METRIC = running_METRIC / dataset_size
        return epoch_loss, epoch_MSE, epoch_ORTH, epoch_METRIC

    def _compute_metric(self, x: np.ndarray, xhat: np.ndarray, latent: np.ndarray) -> float:
        """custom metric for the COAE based on density error and latent space orthogonality

        :param x: model input / desired output
        :type x: np.ndarray
        :param xhat: model prediction
        :type xhat: np.ndarray
        :param latent: latent representation
        :type latent: np.ndarray
        :return: custom metric
        :rtype: float
        """
        x = x.detach().cpu().numpy()
        xhat = xhat.detach().cpu().numpy()
        latent = latent.detach().cpu().numpy()
        # denormalize density
        x = np.power(10, denormalize(x, self._stats))
        xhat = np.power(10, denormalize(xhat, self._stats))
        # compute normalized error component
        mae = np.mean(np.divide(np.abs(x - xhat), x))
        error_metric = (
            10 * mae
        )  # ideally error should be between 2% - 3%; this makes the metric about 0.2 - 0.3
        # compute orthogonality metric
        corr = np.zeros([self._bottleneck_size, self._bottleneck_size])
        for i in range(corr.shape[0]):
            for j in range(corr.shape[1]):
                corr[i, j] = pearsonr(latent[:, i], latent[:, j])[0]
        corr -= np.eye(self._bottleneck_size)
        orth_metric = np.mean(np.abs(corr))
        return 2 * error_metric + orth_metric  # original

    def loss(
        self,
        x: torch.Tensor,
        xhat: torch.Tensor,
        latent: torch.Tensor,
        sample_weights: torch.Tensor = None,
    ) -> Tuple[torch.FloatTensor]:
        """loss function for the COAE

        :param x: model input / desired output
        :type x: torch.Tensor
        :param xhat: model prediction
        :type xhat: torch.Tensor
        :param latent: latent representation
        :type latent: torch.Tensor
        :param sample_weights: sample weights for the batch, defaults to None
        :type sample_weights: torch.Tensor, optional
        :return: overall loss, MSE loss, and orthogonal loss for the batch
        :rtype: Tuple[torch.FloatTensor]
        """
        # compute basic MSE loss
        # mse_loss = torch.mean((x - xhat).pow(2), dim=tuple(range(1,5)))
        mse_loss = (x - xhat).pow(2)
        if self._spatial_weights is not None:
            mse_loss *= self._spatial_weights / self._normalization_term
        if sample_weights is not None:
            mse_loss *= sample_weights
        mse_loss = mse_loss.mean()
        # return mse_loss
        # compute orthogonal loss
        orthogonal_loss = (
            (
                torch.matmul(torch.transpose(latent, 0, 1), latent)
                - torch.eye(self._bottleneck_size).to(self._device)
            ).pow(2)
        ).mean()
        return (
            mse_loss + self._alpha * orthogonal_loss,
            mse_loss,
            orthogonal_loss,
        )

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
                running_MSE = 0.0
                running_ORTH = 0.0
                running_METRIC = 0.0
                set_flag = True
                # iterate over all batches in the phase
                for x in tqdm(dataloaders[phase]):
                    sample_weights = None
                    if isinstance(x, list):
                        sample_weights = x[1]
                        x = x[0]
                        sample_weights = sample_weights.to(self._device)
                    # move input to CPU or GPU
                    x = x.to(self._device)
                    if not set_flag:
                        phase_flag = False
                        break
                    # set gradients to zero before running batch through the model
                    self._optimizer.zero_grad()
                    # have gradients enabled only in training phase
                    with torch.set_grad_enabled(phase == "train"):
                        xhat, latent = self._model(x)
                        # compute all losses
                        loss, mse, orth = self.loss(x, xhat, latent, sample_weights=sample_weights)
                        if np.isnan(loss.detach().cpu().numpy()):
                            print("\n" * 10)
                            print(f"Loss is {loss.detach().cpu().numpy()}, terminating training")
                            print("\n" * 10)
                            set_flag = False
                            break
                        metric = 0.0
                        if phase == "valid":
                            metric = self._compute_metric(x, xhat, latent)
                        # if in training, perform backpropagation and step the optimizer
                        if phase == "train":
                            loss.backward()
                            self._optimizer.step()
                    # update running losses
                    running_loss += loss.item() * x.size(0)
                    running_MSE += mse.item() * x.size(0)
                    running_ORTH += orth.item() * x.size(0)
                    running_METRIC += metric * x.size(0)
                # compute overall losses for the epoch and print results
                epoch_loss, epoch_MSE, epoch_ORTH, epoch_METRIC = self._compile_losses(
                    running_loss,
                    running_MSE,
                    running_ORTH,
                    running_METRIC,
                    dataset_sizes[phase],
                )
                print(
                    f"{phase} Loss: {epoch_loss:.4f} MSE: {epoch_MSE:.4f} ORTH: {epoch_ORTH:.4f} METRIC: {epoch_METRIC:.4f}"
                )
                # if validation phase
                if phase == "valid":
                    # add losses to log
                    self._add_to_logs(
                        val_loss=[epoch_loss],
                        mse=[epoch_MSE],
                        orth=[epoch_ORTH],
                        metric=[epoch_METRIC],
                    )
                    # if best validation loss, update it and save the weights locally
                    if epoch_METRIC < best_metric:
                        counter = 0
                        best_metric = epoch_METRIC
                        best_model_wts = copy.deepcopy(self._model.state_dict())
                        if not self._is_tuner:
                            torch.save(self._model.state_dict(), self._path + "best_weights.pth")
                else:
                    # add training losses to log
                    self._add_to_logs(loss=[epoch_loss])
                # add checkpoint data
                print()
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
                    {"loss": epoch_METRIC},
                    checkpoint=checkpoint,
                )
            if counter == self._patience:
                print("\n" * 10)
                print("Patience has been met, terminating training.")
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

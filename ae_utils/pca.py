from typing import Dict, Tuple, Any,Union
import matplotlib.pyplot as plt
from ae_utils import utils
import pandas as pd
import numpy as np
import torch
import yaml
import h5py

class PCA_Handler:
    """loads data from previous PCA run"""

    def __init__(self, config_filename: str, save_filename: str, num_coeffs: int = 10):
        """initializes PCA_Manager class

        :param config_filename: path to config file
        :type config_filename: str
        :param save_filename: path to save PCA data to
        :type save_filename: str
        :param num_coeffs: number of PCA coefficients, defaults to 10
        :type num_coeffs: int, optional
        """
        self._config = self._load_yaml(config_filename)["data"]
        self._save_filename = save_filename
        print("Loading data")
        data = self._load_data()
        self._train, self._test, inputs = data
        self.F10T, self.F10Te, self.KpT, self.KpTe = inputs
        self.data = self._load_h5(save_filename, num_coeffs)
        print("Data loaded, PCA_Manager initialized")

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """load a yaml file (e.g. config)

        :param filename: filename for yaml you want to load
        :type filename: str
        :return: dictionary from yaml
        :rtype: Dict[str, Any]
        """
        with open(filename, "r") as file:
            data = yaml.safe_load(file)
        return data

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """load data as specified by the config

        :return: train and test density data
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        train_data, stats, F10T, _, KpT, _ = utils.load_and_process_data(self._config, test=False, return_sw=True)
        test_data, _, _, F10Te, _, KpTe = utils.load_and_process_data(self._config, test=True, return_sw=True)
        train_data = utils.denormalize(train_data["train"], stats)
        test_data = utils.denormalize(test_data["test"], stats)
        return train_data, test_data, (F10T, F10Te, KpT, KpTe)

    def _load_h5(self, filename: str, num_coeffs: int) -> np.ndarray:
        """loads a single hdf file and return the density array

        :param filename: name of TIEGCM file
        :type filename: str
        :param num_coeffs: number of PCA coefficients to use
        :type num_coeffs: int
        :return: density array
        :rtype: np.ndarray
        """
        data = {"coeff": [], "U": [], "S": [], "mean_density": []}
        with h5py.File(filename, "r") as hf:
            for key in data.keys():
                data[key] = np.array(hf.get(key))
        data["coeff"] = data["coeff"][:,:num_coeffs]
        data["U"] = data["U"][:,:num_coeffs]
        return data
    
    def convert_coeff_to_den(self, data: Union[np.ndarray, None] = None) -> np.ndarray:
        """converts an array of PCA coefficients to the corresponding density array

        :param data: coefficients to convert to density
        :type data: Union[np.ndarray, None]
        :return: density array
        :rtype: np.ndarray
        """
        if data is None:
            reconstructed = np.power(10, np.matmul(self.data["coeff"], self.data["U"].T) + self.data["mean_density"])
        else:
            reconstructed = np.power(10, np.matmul(data, self.data["U"].T) + self.data["mean_density"])
        if isinstance(reconstructed, torch.Tensor):
            return reconstructed.numpy()
        else:
            return reconstructed
        
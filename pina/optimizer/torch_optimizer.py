""" This module contains the TorchOptimizer class. """

import torch

from .optimizer_interface import OptimizerInterface
from ..utils import check_consistency


class TorchOptimizer(OptimizerInterface):
    """
    This class is used to create a torch optimizer by passing the optimizer
    class and its keyword arguments. The optimizer is then hooked to the
    parameters of the model by using the `hook` method.
    """

    def __init__(self, optimizer_class, **kwargs):
        """
        The constructor of the TorchOptimizer class.

        :param optimizer_class: The optimizer class to use. It must be a
            subclass of `torch.optim.Optimizer`. 
        :type optimizer_class: torch.optim.Optimizer
        :param kwargs: The keyword arguments to pass to the optimizer
            constructor.
        """
        check_consistency(optimizer_class, torch.optim.Optimizer, subclass=True)

        self.optimizer_class = optimizer_class
        self.kwargs = kwargs

    def hook(self, parameters):
        """ 
        It initializes the optimizer instance and hooks it to the given
        parameters.
        """
        self.optimizer_instance = self.optimizer_class(
            parameters, **self.kwargs
        )
""" This module contains the implementation of the TorchScheduler class. """

import torch

try:
    from torch.optim.lr_scheduler import LRScheduler  # torch >= 2.0
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler  # torch < 2.0

from ..optimizer import OptimizerInterface
from .scheduler_interface import SchedulerInterface
from ..utils import check_consistency


class TorchScheduler(SchedulerInterface):
    """ 
    This class is used to create a torch scheduler by passing the scheduler 
    class and its keyword arguments. The scheduler is then hooked to the
    optimizer by using the `hook` method.
    """
    def __init__(self, scheduler_class, **kwargs):
        """ 
        The constructor of the TorchScheduler class.
        
        :param scheduler_class: The scheduler class to use. It must be a
            subclass of `torch.optim.lr_scheduler.LRScheduler`.
        :type scheduler_class: torch.optim.lr_scheduler.LRScheduler
        :param kwargs: The keyword arguments to pass to the scheduler
            constructor.
        """
        check_consistency(scheduler_class, LRScheduler, subclass=True)

        self.scheduler_class = scheduler_class
        self.kwargs = kwargs

    def hook(self, optimizer):
        """
        It initializes the scheduler instance and hooks it to the given
        scheduler
        """
        check_consistency(optimizer, OptimizerInterface)

        self.scheduler_instance = self.scheduler_class(
            optimizer, **self.kwargs
        )

    
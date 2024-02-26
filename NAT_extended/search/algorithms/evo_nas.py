from abc import ABC, abstractmethod

import numpy as np

from NAT_extended.search.algorithms.utils import calc_hv

__all__ = ['EvoNAS']


class EvoNAS(ABC):

    def __init__(
         self,
         search_space,
         evaluator,
         objs='acc&flops',  # objectives to be optimized,
         # choices=['acc', 'flops', 'params', 'latency', 'acc&flops', 'acc&params', 'acc&flops&params', etc]
         pop_size=100,  # the population size
         max_gens=30,  # maximum number of generations/iteration
    ):

        self.search_space = search_space
        self.evaluator = evaluator

        # determine the objectives for optimization
        self.objs = objs
        self.n_objs = len(objs.split('&'))
        self.pop_size = pop_size
        self.max_gens = max_gens

    def _eval(self, archs):
        """
        :param archs: architecture string which is understandable by evaluator directly,
        note that this is NOT decision variables x used during search
        e.g., archs = self.search_space.decode(x)
        :return: [
            {"acc": 53, "params": 123, "flops": 73, "latency": 62},
            {"acc": 32, "params": 47, "flops": 12, "latency": 65},
            ...
            ]
        """
        stats = self.evaluator.evaluate(_subnets=archs, objs=self.objs)
        return stats

    @staticmethod
    def get_attributes(archive, _attr):

        """
        :param archive: [{"acc": 53, "params": 123, "flops": 73, "latency": 62, "arch":{"r":64, "w":1.2, ... }},{...}]
        :param _attr: string with &-separated attributes to return
        :return: [[acc/err, params, flops, latency, arch], [acc/err, params, flops, latency, arch], ...]
        list of list of specified attributes (in _attr) for each element in the archive
        """

        attr_keys = _attr.split('&')
        batch_attr_values = []
        for member in archive:
            attr_values = []

            for attr_key in attr_keys:
                if attr_key == 'err':
                    attr_values.append(100 - member['acc'])
                else:
                    attr_values.append(member[attr_key])

            batch_attr_values.append(attr_values)
        return batch_attr_values

    @abstractmethod
    def _sample_initial_population(self, sample_size):
        """ method to sample the initial population """
        raise NotImplementedError

    def initialization(self):

        """
        :return: [
            {"arch":{"r":64, "w":1.2, ... }, "acc": 53, "params": 123, "flops": 73, "latency": 62 },
            {"arch":{decoded subnet dictionary}, "acc": 32, "params": 47, "flops": 12, "latency": 65 },
            ...
            ]
        """

        archive = []  # initialize an empty archive to store the architectures
        archs = self._sample_initial_population(self.pop_size)  # initialize initial population
        stats_dict = self._eval(archs)  # evaluate initial population

        # store evaluated / trained architectures
        for arch, stats in zip(archs, stats_dict):
            archive.append({'arch': arch, **stats})

        return archive

    @abstractmethod
    def _next(self, **kwargs):
        """ method to run one generation / iteration of evolution """
        raise NotImplementedError

    def _calc_hv(self, archive, ref_pt):
        # reference point (nadir point) for calculating hypervolume
        hv = calc_hv(
            ref_pt=ref_pt,
            F=np.array(self.get_attributes(archive, _attr=self.objs.replace('acc', 'err')))
        )
        return hv

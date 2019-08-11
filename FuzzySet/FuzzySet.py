import numpy as np


class FuzzySet(object):
    def __init__(self, universe_range, analysis_params=None, granularity=500, activation_function='gauss'):
        self.tol = np.divide(universe_range[1] - universe_range[0], granularity)
        # The range of the universe
        self.universe = np.arange(universe_range[0], universe_range[1] + self.tol, granularity)
        # The output of the universe
        self.membership_function = None
        self.analysis_params = analysis_params
        self.activation_function = activation_function
        self.params = ['mean', 'sigma', 'data']
        if self.analysis_params is not None:
            for key in self.params:
                if key not in self.analysis_params:
                    if key == 'mean':
                        self.analysis_params['mean'] = self.activate(mode='mean')
                    elif key == 'sigma':
                        self.analysis_params['sigma'] = self.activate(mode='sigma')
        self.create_membership()

    def __or__(self, other):
        # At this point, we know the universe and the function that comes with it.
        # We can recalculate here with the sum of granularities
        # Max of each function gets the priority
        new_universe = []
        for universe_a, universe_b in zip(self.universe, other.universe):
            pass

    def activate(self, mode=None):
        """
        If there are no analysis parameters given, eg no user input of mean or sigma, we will update those based on the activation function.
        """
        if self.activation_function == 'gauss':
            if mode == 'mean':
                return np.mean(self.universe)
            if mode == 'sigma':
                return np.sigma(self.universe)
        elif self.activation_function == 'composite_gauss':
            if mode == 'mean':
                return np.mean(self.universe)
            if mode == 'sigma':
                return np.sigma(self.universe)
        else:
            raise Exception("Hey man you need a proper activation function here")

    def create_membership(self):
        """
        This function will create the membership output (Y) given a universe (X)
        """
        if self.activation_function == 'gauss':
            self.membership_function = gaussian(self.universe, self.analysis_params['mean'], self.analysis_params['sigma'])
        elif self.activation_function == 'composite_gauss':
            if self.analysis_params.get('data'):
                composite_gauss = CompositeGauss(self.universe, self.analysis_params['data'], self.analysis_params['mean'])
                self.membership_function = composite_gauss.composite_gaussian()
                if composite_gauss.new_universe is not None:
                    self.universe = composite_gauss.new_universe
        else:
            raise Exception("Hey man you need a proper membership function here")
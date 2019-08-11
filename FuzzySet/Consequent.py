from FuzzySet.FuzzySet import FuzzySet


class Consequent(FuzzySet):
    def __init__(self, universe):
        super(Consequent, self).__init__(universe, analysis_params, activation_function=activation_function)



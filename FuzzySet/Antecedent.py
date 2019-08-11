from FuzzySet import FuzzySet
from MembershipFunctions.ActivationFunctions import gaussian
from MembershipFunctions.CompositeGauss import CompositeGauss

class Antecedent(FuzzySet):
    def __init__(self, universe, analysis_params, activation_function='gauss'):
        super(Antecedent, self).__init__(universe, analysis_params, activation_function=activation_function)
        
        



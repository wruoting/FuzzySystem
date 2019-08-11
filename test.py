from RuleSystem.Rule import Rule
from FuzzySet.Antecedent import Antecedent

analysis_params = {

}

a = Antecedent([0, 1, 2], analysis_params)
b = Antecedent([1, 2, 3], analysis_params)

Rule(a | b)

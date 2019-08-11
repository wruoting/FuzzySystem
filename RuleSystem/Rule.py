"""
rule.py : Contains structure to create fuzzy rules.

Most notably, contains the `Rule` object which is used to connect atecedents
with conqeuents in a `ControlSystem`.
"""
from __future__ import print_function, division

import numpy as np
import networkx as nx

class Rule(object):
    """
    Rule in a fuzzy control system, connecting antecedent(s) to consequent(s).

    Parameters
    ----------
    antecedent : Antecedent term(s) or logical combination thereof, optional
        Antecedent terms serving as inputs to this rule. Multiple terms may
        be combined using operators `|` (OR), `&` (AND), `~` (NOT), and
        parentheticals to group terms.
    consequent : Consequent term(s) or logical combination thereof, optional
        Consequent terms serving as outputs from this rule. Multiple terms may
        be combined using operators `|` (OR), `&` (AND), `~` (NOT), and
        parentheticals to group terms.
    label : string, optional
        Label to reference the meaning of this rule. Optional, but recommended.
        If provided, the label must be unique among rules in any particular
        `ControlSystem`.

    Notes
    -----
    Fuzzy Rules can be completely built on instantatiation or one can begin
    with an empty Rule and construct interactively by setting `.antecedent`,
    `.consequent`, and `.label` variables.
    """


    def __init__(self, antecedent=None, consequent=None, label=None,
                 and_func=np.fmin, or_func=np.fmax):
        """
        Rule in a fuzzy system, connecting antecedent(s) to consequent(s).

        Parameters
        ----------
        antecedent : Antecedent term(s) or combination thereof, optional
            Antecedent terms serving as inputs to this rule. Multiple terms may
            be combined using operators `|` (OR), `&` (AND), `~` (NOT), and
            parentheticals to group terms.
        consequent : Consequent term(s) or combination thereof, optional
            Consequent terms serving as outputs from this rule. Multiple terms
            may be combined using operators `|` (OR), `&` (AND), `~` (NOT), and
            parentheticals to group terms.
        label : string, optional
            Label to reference the meaning of this rule. Optional, but
            recommended.
        and_func : function, optional
            Function which accepts multiple floating-point arguments and
            returns a single value. Defalts to NumPy function `fmin`, to
            support both single values and arrays. For multiplication,
            substitute `fuzz.control.mult` or `np.multiply`.
        or_func : function, optional
            Function which accepts multiple floating-point arguments and
            returns a single value. Defalts to NumPy function `fmax`, to
            support both single values and arrays.
        """
        self.and_func = and_func
        self.or_func = or_func

        self._antecedent = None
        self._consequent = None
        if antecedent is not None:
            self.antecedent = self.parse_logic(antecedent)
        if consequent is not None:
            self.consequent = consequent

        if label is not None:
            self.label = label
        else:
            self.label = id(self)

    def __repr__(self):
        """
        Concise, readable summary of the fuzzy rule.
        """
        if len(self.consequent) == 1:
            cons = self.consequent[0]
        else:
            cons = self.consequent

        return ("IF {0} THEN {1}"
                "\n\tAND aggregation function : {2}"
                "\n\tOR aggregation function  : {3}").format(
                    self.antecedent, cons,
                    self.and_func.__name__,
                    self.or_func.__name__)
    
    def parse_logic(self, antecedent):
        print(type(antecedent))
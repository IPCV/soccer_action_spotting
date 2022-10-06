import os
import sys
from jinja2 import Environment
from jinja2 import FileSystemLoader
from copy import deepcopy

import numpy as np
from addict import Dict

MOD_CURRENT_PATH = sys.modules[__name__].__file__
PATH = os.path.dirname(os.path.realpath(MOD_CURRENT_PATH))
TEMPLATE_ENVIRONMENT = Environment(autoescape=False, loader=FileSystemLoader(PATH), trim_blocks=False)

SoccerNetV2 = Dict({'classes': ["Ball out",
                                "Throw-in",
                                "Foul",
                                "Ind. free-kick",
                                "Clearance",
                                "Shots on tar.",
                                "Shots off tar.",
                                "Corner",
                                "Substitution",
                                "Kick-off",
                                "Yellow card",
                                "Offside",
                                "Dir. free-kick",
                                "Goal",
                                "Penalty",
                                "Yel.->Red",
                                "Red card"],
                    'mapping': [8, 9, 10, 11, 7, 5, 6, 13, 3, 1, 14, 4, 12, 2, 0, 16, 15],
                    'num_classes': 17})


class MeanAvgPrecisionTable:
    def __init__(self, results):
        self.html_template = TEMPLATE_ENVIRONMENT.get_template('meanAvgPrecision_template.html')
        self.latex_template = TEMPLATE_ENVIRONMENT.get_template('meanAvgPrecision_template.tex')

        self.results = deepcopy(results)

        self.best = {'a_mAP': 0,
                     'a_mAP_visible': 0,
                     'a_mAP_unshown': 0,
                     'a_mAP_per_class': np.zeros(SoccerNetV2.num_classes)}

        for r in self.results:
            r['a_mAP'] *= 100
            r['a_mAP_visible'] *= 100
            r['a_mAP_unshown'] *= 100
            r['a_mAP_per_class'] = [100 * mAP for mAP in r['a_mAP_per_class']]
            r['a_mAP_per_class'] = [r['a_mAP_per_class'][i] for i in SoccerNetV2.mapping]

            self.best['a_mAP'] = max(self.best['a_mAP'], r['a_mAP'])
            self.best['a_mAP_visible'] = max(self.best['a_mAP_visible'], r['a_mAP_visible'])
            self.best['a_mAP_unshown'] = max(self.best['a_mAP_unshown'], r['a_mAP_unshown'])
            self.best['a_mAP_per_class'] = np.max((self.best['a_mAP_per_class'], r['a_mAP_per_class']), axis=0)

    def _repr_html_(self):
        return self.html_template.render(classes=SoccerNetV2.classes,
                                         results=self.results,
                                         best=self.best)

    def _repr_latex_(self):
        return self.latex_template.render(classes=SoccerNetV2.classes,
                                          results=self.results,
                                          best=self.best)

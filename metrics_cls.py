# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cgi import test
import datasets
import json
import pdb
from collections import defaultdict
import pandas as pd
import numpy as np


_DESCRIPTION = """\
This a metric for the paper "Data Augmentation for Rare Symptoms in Vaccine Side-Effect Detection" (BioNLP 2022)
We evaluate on three test sets: Full, CUI-mapped, and Long-tail
"""

_KWARGS_DESCRIPTION = """
To be updated
"""

class PrecRec():
    def __init__(self, test_type) -> None:
        self.test_type = test_type
        self.global_tp = 0
        self.n_target_doc = 0
        self.global_n_pos = 0
        self.global_n_true = 0

        # for macro precision over whole symptom texts
        self.p_sum = 0
        # for macro recall
        self.r_sum = 0

        self.tp_macro = defaultdict(int)
        self.n_positive_macro = defaultdict(int)
        self.n_true_macro = defaultdict(int)

    def add_tp_macro(self, target_class):
        self.tp_macro[target_class] += 1 

    def add_pos_macro(self, target_class):
        self.n_positive_macro[target_class] += 1

    def add_true_macro(self, target_class):
        self.n_true_macro[target_class] += 1

    def macro_precision(self):
        return np.mean(
            [(self.tp_macro[sym] / self.n_positive_macro[sym]) if self.n_positive_macro[sym] != 0 else 0 for sym in self.n_positive_macro.keys()]
        )
    
    def macro_recall(self):
        return np.mean(
            [(self.tp_macro[sym] / self.n_true_macro[sym]) if self.n_true_macro[sym] != 0 else 0 for sym in self.n_true_macro.keys()]
        )
    
    def macro_f1(self):
        p = self.macro_precision()
        r = self.macro_recall()
        return 2*p*r/(p+r) if (p+r) != 0 else 0

    def micro_precision(self):
        return self.global_tp/self.global_n_pos if self.global_n_pos != 0 else 0

    def micro_recall(self):
        return self.global_tp/self.global_n_true if self.global_n_true != 0 else 0
    
    def micro_f1(self):
        p = self.micro_precision()
        r = self.micro_recall()
        return 2*p*r/(p+r) if (p+r) != 0 else 0

    def export2csv(self, out_path):
        macro_prec = {}
        for sym in self.n_positive_macro.keys():
            macro_prec[sym] = (self.tp_macro[sym] / self.n_positive_macro[sym]) if self.n_positive_macro[sym] != 0 else 0
        macro_recall = {}
        for sym in self.n_true_macro.keys():
            macro_recall[sym] = (self.tp_macro[sym] / self.n_true_macro[sym]) if self.n_true_macro[sym] != 0 else 0

        prec = pd.Series(macro_prec)
        recall = pd.Series(macro_recall)
        tp = pd.Series(self.tp_macro)
        pos = pd.Series(self.n_positive_macro)
        trues = pd.Series(self.n_true_macro)

        frames = [tp, pos, trues, prec, recall]
        result = pd.concat(frames, axis=1, join='outer')
        result.set_axis(["tp", "pred", "true", "precision", "recall"], axis=1, inplace=True)

        result.to_csv(out_path)
        return result

    def get_scores(self):
        return {
            "macro_presicion": self.macro_precision(),
            "macro_recall": self.macro_recall(),
            "micro_precision": self.micro_precision(),
            "micro_recall": self.micro_recall(),
            "macro_f1": self.macro_f1(),
            "micro_f1": self.micro_f1(),
            "global_tp": self.global_tp,
            "global_n_pos": self.global_n_pos,
            "global_n_true": self.global_n_true,
            "n_target_doc": self.n_target_doc,
        }


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class VSEDMetric(datasets.Metric):
    """ Metric for precision, recall and F1 on three test sets: Full, CUI-mapped, and Long-tail"""

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=None,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Sequence(feature=datasets.Value(dtype='float')),
                'references': datasets.Sequence(feature=datasets.Value(dtype='float')),
            }),
            # Additional links to the codebase or references
            codebase_urls=["https://github.com/huggingface/datasets/blob/master/templates/new_metric_script.py"],
            # To be updated
            # reference_urls=["http://path.to.reference.url/new_metric"] 
        )

    def _download_and_prepare(self, dl_manager):
        # evaluation for long-tail
        sympfile = "data/symptoms.tsv"
        symp_df = pd.read_csv(sympfile, sep="\t")
        
        symptoms_cui = list()
        for idx, row in symp_df.iterrows(): 
            symid = int(row["symptomid"])

            if type(row["cui"]) == type("str") and len(row["cui"]) != 0:
                symptoms_cui.append(symid)

        self.cui_mask = np.zeros(len(symp_df), dtype=bool)
        self.cui_mask[np.array(symptoms_cui)] = True

    def _compute(self, predictions, references):
        '''
        predictions: [1000, # of symps]
        references: [1000, # of symps]
        '''
        """Returns the scores"""
        test_type = ["full", "cui", "longtail"]
        
        full, cui, longtail = PrecRec("FULL"), PrecRec("CUI"), PrecRec("LONGTAIL")
        test_class = [full, cui, longtail]

        predictions_full = np.array(predictions)
        references_full = np.array(references)

        for tt, cc in zip(test_type, test_class):
            if tt == "full":
                predictions = predictions_full
                references = references_full
            elif tt == "cui":
                predictions = predictions_full[:, self.cui_mask]
                references = references_full[:, self.cui_mask]
            elif tt == "longtail":
                predictions = predictions_full[:, self.lt_mask]
                references = references_full[:, self.lt_mask]

            for pred, ref in zip(predictions, references):
                # the number of gold entities
                local_gold_ent = np.nonzero(ref)[0]
                
                if len(local_gold_ent) > 0:
        
                    for ge in local_gold_ent:
                        cc.add_true_macro(ge)
                        
                    local_n_true = len(local_gold_ent)  # the number of gold entities per each example
                    cc.global_n_true += local_n_true
                    cc.n_target_doc += 1

                    # get entities from generated texts
                    model_outputs = np.nonzero(pred)[0]
                    pred_symps = []
                    for mo in model_outputs:
                        pred_symps.append(mo)
                        cc.add_pos_macro(mo)

                    local_n_pos = len(pred_symps)
                    cc.global_n_pos += local_n_pos

                    # get true positives
                    local_tp = 0
                    tp_symps = set(local_gold_ent) & set(pred_symps) # true positive
                    n_tp_symps = len(tp_symps)
                    if n_tp_symps > 0:
                        local_tp += n_tp_symps
                        cc.global_tp += n_tp_symps
                        for ts in tp_symps:
                            cc.add_tp_macro(ts)

                    # compute local precision and recall for macro metrics
                    cur_p = (local_tp / local_n_pos) if local_n_pos != 0 else 0
                    cur_r = (local_tp / local_n_true) if local_n_true != 0 else 0
                    cc.p_sum += cur_p
                    cc.r_sum += cur_r

        results = {}
        for tt, cc in zip(test_type, [full, cui, longtail]):
            results[tt] = cc.get_scores()
            csv_out = f"result_{tt}.csv"
            cc.export2csv(csv_out)
            
        return results

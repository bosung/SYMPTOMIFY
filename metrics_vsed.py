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

# TODO: Add BibTeX citation
_CITATION = """\
}
"""

_DESCRIPTION = """\
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

    def macro_precision(self):
        return self.p_sum/self.n_target_doc if self.n_target_doc != 0 else 0
    
    def macro_recall(self):
        return self.r_sum/self.n_target_doc if self.n_target_doc != 0 else 0
    
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
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': {
                        "vid": datasets.Value("int32"),
                        "symptoms": datasets.features.Sequence(datasets.Value('string')),
                }
                ,
            }),
            # Additional links to the codebase or references
            codebase_urls=["https://github.com/huggingface/datasets/blob/master/templates/new_metric_script.py"],
            # To be updated
            # reference_urls=["http://path.to.reference.url/new_metric"] 
        )

    def _download_and_prepare(self, dl_manager):
        self.norm2ori = {}  # normalized symptom name to original symtpom name

        # Full entity set test
        symptomfile = "data/VAERS/symptoms.tsv"
        self.symptoms = set()     
        for i, w in enumerate(open(symptomfile, "r")):
            if i == 0:
                continue
            _, sym, ori, _ = w.split("\t")
            self.symptoms.add(sym)
            self.norm2ori[sym] = ori

        # CUI-mapped entity set
        cuifile = "data/VAERS/symptoms_cui.tsv"
        self.symptoms_cui = set()
        for i, w in enumerate(open(cuifile, "r")):
            if i == 0:
                continue
            tokens = w.strip().split("\t")
            if len(tokens) > 3:
                _, sympnorm, ori, _, _ = tokens
                self.symptoms_cui.add(sympnorm)
        
        # long-tail entity set
        longtailfile = "data/VAERS/symptoms_longtail.tsv"
        self.symptoms_longtail = set()
        for i, w in enumerate(open(longtailfile, "r")):
            if i == 0:
                continue
            _, sym, ori, _ = w.split("\t")
            self.symptoms_longtail.add(sym)

    def _compute(self, predictions, references):
        """Returns the scores"""
        test_type = ["full", "cui", "longtail"]
        target_symp_set = {
            "full": self.symptoms,
            "cui": self.symptoms_cui,
            "longtail": self.symptoms_longtail
        }
        
        full, cui, longtail = PrecRec("FULL"), PrecRec("CUI"), PrecRec("LONGTAIL")
        test_class = [full, cui, longtail]

        for pred, ref in zip(predictions, references):
            for tt, cc in zip(test_type, test_class):
                # the number of gold entities
                gold_entities = ref["symptoms"]
                local_gold_ent = []
                for ge in gold_entities:
                    if ge in target_symp_set[tt]:
                        local_gold_ent.append(ge)
                
                if len(local_gold_ent) > 0:
                    local_n_true = len(local_gold_ent)  # the number of gold entities per each example
                    cc.global_n_true += local_n_true
                    cc.n_target_doc += 1

                    # get entities from generated texts
                    model_outputs = [x.strip().lower().replace(" ", "") for x in pred.strip().split(",")]
                    pred_symps = []
                    for mo in model_outputs:
                        if mo in target_symp_set[tt]:
                            pred_symps.append(mo)

                    local_n_pos = len(pred_symps)
                    cc.global_n_pos += local_n_pos

                    # get true positives
                    local_tp = 0
                    tp_symps = set(local_gold_ent) & set(pred_symps) # true positive
                    n_tp_symps = len(tp_symps)
                    if n_tp_symps > 0:
                        local_tp += n_tp_symps
                        cc.global_tp += n_tp_symps

                    # compute local precision and recall for macro metrics
                    cur_p = (local_tp / local_n_pos) if local_n_pos != 0 else 0
                    cur_r = (local_tp / local_n_true) if local_n_true != 0 else 0
                    cc.p_sum += cur_p
                    cc.r_sum += cur_r

        assert len(predictions) == full.n_target_doc

        results = {}
        for tt, cc in zip(test_type, [full, cui, longtail]):
            results[tt] = cc.get_scores()

        return results

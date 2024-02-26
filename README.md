# SYMPTOMIFY
Dataset and implementation of the paper [SYMPTOMIFY: Transforming Symptom Annotations with Language Model Knowledge Harvesting](https://aclanthology.org/2023.findings-emnlp.781/) (EMNLP Findings 2023). SYMPTOMIFY is a large-scale dataset of over 800k annotated reports, reflecting reactions to medications and vaccines. It includes MedDRA symptoms, annotation explanations, and background knowledge about symptoms, designed to facilitate development of systems that can aid human annotators operating in the critically important public health domain.

SYMPTOMIFY is built based on the Vaccine Adverse Event Reporting System (VAERS) database managed by the U.S. Centers for Disease Control and Prevention (CDC) the U.S. Food and Drug Administration (FDA). The dataset spans three years (2019-2021) and includes 839,215 reports, incorporating symptom texts, relevant MedDRA terms, and associated metadata like age, sex, and vaccine type. The original VAERS data can be found [here](https://vaers.hhs.gov/data.html).


## Download Data and Data Format
Download: https://drive.google.com/drive/folders/1w4zNKoYRWAnQGZv3hVAAyhttKk1RLKvQ?usp=share_link

Each line of .json files is one report of VAERS.
- `vid`: VAERS database entry id; you can map each example to VAERS's database using this id
- `symptom_text`: Symptom text (VAERSreports) 
- `symptom_ids`: Symptom ids for classification models
- `symptoms`:["injectionsiteerythema", "injectionsitepain", "injectionsiteswelling", "tenderness"]
- `age`: 6 {0: 0-9, 1: 10-19, ..., 10} Age can be 
- `sex`: 1 {0: Male, 1: Female, 2: Unknonw}
- `vax_type`: Vaccine type(s) that the patient received in the report
- `vax_type_ids`: Vaccine type id(s) for classification models
- `vax_name`: Vaccine name(s) that the patient received in the report
- `vax_name_ids`: Vaccine name id(s) for classification models
- `symptoms_original`: Symptoms names from MedDRA; same as 'symptoms' but with original names
- `n_symps`: 4 the number of symptoms annotated in this example

## Setup
To install requirements
```
pip install -r requirements.txt
```

## Pretrain-Finetune Baselines

1. Classification Approach

Train the ClinicalBERT+CNN model:

```
python run_classifier.py \
    --model_name_or_path emilyalsentzer/Bio_ClinicalBERT \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --learning_rate 3e-5 \
    --num_train_epochs 5 \
    --output_dir {out_dir} \
    --train_file {data_dir}/train.json \
    --validation_file {data_dir}/dev_1k.json \
    --test_file {data_dir}/test.json \
    --symptoms_file data/symptoms.tsv \
    --do_train
```

2. Generative Approach

Train the generative entity retrieval model with BART (Lewis et al., 2020)

```
python run_generative_model.py \
    --model_name_or_path facebook/bart-base \
    --max_source_length 256 \
    --max_target_length 128 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --output_dir {output_dir} \
    --text_column symptom_text \
    --summary_column symptoms \
    --train_file data/train.json \
    --validation_file data/dev.json \
    --do_train
```

Test the model:
```
python run_generative_model.py \
    --model_name_or_path {test_model_name_or_path} \
    --max_source_length 256 \
    --max_target_length 128 \
    --per_device_eval_batch_size 16 \
    --text_column symptom_text \
    --summary_column symptoms \
    --test_file data/test.json \
    --do_predict
```

for the multi-GPU setting,
```
python -m torch.distributed.launch \
    --nproc_per_node=2 run_generative_model.py \
    --model_name_or_path facebook/bart-base \
    --max_source_length 256 \
    --max_target_length 128 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --output_dir {output_dir} \
    --text_column symptom_text \
    --summary_column symptoms \
    --train_file data/train.json \
    --validation_file data/dev.json \
    --do_train
```



## Don't forget to run tqdm before starting the script

import argparse
import csv
from flax.training.common_utils import shard
from flax.jax_utils import replicate, unreplicate
import gc
import itertools
import jax
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from transformers import FlaxMarianMTModel, MarianTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--tsv_path", type=str, default=None, help="path of directory where the dataset is stored")
parser.add_argument("--lang_list", nargs="+", default=["fr", "de", "es"], help="Language list (apart from English)")
parser.add_argument("--save_path", type=str, default=None, help="path of directory where the translated dataset will be stored")
args = parser.parse_args()


DATASET_PATH = args.tsv_path
LANG_LIST = args.lang_list
SAVE_PATH = args.save_path
BATCH_SIZE = 512
num_devices = 8
lang_dict = {
    "es" : "es_XX",
    "de": "de_DE",
    "fr": "fr_XX",
    # "ru": "ru_RU"  # removed Russian after Patrick's suggestions
}

model_de = FlaxMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de", from_pt=True)
model_es = FlaxMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es", from_pt=True)
model_fr = FlaxMarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr", from_pt=True)

de_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de", source_lang="en")
es_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es", source_lang="en")
fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr", source_lang="en")

def generatefr_XX(params, batch):
      output_ids = model_fr.generate(batch["input_ids"], attention_mask=batch["attention_mask"], params=params, num_beams=4, max_length=64, early_stopping=True).sequences
      return output_ids

def generatees_XX(params, batch):
      output_ids = model_es.generate(batch["input_ids"], attention_mask=batch["attention_mask"], params=params, num_beams=4, max_length=64, early_stopping=True).sequences
      return output_ids

def generatede_DE(params, batch):
      output_ids = model_de.generate(batch["input_ids"], attention_mask=batch["attention_mask"], params=params, num_beams=4, max_length=64, early_stopping=True).sequences
      return output_ids

# def generateru_RU(params, batch, rng):
#       output_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], prng_key=rng, params=params, forced_bos_token_id=tokenizer.lang_code_to_id["ru_RU"]).sequences
#       return output_ids

p_generate_fr_XX = jax.pmap(generatefr_XX, "batch")
p_generate_es_XX = jax.pmap(generatees_XX, "batch")
p_generate_de_DE = jax.pmap(generatede_DE, "batch")
# p_generate_ru_RU = jax.pmap(generateru_RU, "batch")

map_name = {
    "fr_XX": p_generate_fr_XX,
    "es_XX": p_generate_es_XX,
    "de_DE": p_generate_de_DE,
    # "ru_RU": p_generate_ru_RU,
}


tokenizer_map = {
    "fr": fr_tokenizer,
    "es": es_tokenizer,
    "de": de_tokenizer,
    # "ru_RU": p_generate_ru_RU,
}

map_model_params = {
    "fr": replicate(model_fr.params),
    "es": replicate(model_es.params),
    "de": replicate(model_de.params),
    # "ru_RU": p_generate_ru_RU,
}

def run_generate(input_str, p_generate, p_params, tokenizer):
    inputs = tokenizer(input_str, return_tensors="jax", padding="max_length", truncation=True, max_length=64)
    p_inputs = shard(inputs.data)
    output_ids = p_generate(p_params, p_inputs)
    output_strings = tokenizer.batch_decode(output_ids.reshape(-1,64), skip_special_tokens=True, max_length=64)
    return output_strings

def read_tsv_file(tsv_path):
    df = pd.read_csv(tsv_path, delimiter="\t", index_col=False)
    print("Number of Examples:", df.shape[0], "for", tsv_path)
    return df

def arrange_data(image_files, questions, answer_labels, question_types):  # iterates through all the captions and save there translations
    lis_ = []
    # if lang_code=="en":
    for image_file, question, answer_label, question_type in zip(image_files, questions, answer_labels, question_types):  # add english caption first
        lis_.append({"image_file":image_file, "question":question, "answer_label":answer_label, "question_type":question_type, "lang_id": "en"})

    for lang_code in LANG_LIST:
        p_params = map_model_params[lang_code]
        p_generate = map_name[lang_dict[lang_code]]
        tokenizer = tokenizer_map[lang_code]
        outputs = run_generate(questions, p_generate, p_params, tokenizer)

        for image_file, output, answer_label, question_type in zip(tqdm(image_files, total=len(image_files), position=0, leave=False, desc=f"Processing for {lang_code} currently"), outputs, answer_labels, question_types):  # add other captions
            lis_.append({"image_file":image_file, "question":output, "answer_label":answer_label,"question_type":question_type, "lang_id": lang_code})

    gc.collect()
    return lis_


df = read_tsv_file(DATASET_PATH)
# train_df, val_df = train_test_split(_df, test_size=VAL_SPLIT, random_state=1234)

# train_df.reset_index(drop=True, inplace=True)
# val_df.reset_index(drop=True, inplace=True)

# print("\n train/val dataset created. beginning translation")

# if IS_TRAIN:
#     df = train_df
#     output_file_name = os.path.join(SAVE_VAL, "train_file.tsv")
#     with open(output_file_name, 'w', newline='') as outtsv:  # creates a blank tsv with headers (overwrites existing file)
#         writer = csv.writer(outtsv, delimiter='\t')
#         writer.writerow(["image_file", "caption", "url", "lang_id"])

# else:
#     df = val_df
output_file_name = os.path.join(SAVE_PATH)
with open(output_file_name, 'w', newline='') as outtsv:  # creates a blank tsv with headers (overwrites existing file)
    writer = csv.writer(outtsv, delimiter='\t')
    writer.writerow(["image_file", "question", "answer_label", "question_type","lang_id"])

# roulette = 0
for i in tqdm(range(0,len(df),BATCH_SIZE)):
    # lang_code = LANG_LIST[roulette % len(LANG_LIST)]
    output_batch = arrange_data(list(df["image_file"])[i:i+BATCH_SIZE], list(df["question"])[i:i+BATCH_SIZE], list(df["answer_label"])[i:i+BATCH_SIZE], list(df["question_type"])[i:i+BATCH_SIZE])
    # roulette += 1
    with open(output_file_name, "a", newline='') as f:
      writer = csv.DictWriter(f, fieldnames=["image_file", "question", "answer_label", "question_type","lang_id"], delimiter='\t')
      for batch in output_batch:
          writer.writerow(batch)

import json
import os
from tqdm import tqdm

with open("../VQAV2/vocabulary_vqa.txt") as f:
    answers = f.readlines()

answers = [answer.strip() for answer in answers]
answer_dict = {k: v for v, k in enumerate(answers)}

with open("answer_mapping.json", "w") as f:
    json.dump(answer_dict, f)

answer_reverse_dict = {k: v for k, v in enumerate(answers)}

with open("answer_reverse_mapping.json", "w") as f:
    json.dump(answer_reverse_dict, f)

with open("../VQAV2/v2_mscoco_val2014_annotations.json") as f:
    val_annot = json.load(f)

with open("../VQAV2/v2_OpenEnded_mscoco_val2014_questions.json") as f:
    val_quest = json.load(f)


file_name = "val_file.tsv"
writer = open(file_name, "w")
writer.write("image_file\tquestion\tanswer_label\tquestion_type\n")
for idx, annotation in tqdm(enumerate(val_annot["annotations"])):
    qimage_id = val_quest["questions"][idx]["image_id"]
    qquest_id = val_quest["questions"][idx]["question_id"]
    vimage_id = annotation["image_id"]
    vquest_id = annotation["question_id"]

    assert qimage_id == vimage_id
    assert qquest_id == vquest_id

    image_file = f"val2014/COCO_val2014_{vimage_id:012d}.jpg"
    if os.path.exists(os.path.join("../VQAV2", image_file)):
        question = val_quest["questions"][idx]["question"]
        answer_label = answer_dict.get(
            annotation["multiple_choice_answer"], answer_dict.get("<unk>")
        )
        question_type = annotation["question_type"]
        writer.write(
            image_file.strip()
            + "\t"
            + question.strip()
            + "\t"
            + str(answer_label)
            + "\t"
            + question_type.strip()
            + "\n"
        )
    else:
        print(f"Skipping {image_file}")
writer.close()

with open("../VQAV2/v2_mscoco_train2014_annotations.json") as f:
    train_annot = json.load(f)

with open("../VQAV2/v2_OpenEnded_mscoco_train2014_questions.json") as f:
    train_quest = json.load(f)


file_name = "train_file.tsv"
writer = open(file_name, "w")
writer.write("image_file\tquestion\tanswer_label\tquestion_type\n")
for idx, annotation in tqdm(enumerate(train_annot["annotations"])):
    qimage_id = train_quest["questions"][idx]["image_id"]
    qquest_id = train_quest["questions"][idx]["question_id"]
    vimage_id = annotation["image_id"]
    vquest_id = annotation["question_id"]

    assert qimage_id == vimage_id
    assert qquest_id == vquest_id

    image_file = f"train2014/COCO_train2014_{vimage_id:012d}.jpg"
    if os.path.exists(os.path.join("../VQAV2", image_file)):
        question = train_quest["questions"][idx]["question"]
        answer_label = answer_dict.get(
            annotation["multiple_choice_answer"], answer_dict.get("<unk>")
        )
        question_type = annotation["question_type"]
        writer.write(
            image_file.strip()
            + "\t"
            + question.strip()
            + "\t"
            + str(answer_label)
            + "\t"
            + question_type.strip()
            + "\n"
        )
    else:
        print(f"Skipping {image_file}")

writer.close()

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import TrainingArguments, Trainer

from dataset.AlternateDataset import KTDataset
from dataset.QuestionTags import QuestionTags
from models.AAKT import AAKT


def collate_fn(batch_of_data):
    return {
        "input_ids": torch.cat([data[0] for data in batch_of_data], dim=0).long(),
        "input_times": torch.cat([data[1] for data in batch_of_data], dim=0).float(),
        "labels_kts": torch.cat([data[2] for data in batch_of_data], dim=0).long(),
        "labels_tags": torch.cat([data[3] for data in batch_of_data], dim=0).float(),
    }


def preprocess_logits_for_metrics(predictions, labels):
    loss_kts, loss_tags, preds_kts = predictions
    probs_kts = 1 / (torch.exp(preds_kts[:, :, 0] - preds_kts[:, :, 1]) + 1)  # most are [bs, seq_len]
    return loss_kts, loss_tags, probs_kts


def compute_metrics(eval_preds):
    loss_kts, loss_tags, probs_kts = eval_preds.predictions
    labels_kts = eval_preds.label_ids

    loss_kts = loss_kts.mean()
    loss_tags = loss_tags.mean()

    labels_kts = labels_kts.reshape(-1)
    probs_kts = probs_kts.reshape(-1)

    non_pad_indices = np.nonzero(labels_kts != -100)
    labels_kts = labels_kts[non_pad_indices]
    probs_kts = probs_kts[non_pad_indices]

    assert labels_kts.shape[0] == probs_kts.shape[0], f'{labels_kts.shape}, {probs_kts.shape}'

    acc = ((probs_kts > 0.5) == labels_kts).mean()
    auc = roc_auc_score(labels_kts, probs_kts)
    rmse = np.sqrt(((labels_kts - probs_kts) ** 2 / labels_kts.shape[0]).sum())

    result = {"loss_kts": loss_kts,
              "loss_tags": loss_tags,
              "ACC": acc,
              "AUC": auc,
              "RMSE": rmse,
              "num_kts": labels_kts.size}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse args for modeling and training')

    parser.add_argument('--without_tags', type=bool, default=False)
    parser.add_argument('--without_times', type=bool, default=False)
    parser.add_argument('--eval', type=int)

    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=144)
    parser.add_argument('--n_head', type=int, default=12)

    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--n_devices', type=int, default=1)
    parser.add_argument('--global_batch_size', type=int, default=256)
    parser.add_argument('--mini_batch_size', type=int, default=64)

    parser.add_argument('--file_prefix', type=str, required=True)
    parser.add_argument('--tag_prefix', type=str, required=True)
    parser.add_argument('--record_prefix', type=str, required=True)

    args = parser.parse_args()
    print(f"args: {json.dumps(vars(args), ensure_ascii=False, indent=4)}")

    eval = args.eval == 1

    file_prefix = args.file_prefix
    tag_prefix = args.tag_prefix
    record_prefix = args.record_prefix

    max_seq_len = args.max_seq_len
    n_layer = args.n_layer
    n_embd = args.n_embd
    n_head = args.n_head
    rotary_dim = n_embd // n_head // 2

    output_dir = f"output-{record_prefix}"
    print(f"output_dir: {output_dir}")

    question_tags = QuestionTags(tag_prefix=tag_prefix)
    usage2dataset = {
        usage: KTDataset(question_tags, max_seq_len, usage=usage, file_prefix=file_prefix)
        for usage in ["train", "valid", "test"]
    }
    usage2nums = {"names": ['cases', 'kts', 'kts_labels']}
    print(f"num of cases / kts / kts_labels:")
    for usage, dataset in usage2dataset.items():
        total_num_kt_labels = int(sum([
            (labels_kts != -100).float().sum().item()
            for input_ids, input_times, labels_kts, labels_tags in tqdm(dataset, desc=usage)
        ]))
        print(f"{usage}: {len(dataset)} / {dataset.total_num_kts} / {total_num_kt_labels}")
        usage2nums[usage] = [len(dataset), dataset.total_num_kts, total_num_kt_labels]
        assert usage == "train" or total_num_kt_labels == dataset.total_num_kts

    learning_rate = args.learning_rate
    n_devices = args.n_devices
    global_batch_size = args.global_batch_size
    mini_batch_size = args.mini_batch_size
    gradient_accumulation_steps = global_batch_size // (n_devices * mini_batch_size)

    epochs = args.epochs
    total_training_steps = np.math.ceil(len(usage2dataset["train"]) / global_batch_size) * epochs
    logging_steps = int(total_training_steps / epochs / 10)

    print(f"training epochs = {epochs:.2f}")

    model = AAKT(
        num_questions=question_tags.num_questions,
        num_tags=question_tags.num_tags,
        max_seq_len=max_seq_len,
        with_tags=not args.without_tags,
        with_times=not args.without_times,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        rotary_dim=rotary_dim
    )
    print(f"model: {model}")

    training_args = TrainingArguments(
        output_dir=output_dir,

        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,

        learning_rate=learning_rate,
        lr_scheduler_type='cosine',
        per_device_train_batch_size=mini_batch_size,
        per_device_eval_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=1,
        fp16=True,
        warmup_ratio=0.1,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="AUC",
        greater_is_better=True,
        log_level='info',
        logging_steps=logging_steps,
        dataloader_num_workers=14,
        label_names=["labels_kts"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=usage2dataset["train"],
        eval_dataset=usage2dataset["valid"] if eval else usage2dataset['valid'] + usage2dataset['test'],
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    train_output = trainer.train()

    test_result = trainer.evaluate(usage2dataset["test"] if eval else usage2dataset["test"] + usage2dataset["valid"],
                                   metric_key_prefix="test")
    print(test_result)

    best_model_path = os.path.join(trainer.state.best_model_checkpoint, "pytorch_model.bin")
    state_dict = torch.load(best_model_path)
    emb_weight = state_dict['model.wte.weight'].detach().cpu().numpy()
    np.save(f'emb_weight/{record_prefix}.npy', emb_weight)

    results = {}
    results.update(train_output._asdict())
    results.update(test_result)
    record = {
        output_dir: {
            "args": vars(args),
            "usage2nums": usage2nums,
            "results": results,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
        }
    }

    record_file = f'records/{record_prefix}.json'
    if os.path.exists(record_file):
        with open(record_file, "r") as f:
            records = json.loads(f.read())
        records.update(record)
    else:
        records = record
    with open(record_file, "w") as f:
        f.write(json.dumps(records, ensure_ascii=False, indent=4) + "\n")

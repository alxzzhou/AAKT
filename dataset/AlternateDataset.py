import json
import multiprocessing
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class KTDataset(Dataset):
    def __init__(self, question_tags, max_seq_len, usage="train", path=None, preprocess=True, file_prefix=None):
        self.question_tags = question_tags
        self.max_seq_len = max_seq_len
        self.usage = usage
        self.pad_token_id = question_tags.num_questions + 3
        if not usage in ["train", "valid", "test"]: raise NotImplementedError
        if path is None: path = f"autodl-tmp/{file_prefix}.{usage}.json"
        cache_path = path + (".pt" if not preprocess else ".pp.pt")
        if os.path.exists(cache_path):
            if preprocess:
                self.data = torch.load(cache_path)
            else:
                self.kts = torch.load(cache_path)
        else:
            self.kts = []
            with open(path, "r") as f:
                for line in f:
                    try:
                        kt = json.loads(line)
                        self.kts.append(kt)
                    except:
                        pass
            kt_lengths = np.array([len(kt) for kt in self.kts])
            for ratio in [80, 90, 95]:
                print(f"{usage}: {ratio}% of kt lengths < {np.percentile(kt_lengths, ratio)}")
            all_questions = sorted(list(set([k[0] for kt in self.kts for k in kt])))
            print(f"{usage}: {len(all_questions)} questions: from {all_questions[0]} to {all_questions[-1]}")
            all_spendtime = np.array(list([float(k[2]) for kt in self.kts for k in kt]))
            for ratio in [80, 90, 95, 98]:
                print(f"{usage}: {ratio}% of spend times < {np.percentile(all_spendtime, ratio)}")
            if preprocess:
                self.data = self.preprocess(processes=None)
                torch.save(self.data, cache_path)
            else:
                torch.save(self.kts, cache_path)
        self.preprocessed = preprocess
        self.expand()
        return

    def preprocess(self, processes=None):
        if processes is not None:
            torch.multiprocessing.set_sharing_strategy('file_system')
            with multiprocessing.Pool(processes=processes) as pool:
                data = list(tqdm(pool.imap(process, [(kt, self.question_tags) for kt in self.kts]),
                                 total=len(self.kts), desc="preprocessing data ..."))
        else:
            data = [process((kt, self.question_tags))
                    for kt in tqdm(self.kts, desc="preprocessing data ...")]
        return data

    def expand(self):
        pred_ratio = .5 if self.usage == "train" else .8  # r_overlap = pred_ratio
        self.labels_seq_len = max(int(pred_ratio * self.max_seq_len), 2)
        self.labels_seq_len = self.labels_seq_len - self.labels_seq_len % 2
        self.prefix_seq_len = self.max_seq_len - self.labels_seq_len
        if self.preprocessed:
            self.item_lengths = [input_ids.shape[1] for input_ids, input_times, labels, labels_tags in self.data]
        else:
            self.item_lengths = [len(kt) * 2 for kt in self.kts]
        self.total_num_kts = sum(self.item_lengths) // 2
        self.splits = []
        for item, item_length in enumerate(self.item_lengths):
            if item_length < self.max_seq_len:
                start, end, start_label = 0, item_length, 0
                self.splits.append((item, start, end, start_label))
            else:
                for offset in range(0, item_length - self.prefix_seq_len, self.labels_seq_len):
                    if offset + self.max_seq_len > item_length:
                        start, end = item_length - self.max_seq_len, item_length
                    else:
                        start, end = offset, offset + self.max_seq_len
                    if offset == 0:
                        start_label = start
                    elif self.usage == "train":
                        start_label = start
                    else:
                        start_label = offset + self.prefix_seq_len
                    self.splits.append((item, start, end, start_label))
        self.labels_kts_prefix = torch.empty([1, self.max_seq_len], dtype=torch.int16).fill_(-100)
        self.labels_tags_prefix = self.labels_kts_prefix.unsqueeze(-1).repeat(1, 1, self.question_tags.num_tags)
        self.input_ids_padding = torch.empty([1, self.max_seq_len], dtype=torch.int16).fill_(self.pad_token_id)
        self.input_times_padding = torch.empty([1, self.max_seq_len], dtype=torch.float16).fill_(-1)
        self.labels_padding = torch.empty([1, self.max_seq_len], dtype=torch.int16).fill_(-100)
        self.labels_tags_padding = self.labels_padding.unsqueeze(-1).repeat(1, 1, self.question_tags.num_tags)
        return

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, item):
        split = self.splits[item]
        item, start, end, start_label = split

        if self.preprocessed:
            input_ids_, input_times_, labels_kts_, labels_tags_, = self.data[item]
        else:
            input_ids_, input_times_, labels_kts_, labels_tags_, = process([self.kts[item], self.question_tags])

        input_ids = input_ids_[:, start:end]
        input_times = input_times_[:, start:end]
        labels_kts = labels_kts_[:, start_label:end]
        labels_tags = labels_tags_[:, start_label:end, :]
        if start_label > start:
            labels_prefix_len = start_label - start
            labels_kts = torch.cat([
                self.labels_kts_prefix[:, :labels_prefix_len],
                labels_kts
            ], dim=1)
            labels_tags = torch.cat([
                self.labels_tags_prefix[:, :labels_prefix_len, :],
                labels_tags
            ], dim=1)

        assert input_ids.shape[1] == input_times.shape[1] == labels_kts.shape[1] == labels_tags.shape[1]
        if input_ids.shape[1] < self.max_seq_len:
            pad_len = self.max_seq_len - input_ids.shape[1]
            input_ids = torch.cat([input_ids, self.input_ids_padding[:, :pad_len]], dim=1)
            input_times = torch.cat([input_times, self.input_times_padding[:, :pad_len]], dim=1)
            labels_kts = torch.cat([labels_kts, self.labels_padding[:, :pad_len]], dim=1)
            labels_tags = torch.cat([labels_tags, self.labels_tags_padding[:, :pad_len, :]], dim=1)

        assert input_ids.shape[1] == input_times.shape[1] == labels_kts.shape[1] \
               == labels_tags.shape[1] == self.max_seq_len

        return input_ids, input_times, labels_kts, labels_tags


def process(inputs, times_factor=60000):
    kt, question_tags = inputs
    num_questions = question_tags.num_questions
    num_tags = question_tags.num_tags
    pad_token_id = num_questions + 3
    input_ids = torch.empty([1, len(kt) * 2], dtype=torch.int16).fill_(pad_token_id)
    input_times = torch.empty([1, len(kt) * 2], dtype=torch.float16).fill_(-1)
    labels_kts = torch.empty([1, len(kt) * 2], dtype=torch.int16).fill_(-100)
    labels_tags = torch.empty([1, len(kt) * 2, num_tags], dtype=torch.int16).fill_(-100)
    i = 0
    for j, (question, result, time) in enumerate(kt):
        assert question < num_questions
        input_ids[i, j * 2] = question
        input_ids[i, j * 2 + 1] = result + num_questions
        input_times[i, j * 2 + 1] = float(time) / times_factor
        labels_kts[i, j * 2] = result
        labels_tags[i, j * 2, :] = 0
        for tag in question_tags[question]:
            labels_tags[i, j * 2, tag] = 1
    return [input_ids, input_times, labels_kts, labels_tags]

import json


class QuestionTags:
    def __init__(self, tag_prefix):
        path = f'autodl-tmp/{tag_prefix}.json'
        with open(path, "r") as f:
            self.tags = json.loads(f.read())
        self.num_questions = len(self.tags)
        self.all_tags = sorted(list(set([tag for tags in self.tags for tag in tags])))
        self.num_tags = len(self.all_tags)
        self.tag_indexes = {tag: i for i, tag in enumerate(self.all_tags)}
        print(f"tags: num_questions = {self.num_questions}, num_tags = {self.num_tags}")

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, question):
        tags = self.tags[question]
        return [self.tag_indexes[tag] for tag in tags]

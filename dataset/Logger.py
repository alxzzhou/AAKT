import datetime
import json
import os


class Logger:
    def __init__(self, model, record):
        self.model = model
        self.record = {
            str(datetime.date.today()): record
        }

    def write(self):
        record_file = 'records/other/' + self.model + '.json'
        if os.path.exists(record_file):
            with open(record_file, "r") as f:
                records = json.loads(f.read())
            records.update(self.record)
        else:
            records = self.record
        with open(record_file, "w+") as f:
            f.write(json.dumps(records, ensure_ascii=False, indent=4) + "\n")

import json
import tempfile
import unittest
from pathlib import Path

from plume.data import dataset_directory, load_paired_documents


class DataTests(unittest.TestCase):
    def test_dataset_groups(self):
        self.assertEqual(dataset_directory("data", "squad"), Path("data/benchmarks/squad"))
        self.assertEqual(dataset_directory("data", "gsm8k"), Path("data/generalization/gsm8k"))
        self.assertEqual(dataset_directory("data", "crux"), Path("data/generalization/crux"))

    def test_load_compact_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            old_path, full_path = Path(directory) / "old.jsonl", Path(directory) / "full.jsonl"
            old = {"context": "old", "prompts": ["q"], "responses": ["old answer"]}
            full = {"context": "old update", "prompts": ["q"], "responses": ["new answer"]}
            old_path.write_text(json.dumps(old) + "\n")
            full_path.write_text(json.dumps(full) + "\n")
            document = load_paired_documents(old_path, full_path)[0]
            self.assertEqual(document.responses, ("new answer",))
            self.assertEqual(document.old_responses, ("old answer",))

    def test_reject_prompt_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            old_path, full_path = Path(directory) / "old.jsonl", Path(directory) / "full.jsonl"
            old_path.write_text(json.dumps({"context": "a", "prompts": ["x"]}) + "\n")
            full_path.write_text(json.dumps({"context": "b", "prompts": ["y"], "responses": ["z"]}) + "\n")
            with self.assertRaises(ValueError):
                load_paired_documents(old_path, full_path)


if __name__ == "__main__":
    unittest.main()

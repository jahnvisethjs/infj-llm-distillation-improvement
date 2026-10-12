import unittest
import json
from pathlib import Path
import tempfile

from capstone_person1 import person1_data_contract as contract
from capstone_person1.tokenizer_adapter import SnapshotTokenizer


class DataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = SnapshotTokenizer.from_pretrained(contract.TOKENIZER_ID, revision=contract.PINNED_TOKENIZER_REVISION)

    def example(self, **kwargs):
        return contract.encode_example(self.tokenizer, 'What is 2 + 3?',
                                       'Adding the two quantities gives the result.\nFinal answer: 5', '5', **kwargs)

    def test_prompt_bytes_and_evaluation(self):
        self.assertEqual(contract.render_prompt('2 + 3?'),
                         'Question: 2 + 3?\nSolve the problem step by step. End with: Final answer: <answer>.\nSolution:\n')
        self.assertNotIn('<|', contract.render_prompt('2 + 3?'))
        with self.assertRaises(ValueError): contract.render_prompt(float('nan'))

    def test_combined_ids_and_loss_mask(self):
        row = self.example()
        combined = self.tokenizer(row['prompt_text'] + row['completion'], add_special_tokens=False)['input_ids']
        self.assertEqual(row['input_ids'][:-1], combined)
        self.assertEqual(row['labels'][:row['completion_start']], [-100] * row['prompt_tokens'])
        self.assertEqual(row['labels'][row['completion_start']:], row['input_ids'][row['completion_start']:])
        self.assertEqual(row['attention_mask'], [1] * row['total_tokens'])
        self.assertEqual(row['supervised_tokens'], sum(label != -100 for label in row['labels']))
        self.assertEqual(row['supervised_tokens'], row['completion_tokens'] + 1)
        self.assertEqual(row['input_ids'].count(151643), 1)
        self.assertEqual(row['labels'][-1], 151643)

    def test_padding_preserves_real_eos(self):
        row = self.example()
        padded = contract.pad_encoded(row, row['total_tokens'] + 3)
        self.assertEqual(padded['input_ids'][-4:], [151643] * 4)
        self.assertEqual(padded['labels'][-4:], [151643, -100, -100, -100])
        self.assertEqual(padded['attention_mask'][-4:], [1, 0, 0, 0])
        self.assertEqual(row['attention_mask'], [1] * row['total_tokens'])
        with self.assertRaises(contract.ContextLengthError): contract.pad_encoded(row, row['total_tokens'] - 1)

    def test_preserves_equivalent_terminal_block(self):
        solution = 'Working through the calculation gives nine halves.\n\nFinal answer: 4.5\n'
        self.assertEqual(contract.render_completion(solution, '9/2'), solution)

    def test_missing_or_contradictory_answer_is_rejected(self):
        for solution, answer in [('Final answer: 3', '4'), ('Final answer: 3\nCorrection: 4', '3'),
                                 ('We consider possibilities carefully.', '4'),
                                 ('Final answer: 3 or 4', '3'), ('We calculated 4.', '4')]:
            with self.subTest(solution=solution):
                with self.assertRaises(ValueError): contract.render_completion(solution, answer)

    def test_encoder_rejects_single_target_for_multi_part_problem(self):
        with self.assertRaises(ValueError):
            contract.encode_example(self.tokenizer, 'Find the length and width of the rectangle.',
                'Calculating the first dimension gives the required length.\nFinal answer: 5', '5')

    def test_context_length_never_truncates(self):
        with self.assertRaises(contract.ContextLengthError): self.example(max_total_tokens=3)

    def test_eos_text_injection_is_rejected(self):
        with self.assertRaises(ValueError):
            contract.encode_text_pair(self.tokenizer, contract.render_prompt('2?'), '4<|endoftext|>')

    def test_tokenizer_identity_is_verified(self):
        info = contract.verify_tokenizer(self.tokenizer)
        self.assertEqual(info['eos_token_id'], 151643)
        self.assertEqual(info['pad_token_id'], 151643)
        class Wrong:
            is_fast = False
        with self.assertRaises(ValueError): contract.verify_tokenizer(Wrong())

    def test_crossing_token_rejected_despite_matching_prompt_prefix(self):
        base = self.tokenizer
        prompt = contract.render_prompt('2?')
        class Crossing:
            def __getattr__(self, key): return getattr(base, key)
            def __call__(self, text, **kwargs):
                if kwargs.get('return_offsets_mapping'):
                    return {'input_ids': [1, 2], 'offset_mapping': [(0, len(prompt) - 1), (len(prompt) - 1, len(text))]}
                return {'input_ids': [1]}
        with self.assertRaises(contract.BoundaryError):
            contract.encode_text_pair(Crossing(), prompt, '4')

    def test_manifest_reencoding_detects_changed_counts(self):
        encoded = self.example()
        record = dict(encoded, row_id='example', problem_group_id='group', problem='What is 2 + 3?',
                      solution='Adding the two quantities gives the result.\nFinal answer: 5', final_answer='5')
        contract.validate_manifest_record(record, self.tokenizer)
        record['supervised_tokens'] += 1
        with self.assertRaises(ValueError): contract.validate_manifest_record(record, self.tokenizer)

    def test_manifest_file_requires_matching_hash_and_budget(self):
        record = dict(self.example(), row_id='example', problem_group_id='group', problem='What is 2 + 3?',
                      solution='Adding the two quantities gives the result.\nFinal answer: 5', final_answer='5')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.jsonl'
            path.write_text(json.dumps(record) + '\n', encoding='utf-8')
            metadata = dict(tokenizer_id=contract.TOKENIZER_ID, tokenizer_revision=contract.PINNED_TOKENIZER_REVISION,
                            manifest_sha256=contract.hashlib.sha256(path.read_bytes()).hexdigest(),
                            rows=1, supervised_tokens=record['supervised_tokens'])
            self.assertEqual(contract.validate_manifest(path, metadata, self.tokenizer)['rows'], 1)
            metadata['supervised_tokens'] += 1
            with self.assertRaises(ValueError): contract.validate_manifest(path, metadata, self.tokenizer)


if __name__ == '__main__': unittest.main()

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import LlamaConfig, LlamaModel
from transformers.models.llama import modeling_llama

from scripts.extract_full_attention_scores import (
    DATASET, array_bytes, capture_one, completed, exclusive_json, input_tasks,
)
from utils.full_attention_scores import extract_to_npy, full_fp32_scores, prefix_validation


def tiny_model():
    torch.manual_seed(123)
    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                         max_position_embeddings=64, attention_dropout=0.0)
    config._attn_implementation = 'eager'
    return LlamaModel(config).half().eval()


class FullScoreTests(unittest.TestCase):
    def test_fp32_matmul_is_not_fp16_score_promoted_afterwards(self):
        torch.manual_seed(1)
        q = torch.randn(1,4,7,8).half()*4
        k = torch.randn(1,4,7,8).half()*4
        result = full_fp32_scores(q,k)
        reference = (q.float() @ k.float().transpose(-1,-2))[0]/(8**0.5)
        torch.testing.assert_close(result,reference,rtol=0,atol=0)
        rounded = (q @ k.transpose(-1,-2))[0].div(8**0.5).float()
        self.assertTrue(bool((result != rounded).any()))
        with self.assertRaises(ValueError):
            full_fp32_scores(q.float(),k.float())
        q[0,0,0,0] = float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            full_fp32_scores(q,k)

    def test_native_post_rope_reference_actual_positions_and_full_future_scores(self):
        model = tiny_model()
        ids = torch.tensor([[1,5,9,13]],dtype=torch.long)
        actual_positions = torch.tensor([[7,9,12,16]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'scores.npy'
            info = extract_to_npy(model,ids,path,validate_native=True,
                                 forward_kwargs={'position_ids':actual_positions})
            values = np.load(path,allow_pickle=False)
            self.assertEqual(values.shape,(2,4,4,4))
            self.assertEqual(values.dtype,np.float32)
            self.assertTrue(np.isfinite(values).all())
            self.assertTrue(np.any(values[:,:,0,1:] != 0))
            self.assertEqual(info['position_ids'],actual_positions.tolist())
            self.assertEqual(len(info['native_qk_reference_checks']),2)
            self.assertFalse(info['causal_mask_applied_to_saved_scores'])
            self.assertEqual(path.stat().st_size,array_bytes(values.shape))
            with self.assertRaises(FileExistsError):
                extract_to_npy(model,ids,path)
        self.assertTrue(all(not layer.self_attn._forward_pre_hooks for layer in model.layers))

    def test_hooks_are_removed_on_failure(self):
        model = tiny_model()
        native_rope = modeling_llama.apply_rotary_pos_emb
        with tempfile.TemporaryDirectory() as tmp:
            with patch('utils.full_attention_scores.full_fp32_scores',side_effect=RuntimeError('test failure')):
                with self.assertRaisesRegex(RuntimeError,'test failure'):
                    extract_to_npy(model,torch.tensor([[1,2]]),Path(tmp)/'partial.npy',validate_native=True)
        self.assertTrue(all(not layer.self_attn._forward_pre_hooks for layer in model.layers))
        self.assertIs(modeling_llama.apply_rotary_pos_emb,native_rope)

    def test_prefix_check_compares_only_shared_square(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths={}
            for name,length in [('original',3),('gcg',5),('shuffled',5)]:
                data=np.zeros((2,4,length,length),dtype=np.float32)
                data[:,:,:2,:2]=1
                if name=='shuffled': data[:,:,3:,3:]=999
                path=Path(tmp)/(name+'.npy');np.save(path,data);paths[name]=path
            result=prefix_validation(paths,2)
            self.assertTrue(all(x['close'] for x in result['comparisons_to_gcg'].values()))


class CheckpointTests(unittest.TestCase):
    def test_resume_skips_complete_and_rejects_corrupted_scores(self):
        model=SimpleNamespace(model=tiny_model(),device=torch.device('cpu'))
        task=dict(dataset_index=68,condition='original',input_ids=[1,5,9],sequence_length=3,
                  shape=[2,4,3,3],expected_npy_bytes=array_bytes([2,4,3,3]))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);exclusive_json(root/'plan.json',{'test':True})
            runtime=root/'runtime.json';exclusive_json(runtime,{})
            first=capture_one(root,task,model,runtime,True)
            with patch('scripts.extract_full_attention_scores.extract_to_npy',side_effect=AssertionError('must skip')):
                self.assertEqual(capture_one(root,task,model,runtime,True),first)
            score=root/first['score_file']
            with score.open('r+b') as handle:
                handle.seek(-4,2);handle.write(b'xxxx')
            with self.assertRaisesRegex(ValueError,'score hash'):
                completed(root,task)

    def test_failed_attempt_is_preserved_and_resume_uses_new_attempt(self):
        model=SimpleNamespace(model=tiny_model(),device=torch.device('cpu'))
        task=dict(dataset_index=68,condition='gcg',input_ids=[1,5,9],sequence_length=3,
                  shape=[2,4,3,3],expected_npy_bytes=array_bytes([2,4,3,3]))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);exclusive_json(root/'plan.json',{})
            runtime=root/'runtime.json';exclusive_json(runtime,{})
            with patch('scripts.extract_full_attention_scores.extract_to_npy',side_effect=RuntimeError('simulated interruption')):
                with self.assertRaises(RuntimeError): capture_one(root,task,model,runtime,False)
            failures=list(root.rglob('failure.json'));self.assertEqual(len(failures),1)
            before=failures[0].read_bytes()
            self.assertIsNone(completed(root,task))
            capture_one(root,task,model,runtime,False)
            self.assertEqual(before,failures[0].read_bytes())
            self.assertEqual(len(list((root/'inputs/index_0068/gcg').glob('attempt_*'))),2)


@unittest.skipUnless(DATASET.exists(),'local comparison dataset unavailable')
class DatasetTests(unittest.TestCase):
    def test_all_sixty_inputs_are_read_without_tokenization(self):
        with patch('transformers.AutoTokenizer.from_pretrained',side_effect=AssertionError('no tokenizer')):
            tasks,plan=input_tasks()
        self.assertEqual(len(tasks),60)
        self.assertEqual(len({(t['dataset_index'],t['condition']) for t in tasks}),60)
        self.assertTrue(all(t['dataset_index']<220 for t in tasks))
        source=[json.loads(line) for line in DATASET.read_text().splitlines()]
        for task in tasks:
            row=next(r for r in source if r['dataset_index']==task['dataset_index'])
            self.assertEqual(task['input_ids'],row['conditions'][task['condition']]['input_ids'])
            self.assertEqual(task['token_ranges'],row['conditions'][task['condition']]['token_ranges'])


if __name__=='__main__':
    unittest.main()

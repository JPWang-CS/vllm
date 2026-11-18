"""Prepare the two selected official RULER tasks and validate actual chat lengths.

Adapted from the existing Pangu portable evaluation workflow. Keep generator,
task selection, seeds, output budgets and token-budget checks consistent.
"""
import argparse
import importlib.util
import json
import os
import pathlib
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ruler-root', type=pathlib.Path, required=True)
    p.add_argument('--model', type=pathlib.Path, required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    p.add_argument('--raw-root', type=pathlib.Path)
    p.add_argument('--adapt-only', action='store_true')
    p.add_argument('--lengths', type=int, nargs='+', default=[4096, 8192, 16384])
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model-template-token', type=int, default=16)
    p.add_argument('--effective-context-limit', type=int)
    p.add_argument('--chat-template-kwargs', default='{"thinking": false}', help='JSON matching the request configuration')
    p.add_argument('--user-suffix', default='', help='Declared model control suffix, included before token counting')
    a = p.parse_args()
    template_kwargs = json.loads(a.chat_template_kwargs)
    assert isinstance(template_kwargs, dict)
    project = pathlib.Path(__file__).resolve().parents[2]
    assert a.output.resolve().is_relative_to(project), 'output must stay inside this project'
    assert a.samples > 0 and all(v > 0 for v in a.lengths)
    assert a.effective_context_limit is None or a.effective_context_limit > 0
    assert a.model.is_dir() and (a.ruler_root / 'scripts/synthetic.yaml').is_file()
    assert not a.output.exists(), 'Use a new private output directory; existing data is retained'
    a.output.mkdir(parents=True)
    for key, suffix in {'HF_MODULES_CACHE':'cache/hf-modules', 'XDG_CACHE_HOME':'cache',
                        'HF_HOME':'cache/huggingface', 'TMPDIR':'tmp', 'TMP':'tmp', 'TEMP':'tmp'}.items():
        target = a.output / suffix
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
    from transformers import AutoTokenizer
    import yaml
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True, local_files_only=True)
    spec = importlib.util.spec_from_file_location('ruler_constants', a.ruler_root / 'scripts/data/synthetic/constants.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((a.ruler_root / 'scripts/synthetic.yaml').read_text(encoding='utf-8'))
    summary = []
    for length in a.lengths:
        effective_limit = min(length, a.effective_context_limit) if a.effective_context_limit else length
        for name in ['niah_single_1', 'vt']:
            conf = config[name]
            base = module.TASKS[conf['task']]
            budget = base['tokens_to_generate']
            raw_root = a.raw_root if a.adapt_only else a.output
            assert raw_root is not None
            args = [sys.executable, '-B', str(a.ruler_root / 'scripts/data/synthetic' / (conf['task'] + '.py')),
                    '--save_dir', str(raw_root / str(length)), '--save_name', name,
                    '--tokenizer_path', str(a.model), '--tokenizer_type', 'hf',
                    '--max_seq_length', str(effective_limit), '--model_template_token', str(a.model_template_token),
                    '--tokens_to_generate', str(budget), '--num_samples', str(a.samples),
                    '--random_seed', str(a.seed), '--template', base['template'] + base.get('answer_prefix', '')]
            for key, value in conf['args'].items():
                args += ['--' + key, str(value)]
            if not a.adapt_only:
                log = a.output / f'generate-{length}-{name}'
                with log.with_suffix('.stdout').open('w', encoding='utf-8') as stdout, log.with_suffix('.stderr').open('w', encoding='utf-8') as stderr:
                    result = subprocess.run(args, stdout=stdout, stderr=stderr, timeout=180)
                log.with_suffix('.exit').write_text(str(result.returncode), encoding='utf-8')
                assert result.returncode == 0, (name, length, result.returncode)
            source = raw_root / str(length) / name / 'validation.jsonl'
            rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
            assert len(rows) == a.samples
            adapted = []
            for index, row in enumerate(rows):
                prompt = row['input'] + row.get('answer_prefix', '') + a.user_suffix
                actual = len(tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], **template_kwargs, tokenize=True, add_generation_prompt=True))
                assert actual + budget <= effective_limit, (length, effective_limit, name, index, actual, budget)
                adapted.append({'id': f'{name}/{length}/{index}', 'original_index': row['index'],
                                'task': name, 'context_limit': length, 'effective_context_limit': effective_limit, 'prompt': prompt,
                                'outputs': row['outputs'], 'actual_chat_tokens': actual,
                                'output_tokens': budget, 'source_row': index, 'user_suffix': a.user_suffix})
            (a.output / f'{name}-{length}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in adapted), encoding='utf-8')
            summary.append({'task': name, 'limit': length, 'effective_limit': effective_limit, 'samples': len(rows),
                            'actual_input_tokens': [row['actual_chat_tokens'] for row in adapted],
                            'output_budget': budget, 'generator_args': args, 'chat_template_kwargs': template_kwargs,
                            'adapt_only': a.adapt_only, 'raw_source': str(source), 'user_suffix': a.user_suffix})
    (a.output / 'generation-summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

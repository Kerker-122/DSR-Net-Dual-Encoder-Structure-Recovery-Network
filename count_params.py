"""Count unique model parameters without inference or pretrained weights.

Examples:
    python count_params.py
    python count_params.py --json output/parameter_counts.json
    python count_params.py --factory my_model:DSRNet --kwargs '{"channels":128}'

The default reports the checked-in SAM2UNeXT, not the unpublished DSR-Net.
Custom factories must support CPU construction; use --kwargs for their options.
"""

import argparse
import importlib
import json
from pathlib import Path
from unittest.mock import patch


def count_parameters(model):
    """Count shared parameters once; exclude buffers such as running statistics."""
    groups = {}
    for name, parameter in model.named_parameters():
        group = name.split('.')[0] if '.' in name else '(root)'
        row = groups.setdefault(group, {'total': 0, 'trainable': 0, 'frozen': 0})
        size = parameter.numel()
        row['total'] += size
        row['trainable' if parameter.requires_grad else 'frozen'] += size
    totals = {key: sum(row[key] for row in groups.values())
              for key in ('total', 'trainable', 'frozen')}
    return {'counts': totals, 'modules': groups}


def build_model(factory_name, kwargs):
    module_name, separator, attribute = factory_name.partition(':')
    if not separator or not module_name or not attribute:
        raise ValueError('--factory must have the form module:callable')
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if factory_name == 'SAM2UNeXT:SAM2UNeXT':
        # The repository builder defaults to CUDA. Override only construction
        # device locally; keep architecture and requires_grad flags unchanged.
        original_builder = module.build_sam2

        def cpu_builder(*args, **options):
            options['device'] = 'cpu'
            return original_builder(*args, **options)

        if kwargs.get('checkpoint_path') or kwargs.get('dinov2_path'):
            raise ValueError('Default counting needs no checkpoints; omit weight paths.')
        with patch.object(module, 'build_sam2', cpu_builder):
            return factory(**kwargs)
    return factory(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--factory', default='SAM2UNeXT:SAM2UNeXT',
                        help='Model constructor as module:callable')
    parser.add_argument('--kwargs', default='{}', help='Constructor keyword arguments as JSON')
    parser.add_argument('--json', type=Path, help='Optional JSON report path')
    args = parser.parse_args()
    kwargs = json.loads(args.kwargs)
    if not isinstance(kwargs, dict):
        parser.error('--kwargs must be a JSON object')
    model = build_model(args.factory, kwargs)
    report = {'factory': args.factory, 'constructor_kwargs': kwargs,
              'unit': 'M = 1,000,000 parameters', **count_parameters(model)}
    print(f'\nModel: {args.factory}')
    if args.factory == 'SAM2UNeXT:SAM2UNeXT':
        print('Scope: checked-in SAM2-UNeXT baseline; not DSR-Net.')
    print('Unique parameters only; buffers excluded. No forward pass required.\n')
    print(f'{"Module":<26}{"Total (M)":>15}{"Trainable (M)":>17}{"Frozen (M)":>15}')
    for name, row in report['modules'].items():
        print(f'{name:<26}{row["total"]/1e6:>15.6f}'
              f'{row["trainable"]/1e6:>17.6f}{row["frozen"]/1e6:>15.6f}')
    print('-' * 73)
    row = report['counts']
    print(f'{"TOTAL":<26}{row["total"]/1e6:>15.6f}'
          f'{row["trainable"]/1e6:>17.6f}{row["frozen"]/1e6:>15.6f}')
    for key, value in row.items():
        print(f'{key}: {value:,}')
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'Report saved: {args.json.resolve()}')


if __name__ == '__main__':
    main()

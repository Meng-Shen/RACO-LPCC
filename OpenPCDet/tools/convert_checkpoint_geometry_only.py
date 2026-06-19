import argparse
import shutil
from pathlib import Path

import torch


def crop_four_to_three_channels(tensor):
    for dim, size in enumerate(tensor.shape):
        if size == 4:
            slices = [slice(None)] * tensor.dim()
            slices[dim] = slice(0, 3)
            return tensor[tuple(slices)].contiguous(), dim
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description='Convert a PV-RCNN checkpoint from xyz+intensity input to xyz-only input.')
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default='ckpt/model_non_reflectance.pth',
        help='Checkpoint path to convert in place.')
    parser.add_argument(
        '--backup',
        default=None,
        help='Backup path. Defaults to <checkpoint>.bak.')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite an existing backup file.')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    backup_path = Path(args.backup) if args.backup else checkpoint_path.with_suffix(checkpoint_path.suffix + '.bak')
    if backup_path.exists() and not args.force:
        raise FileExistsError(f'Backup already exists: {backup_path}. Use --force to overwrite it.')

    shutil.copy2(checkpoint_path, backup_path)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state') if isinstance(checkpoint, dict) else checkpoint
    if state_dict is None:
        state_dict = checkpoint.get('state_dict')
    if state_dict is None:
        raise KeyError('Cannot find model_state or state_dict in checkpoint.')

    converted = []
    for key, tensor in list(state_dict.items()):
        if key.endswith('conv_input.0.weight') and hasattr(tensor, 'shape') and 4 in tuple(tensor.shape):
            new_tensor, dim = crop_four_to_three_channels(tensor)
            if new_tensor is not None:
                state_dict[key] = new_tensor
                converted.append((key, tuple(tensor.shape), tuple(new_tensor.shape), dim))

    if not converted:
        raise RuntimeError('No 4-channel conv_input.0.weight tensor was found to convert.')

    if isinstance(checkpoint, dict):
        checkpoint['model_state'] = state_dict
        checkpoint['optimizer_state'] = None
        checkpoint['geometry_only_input'] = True
        checkpoint['geometry_only_conversion'] = {
            'dropped_channel': 'reflectance',
            'converted_tensors': [
                {
                    'key': key,
                    'old_shape': old_shape,
                    'new_shape': new_shape,
                    'cropped_dim': dim,
                }
                for key, old_shape, new_shape, dim in converted
            ],
        }
        torch.save(checkpoint, checkpoint_path)
    else:
        torch.save(state_dict, checkpoint_path)

    print(f'Backup written to: {backup_path}')
    print(f'Converted checkpoint written to: {checkpoint_path}')
    for key, old_shape, new_shape, dim in converted:
        print(f'{key}: {old_shape} -> {new_shape}, cropped dim {dim}')


if __name__ == '__main__':
    main()

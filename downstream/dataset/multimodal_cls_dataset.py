# downstream/dataset/multimodal_cls_dataset.py
from pathlib import Path
import json
from monai.data import Dataset
from .transforms_cls import cls_transform_train, cls_transform_val

def _format_item(data_root: Path, item: dict, mix_template: bool, template_dir: str, modality_names):
    images = [str(data_root / p) for p in item['image']]
    d = {'image': images, 'label': int(item['label'])}
    if mix_template:
        # T1-only template list of same length as images
        tpaths = [str(Path(template_dir) / (f'{m}.nii.gz')) for m in modality_names[:len(images)]]
        d['template'] = tpaths
    return d

def get_datasets_cls(args):
    data_root = Path(args.data_root)
    json_path = Path(args.json_file)
    if not json_path.is_file():
        json_path = data_root / args.json_file
    with open(json_path, 'r') as fr:
        data_list = json.load(fr)

    modality_names = ['T1']

    train_list = [_format_item(data_root, it, args.mix_template, args.template_dir, modality_names) for it in data_list['training']]
    val_list   = [_format_item(data_root, it, False,               '',                 modality_names) for it in data_list['validation']]
    test_list  = [_format_item(data_root, it, False,               '',                 modality_names) for it in data_list['test']]

    patch_shape = (args.patch_shape, args.patch_shape, 96)
    resize_shape = (128, 128, 128)
    train_tf = cls_transform_train(
        patch_shape=patch_shape,
        resize_shape=resize_shape,
        enable_channel_cutmix=args.mix_template,
        pair_aug=args.use_cl,
        in_channels=args.in_channels,
    )
    val_tf   = cls_transform_val(patch_shape=patch_shape, resize_shape=resize_shape)


    return Dataset(data=train_list, transform=train_tf), Dataset(data=val_list, transform=val_tf), Dataset(data=test_list, transform=val_tf)

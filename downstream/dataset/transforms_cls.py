# downstream/dataset/transforms_cls.py
import monai.transforms as T
import torch
import time
from monai.transforms.transform import MapTransform
from monai.config import KeysCollection
from typing import Mapping, Hashable, Dict, Optional
from .mix_transform import RandomChannelCutmixd
from utils.custom_trans import CenterCropForegroundd

class CloneKeyd(MapTransform):
    def __init__(self, keys: KeysCollection, new_key_map: Dict[str, str], allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.new_key_map = new_key_map
    def __call__(self, data: Mapping[Hashable, object]):
        d = dict(data)
        for src, dst in self.new_key_map.items():
            d[dst] = d[src]
        return d


class RandomTemplateMaskd(MapTransform):
    def __init__(self, keys: KeysCollection, template_key: str = "template", num_patches: int = 1,
                 min_ratio: float = 0.05, max_ratio: float = 0.2, prob: float = 1.0, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.template_key = template_key
        self.num_patches = num_patches
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.prob = prob

    def __call__(self, data: Mapping[Hashable, object]):
        d = dict(data)
        tpl = d.get(self.template_key, None)
        if tpl is None:
            return d
        for key in self.keys:
            if key not in d:
                continue
            if torch.rand(()) > self.prob:
                continue
            img = d[key]
            if not torch.is_tensor(img) or not torch.is_tensor(tpl):
                continue
            # ensure shapes compatible
            if img.shape != tpl.shape:
                continue
            _, H, W, D = img.shape
            for _ in range(self.num_patches):
                ratio = float(torch.empty(()).uniform_(self.min_ratio, self.max_ratio))
                dz = max(1, int(D * ratio))
                dy = max(1, int(W * ratio))
                dx = max(1, int(H * ratio))
                z0 = int(torch.randint(0, max(1, D - dz + 1), (1,)).item())
                y0 = int(torch.randint(0, max(1, W - dy + 1), (1,)).item())
                x0 = int(torch.randint(0, max(1, H - dx + 1), (1,)).item())
                img[:, x0:x0+dx, y0:y0+dy, z0:z0+dz] = tpl[:, x0:x0+dx, y0:y0+dy, z0:z0+dz]
            d[key] = img
        return d


class RetryLoadImaged(MapTransform):
    """
    Wrapper around MONAI LoadImaged with simple retry to handle transient IO/gzip EOF
    when multiple jobs hit the same storage. Raises after max retries.
    """
    def __init__(self, keys: KeysCollection, retries: int = 3, delay: float = 0.1, allow_missing_keys: bool = False, **kwargs):
        super().__init__(keys, allow_missing_keys)
        self.retries = max(1, int(retries))
        self.delay = max(0.0, float(delay))
        self.loader = T.LoadImaged(keys=keys, **kwargs)

    def __call__(self, data: Mapping[Hashable, object]):
        d = dict(data)
        last_err = None
        for attempt in range(self.retries):
            try:
                return self.loader(d)
            except (EOFError, OSError) as e:
                last_err = e
                if attempt + 1 < self.retries and self.delay > 0:
                    time.sleep(self.delay)
                continue
        if last_err:
            raise last_err
        return d

# def cls_transform_train(patch_shape=(96,96,64), resize_shape=(128,128,64), enable_channel_cutmix=False, pair_aug=False, in_channels=1):
def cls_transform_train(patch_shape=(96,96,96), resize_shape=(128,128,128), enable_channel_cutmix=False, pair_aug=False, in_channels=1):
    img_keys = ['image'] + (['template'] if enable_channel_cutmix else [])

    def _view_ops(img_key: str, tpl_key: Optional[str]):
        spatial_keys = [img_key] + ([tpl_key] if tpl_key else [])
        return [
            T.CenterSpatialCropd(keys=spatial_keys, roi_size=patch_shape),
            T.SpatialPadd(keys=spatial_keys, spatial_size=patch_shape),
            RandomTemplateMaskd(keys=[img_key], template_key=tpl_key if tpl_key else 'template',
                                num_patches=1, min_ratio=0.05, max_ratio=0.2, prob=1.0, allow_missing_keys=True),
            T.RandFlipd(keys=[img_key], spatial_axis=(0,1,2), prob=0.5),
            T.RandShiftIntensityd(keys=[img_key], offsets=0.1, prob=1.0),
            T.RandScaleIntensityd(keys=[img_key], factors=0.1, prob=1.0),
        ]

    ops = [
        RetryLoadImaged(keys=img_keys, retries=3, delay=0.1),
        T.EnsureChannelFirstd(keys=img_keys, channel_dim='no_channel'),
        T.Lambdad(keys=img_keys, func=lambda x: x if x.ndim == 4 else x.unsqueeze(-1)),
        T.EnsureTyped(keys=img_keys, dtype=torch.float32, track_meta=True),
        T.ScaleIntensityRangePercentilesd(keys=['image'], lower=5, upper=95, b_min=0.0, b_max=1.0, channel_wise=True, clip=True),
        T.Orientationd(keys=img_keys, axcodes='RAS'),
        T.Spacingd(keys=img_keys, pixdim=(1.0,1.0,1.0),
                   mode=(['bilinear'] * len(img_keys))),
        CenterCropForegroundd(keys=img_keys, source_key='image', margin=1),
        T.Resized(keys=img_keys, spatial_size=resize_shape, mode='trilinear'),
    ]

    if pair_aug:
        clone_map = {'image':'image2'}
        tpl1, tpl2 = (None, None)
        if enable_channel_cutmix:
            clone_map['template'] = 'template2'
            tpl1, tpl2 = 'template', 'template2'
        ops += [CloneKeyd(keys=list(clone_map.keys()), new_key_map=clone_map)]
        ops += _view_ops('image', tpl1)
        ops += _view_ops('image2', tpl2)
        if enable_channel_cutmix:
            ops += [
                RandomChannelCutmixd(keys=['image'],  num_mix=3, pair_aug=False, max_num_channel=in_channels),
                RandomChannelCutmixd(keys=['image2'], num_mix=3, pair_aug=False, max_num_channel=in_channels),
            ]
    else:
        tpl_key = 'template' if enable_channel_cutmix else None
        ops += _view_ops('image', tpl_key)
        if enable_channel_cutmix:
            ops += [RandomChannelCutmixd(keys=['image'], num_mix=3, pair_aug=False, max_num_channel=in_channels)]

    ops += [
        T.EnsureTyped(keys=['label'], dtype=torch.long, track_meta=False),
    ]
    return T.Compose(ops)

def cls_transform_val(patch_shape=(96, 96, 96), resize_shape=(128, 128, 128)):
    return T.Compose([
        RetryLoadImaged(keys=['image'], retries=3, delay=0.1),
        T.EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'),
        T.Lambdad(keys=['image'], func=lambda x: x if x.ndim == 4 else x.unsqueeze(-1)),
        T.EnsureTyped(keys=['image'], dtype=torch.float32, track_meta=True),

        T.ScaleIntensityRangePercentilesd(
            keys=['image'], lower=5, upper=95, b_min=0.0, b_max=1.0, channel_wise=True, clip=True
        ),
        T.Orientationd(keys=['image'], axcodes='RAS'),
        T.Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode=['bilinear']),
        CenterCropForegroundd(keys=['image'], source_key='image', margin=1),
        T.Resized(keys=['image'], spatial_size=resize_shape, mode='trilinear'),
        T.SpatialPadd(keys=['image'], spatial_size=patch_shape),
        T.CenterSpatialCropd(keys=['image'], roi_size=patch_shape),

        # EITHER keep this:
        T.EnsureTyped(keys=['label'], dtype=torch.long, track_meta=False),
        # OR replace the previous line with this if you still see label collate errors:
        # T.Lambdad(keys=['label'], func=lambda x: int(x)),
    ])

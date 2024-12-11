# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class whdlddataset(BaseSegDataset):

    METAINFO = dict(
        classes=('building', 'road', 'pavement', 'vegetation', 'bare soil', 'water'),
        palette=[[255, 0, 0], [255, 255, 0], [192, 192, 0], [0, 255, 0], [128, 128, 128], [0, 0, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

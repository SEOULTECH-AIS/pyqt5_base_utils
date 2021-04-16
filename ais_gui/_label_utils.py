from ais_utils import _error
from ais_utils import _cv2

import numpy as np


BDD_100k = {
    1: {"color": (0x00, 0x00, 0x00), "name": "unlabeled or statick"},
    2: {"color": (0x00, 0x4A, 0x6F), "name": "dynamic"},
    3: {"color": (0x51, 0x00, 0x51), "name": "ground"},
    4: {"color": (0xA0, 0xAA, 0xFA), "name": "parking"},
    5: {"color": (0x8C, 0x96, 0xE6), "name": "rail track"},
    6: {"color": (0x80, 0x40, 0x80), "name": "road"},
    7: {"color": (0xE8, 0x23, 0xF4), "name": "sidewalk"},
    8: {"color": (0x64, 0x64, 0x96), "name": "bridge"},
    9: {"color": (0x46, 0x46, 0x46), "name": "building"},
    10: {"color": (0x99, 0x99, 0xBE), "name": "fence"},
    11: {"color": (0xB4, 0x64, 0xB4), "name": "garage"},
    12: {"color": (0xB4, 0xA5, 0xB4), "name": "guard rail"},
    13: {"color": (0x5A, 0x78, 0x96), "name": "tunnel"},
    14: {"color": (0x9C, 0x66, 0x66), "name": "wall"},
    15: {"color": (0x64, 0xAA, 0xFA), "name": "banner"},
    16: {"color": (0xFA, 0xDC, 0xDC), "name": "billboard"},
    17: {"color": (0x00, 0xA5, 0xFF), "name": "lane divider"},
    18: {"color": (0x99, 0x99, 0x99), "name": "pole"},
    19: {"color": (0x64, 0xDC, 0xDC), "name": "street light"},
    20: {"color": (0x00, 0x46, 0xFF), "name": "traffic cone"},
    21: {"color": (0xDC, 0xDC, 0xDC), "name": "traffic device"},
    22: {"color": (0x1E, 0xAA, 0xFA), "name": "traffic light"},
    23: {"color": (0x00, 0xDC, 0xDC), "name": "traffic sign"},
    24: {"color": (0xFA, 0xAA, 0xFA), "name": "traffic sign frame"},
    25: {"color": (0x98, 0xFB, 0x98), "name": "terrain"},
    26: {"color": (0x23, 0x8E, 0x6B), "name": "vegetation"},
    27: {"color": (0xB4, 0x82, 0x46), "name": "sky"},
    28: {"color": (0x3C, 0x14, 0xDC), "name": "person or parking sign"},
    29: {"color": (0x00, 0x00, 0xFF), "name": "rider"},
    30: {"color": (0x20, 0x0B, 0x77), "name": "bicycle"},
    31: {"color": (0x64, 0x3C, 0x00), "name": "bus"},
    32: {"color": (0x8E, 0x00, 0x00), "name": "car"},
    33: {"color": (0x5A, 0x00, 0x00), "name": "caravan"},
    34: {"color": (0xE6, 0x00, 0x00), "name": "motorcycle"},
    35: {"color": (0x6E, 0x00, 0x00), "name": "trailer"},
    36: {"color": (0x64, 0x50, 0x00), "name": "train"},
    37: {"color": (0x46, 0x00, 0x00), "name": "truck"}}

CD_net_2014 = {
    1: {"color": (0x00, 0x00, 0x00), "name": "static"},
    2: {"color": (0x32, 0x32, 0x32), "name": "Hard shadow"},
    3: {"color": (0x55, 0x55, 0x55), "name": "Outside region of interest"},
    4: {"color": (0xAA, 0xAA, 0xAA), "name": "Unknown motion"},
    5: {"color": (0xFF, 0xFF, 0xFF), "name": "Motion"}}

YTOVS = {
    1: {"color": (0x51, 0x46, 0x6B), "name": "person"},
    2: {"color": (0x00, 0x4A, 0x6F), "name": "giant_panda"},
    3: {"color": (0x51, 0x00, 0x51), "name": "lizard"},
    4: {"color": (0xA0, 0xAA, 0xFA), "name": "parrot"},
    5: {"color": (0x8C, 0x96, 0xE6), "name": "skateboard"},
    6: {"color": (0x80, 0x40, 0x80), "name": "sedan"},
    7: {"color": (0xE8, 0x23, 0xF4), "name": "ape"},
    8: {"color": (0x64, 0x64, 0x96), "name": "dog"},
    9: {"color": (0x46, 0x46, 0x46), "name": "snake"},
    10: {"color": (0x99, 0x99, 0xBE), "name": "monkey"},
    11: {"color": (0xB4, 0x64, 0xB4), "name": "hand"},
    12: {"color": (0xB4, 0xA5, 0xB4), "name": "rabbit"},
    13: {"color": (0x5A, 0x78, 0x96), "name": "duck"},
    14: {"color": (0x9C, 0x66, 0x66), "name": "cat"},
    15: {"color": (0x64, 0xAA, 0xFA), "name": "cow"},
    16: {"color": (0xFA, 0xDC, 0xDC), "name": "fish"},
    17: {"color": (0x00, 0xA5, 0xFF), "name": "train"},
    18: {"color": (0x99, 0x99, 0x99), "name": "horse"},
    19: {"color": (0x64, 0xDC, 0xDC), "name": "turtle"},
    20: {"color": (0x00, 0x46, 0xFF), "name": "bear"},
    21: {"color": (0xDC, 0xDC, 0xDC), "name": "motorbike"},
    22: {"color": (0x1E, 0xAA, 0xFA), "name": "giraffe"},
    23: {"color": (0x00, 0xDC, 0xDC), "name": "leopard"},
    24: {"color": (0xFA, 0xAA, 0xFA), "name": "fox"},
    25: {"color": (0x98, 0xFB, 0x98), "name": "deer"},
    26: {"color": (0x23, 0x8E, 0x6B), "name": "owl"},
    27: {"color": (0xB4, 0x82, 0x46), "name": "surfboard"},
    28: {"color": (0x3C, 0x14, 0xDC), "name": "airplane"},
    29: {"color": (0x00, 0x00, 0xFF), "name": "truck"},
    30: {"color": (0x20, 0x0B, 0x77), "name": "zebra"},
    31: {"color": (0x64, 0x3C, 0x00), "name": "tiger"},
    32: {"color": (0x8E, 0x00, 0x00), "name": "elephant"},
    33: {"color": (0x5A, 0x00, 0x00), "name": "snowboard"},
    34: {"color": (0xE6, 0x00, 0x00), "name": "boat"},
    35: {"color": (0x6E, 0x00, 0x00), "name": "shark"},
    36: {"color": (0x64, 0x50, 0x00), "name": "mouse"},
    37: {"color": (0x46, 0x00, 0x00), "name": "frog"},
    38: {"color": (0xDE, 0x21, 0x3D), "name": "eagle"},
    39: {"color": (0xCE, 0x46, 0x47), "name": "earless_seal"},
    40: {"color": (0x0C, 0x29, 0xB9), "name": "tennis_racket"}}

SUPORT_LIST = {
    "CDnet-2014": CD_net_2014,
    "YTOVS": YTOVS,
    "BDD-100k": BDD_100k
}


class _label_dict():
    def __init__(self, label_style):
        self.make_label_dict(label_style)

    def make_label_dict(self, label_style):
        if label_style == "BDD-100k":
            self.label_dict = BDD_100k

        elif label_style == "CDnet-2014":
            self.label_dict = CD_net_2014

        elif label_style == "YTOVS":
            self.label_dict = YTOVS

        else:
            _error.Custom_Variable_Error(
                loacation="label utils / visualize",
                parameters=str(label_style),
                detail="this data set label is not supported"
            )

        self.inverse_dict = {}

        for _tmp_key in self.label_dict.keys():
            _name = self.label_dict[_tmp_key]["name"]
            self.inverse_dict[_name] = _tmp_key

    def name_to_id(self, name):
        _is_exist = name in self.inverse_dict.keys()
        id_num = self.inverse_dict[name] if _is_exist else -1
        return _is_exist, id_num

    def id_to_class_dict(self, id_num):
        _is_exist = id_num in self.label_dict.keys()
        class_dict = self.label_dict[id_num] if _is_exist else \
            {"color": (0x00, 0x00, 0x00), "name": "Error"}
        return _is_exist, class_dict

    def make_label_image(self, label_info_list, label_selction, dispaly_categories, input_img, is_mixing):
        _displayed_label = [label_info_list[_ct] for _ct in label_selction]
        _h, _w, _c = np.shape(input_img)

        _base = np.zeros((_h, _w, _c), np.uint8)
        _seg_bool = np.zeros((_h, _w), np.uint8)

        if "segmentation" in dispaly_categories:
            for _label in _displayed_label:
                _info = self.id_to_class_dict(_label["class"])[1]
                _3c_seg = np.dstack([_label["segmentation"], _label["segmentation"], _label["segmentation"]])

                _base = (_base * (1 - _3c_seg)) + (_info["color"] * _3c_seg).astype(np.uint8)
                _seg_bool = np.logical_or(_seg_bool, _label["segmentation"])

        if is_mixing:
            _base = ((0.3 * _base).astype(np.uint8) + (0.7 * input_img).astype(np.uint8))\
                if np.max(_seg_bool) else input_img

        if "box" in dispaly_categories:
            for _label in _displayed_label:
                if _label["box"][:2] is not None:
                    _info = self.id_to_class_dict(_label["class"])[1]

                    _start_w, _start_h, _delta_w, _delta_h = _label["box"]
                    _start_point = (int(_start_w), int(_start_h))
                    _end_point = (int(_start_w + _delta_w), int(_start_h + _delta_h))
                    _base = _cv2.cv2.rectangle(_base.copy(), _start_point, _end_point, _info["color"], 3)

                    # _s_w, _s_h, _e_w, _e_h = _label["box"]
                    # pts = np.array([[_s_h, _s_w], [_s_h, _e_w], [_e_h, _e_w], [_e_h, _s_w]], np.int32)
                    # pts = pts.reshape((-1, 1, 2))
                    # _base = cv2.polylines(_base, [pts], True, _info["color"], 3)

        return _base

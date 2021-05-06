from ais_utils import _error
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

COCO = {
    1: {"color": (0X38, 0XA4, 0XC4), "name": "person"},
    2: {"color": (0XB8, 0X24, 0X63), "name": "bicycle"},
    3: {"color": (0XB8, 0XC4, 0X59), "name": "car"},
    4: {"color": (0XE2, 0X3D, 0X72), "name": "motorcycle"},
    5: {"color": (0XEA, 0X59, 0XDD), "name": "airplane"},
    6: {"color": (0X3D, 0X90, 0X6D), "name": "bus"},
    7: {"color": (0X38, 0XB0, 0X24), "name": "train"},
    8: {"color": (0XCE, 0X3B, 0X42), "name": "truck"},
    9: {"color": (0X24, 0X45, 0X86), "name": "boat"},
    10: {"color": (0XE7, 0X31, 0X36), "name": "traffic light"},
    11: {"color": (0X8B, 0X0B, 0XEC), "name": "fire hydrant"},
    13: {"color": (0XAE, 0X1A, 0X9F), "name": "stop sign"},
    14: {"color": (0XAB, 0X9C, 0XC2), "name": "parking meter"},
    15: {"color": (0X90, 0XBD, 0XBF), "name": "bench"},
    16: {"color": (0XF1, 0X3B, 0XB3), "name": "bird"},
    17: {"color": (0X24, 0X9A, 0XB5), "name": "cat"},
    18: {"color": (0X40, 0X9C, 0X65), "name": "dog"},
    19: {"color": (0X95, 0XC2, 0X65), "name": "horse"},
    20: {"color": (0XB8, 0X04, 0X3B), "name": "sheep"},
    21: {"color": (0XA6, 0XDD, 0X92), "name": "cow"},
    22: {"color": (0X9A, 0X3B, 0X79), "name": "elephant"},
    23: {"color": (0XEA, 0X1A, 0XCE), "name": "bear"},
    24: {"color": (0XE2, 0X42, 0XFB), "name": "zebra"},
    25: {"color": (0XCC, 0X77, 0X74), "name": "giraffe"},
    27: {"color": (0X6D, 0XD6, 0XB3), "name": "backpack"},
    28: {"color": (0X18, 0X3B, 0X31), "name": "umbrella"},
    31: {"color": (0X72, 0X74, 0X1D), "name": "handbag"},
    32: {"color": (0X60, 0XFE, 0X79), "name": "tie"},
    33: {"color": (0X63, 0X63, 0XA6), "name": "suitcase"},
    34: {"color": (0X3B, 0XE5, 0XA1), "name": "frisbee"},
    35: {"color": (0X24, 0X1F, 0XFB), "name": "skis"},
    36: {"color": (0XB3, 0X29, 0XD1), "name": "snowboard"},
    37: {"color": (0X88, 0XB0, 0XD8), "name": "sports ball"},
    38: {"color": (0X0E, 0X06, 0X3D), "name": "kite"},
    39: {"color": (0X24, 0X83, 0X2E), "name": "baseball bat"},
    40: {"color": (0X6A, 0XB3, 0X15), "name": "baseball glove"},
    41: {"color": (0X6D, 0X8B, 0X92), "name": "skateboard"},
    42: {"color": (0XBF, 0X51, 0X40), "name": "surfboard"},
    43: {"color": (0XE0, 0X8B, 0XB3), "name": "tennis racket"},
    44: {"color": (0X22, 0XCC, 0X2E), "name": "bottle"},
    46: {"color": (0XF9, 0X36, 0X7E), "name": "wine glass"},
    47: {"color": (0X9C, 0X65, 0XA1), "name": "cup"},
    48: {"color": (0XB5, 0X79, 0X9F), "name": "fork"},
    49: {"color": (0XA1, 0XE7, 0X72), "name": "knife"},
    50: {"color": (0X15, 0X40, 0XB0), "name": "spoon"},
    51: {"color": (0X2C, 0X0E, 0X97), "name": "bowl"},
    52: {"color": (0XF9, 0XBA, 0XB5), "name": "banana"},
    53: {"color": (0XC7, 0X36, 0X6F), "name": "apple"},
    54: {"color": (0X47, 0X15, 0X4A), "name": "sandwich"},
    55: {"color": (0X65, 0X3D, 0XBA), "name": "orange"},
    56: {"color": (0X63, 0XDB, 0XC2), "name": "broccoli"},
    57: {"color": (0X77, 0XCC, 0X31), "name": "carrot"},
    58: {"color": (0X8D, 0X95, 0XB0), "name": "hot dog"},
    59: {"color": (0X81, 0XAE, 0XE0), "name": "pizza"},
    60: {"color": (0X79, 0X36, 0XEA), "name": "donut"},
    61: {"color": (0XEA, 0X97, 0X6A), "name": "cake"},
    62: {"color": (0X3B, 0X4F, 0X63), "name": "chair"},
    63: {"color": (0X18, 0X51, 0XE5), "name": "couch"},
    64: {"color": (0XAB, 0X60, 0X95), "name": "potted plant"},
    65: {"color": (0X72, 0X45, 0X72), "name": "bed"},
    67: {"color": (0X77, 0XC2, 0X31), "name": "dining table"},
    70: {"color": (0X01, 0X29, 0X09), "name": "toilet"},
    72: {"color": (0XA9, 0X81, 0X7C), "name": "tv"},
    73: {"color": (0X4F, 0X51, 0X0E), "name": "laptop"},
    74: {"color": (0XA6, 0XFB, 0X4C), "name": "mouse"},
    75: {"color": (0X60, 0X77, 0XD1), "name": "remote"},
    76: {"color": (0XA4, 0XEC, 0X01), "name": "keyboard"},
    77: {"color": (0XE5, 0X7C, 0X5E), "name": "cell phone"},
    78: {"color": (0X4A, 0X4C, 0X54), "name": "microwave"},
    79: {"color": (0XF6, 0X9C, 0X0B), "name": "oven"},
    80: {"color": (0X59, 0X1F, 0X31), "name": "toaster"},
    81: {"color": (0XD1, 0XE2, 0XE0), "name": "sink"},
    82: {"color": (0XA1, 0XA9, 0X3B), "name": "refrigerator"},
    84: {"color": (0X33, 0X88, 0XF1), "name": "book"},
    85: {"color": (0X7E, 0XA9, 0XC9), "name": "clock"},
    86: {"color": (0XF1, 0X95, 0XC2), "name": "vase"},
    87: {"color": (0XD8, 0X77, 0X79), "name": "scissors"},
    88: {"color": (0XD8, 0XF1, 0X06), "name": "teddy bear"},
    89: {"color": (0XBA, 0XF4, 0X9C), "name": "hair drier"},
    90: {"color": (0X31, 0X83, 0XD3), "name": "toothbrush"},
    92: {"color": (0X31, 0X6D, 0XEF), "name": "banner"},
    93: {"color": (0X74, 0XBF, 0XC4), "name": "blanket"},
    95: {"color": (0X95, 0XEF, 0X92), "name": "bridge"},
    100: {"color": (0XAB, 0X33, 0X92), "name": "cardboard"},
    107: {"color": (0XA1, 0X9A, 0X04), "name": "counter"},
    109: {"color": (0X92, 0X27, 0XAB), "name": "curtain"},
    112: {"color": (0X45, 0XA6, 0X97), "name": "door-stuff"},
    118: {"color": (0X60, 0XF6, 0X83), "name": "floor-wood"},
    119: {"color": (0X56, 0X27, 0X59), "name": "flower"},
    122: {"color": (0X92, 0X81, 0X51), "name": "fruit"},
    125: {"color": (0XA9, 0XC2, 0XEC), "name": "gravel"},
    128: {"color": (0X10, 0XA9, 0X97), "name": "house"},
    130: {"color": (0XD3, 0XC4, 0XB8), "name": "light"},
    133: {"color": (0X54, 0XF4, 0X04), "name": "mirror-stuff"},
    138: {"color": (0X47, 0X3D, 0X4A), "name": "net"},
    141: {"color": (0X33, 0X22, 0X45), "name": "pillow"},
    144: {"color": (0X27, 0X45, 0XBF), "name": "platform"},
    145: {"color": (0XE2, 0X1A, 0X27), "name": "playingfield"},
    147: {"color": (0X36, 0XEF, 0X3D), "name": "railroad"},
    148: {"color": (0XE5, 0X8D, 0XB0), "name": "river"},
    149: {"color": (0X72, 0XFB, 0X7E), "name": "road"},
    151: {"color": (0X59, 0XDB, 0X5B), "name": "roof"},
    154: {"color": (0X97, 0XA4, 0XE5), "name": "sand"},
    155: {"color": (0X60, 0X15, 0X7C), "name": "sea"},
    156: {"color": (0XD8, 0XC7, 0XDD), "name": "shelf"},
    159: {"color": (0X72, 0X9A, 0XEA), "name": "snow"},
    161: {"color": (0X86, 0XC9, 0X9F), "name": "stairs"},
    166: {"color": (0XBA, 0X5B, 0XA1), "name": "tent"},
    168: {"color": (0X81, 0XAB, 0X9C), "name": "towel"},
    171: {"color": (0X59, 0XEC, 0X97), "name": "wall-brick"},
    175: {"color": (0X45, 0X2C, 0XE7), "name": "wall-stone"},
    176: {"color": (0X72, 0X97, 0X24), "name": "wall-tile"},
    177: {"color": (0X90, 0X6D, 0X1A), "name": "wall-wood"},
    178: {"color": (0X9F, 0X2E, 0XEF), "name": "water-other"},
    180: {"color": (0XEC, 0X22, 0X83), "name": "window-blind"},
    181: {"color": (0X92, 0X09, 0X92), "name": "window-other"},
    184: {"color": (0XF6, 0XAB, 0X9F), "name": "tree-merged"},
    185: {"color": (0X01, 0X3B, 0XA6), "name": "fence-merged"},
    186: {"color": (0X4F, 0XE7, 0XDD), "name": "ceiling-merged"},
    187: {"color": (0X8B, 0X81, 0X88), "name": "sky-other-merged"},
    188: {"color": (0X4C, 0X40, 0X95), "name": "cabinet-merged"},
    189: {"color": (0X51, 0XAB, 0X2E), "name": "table-merged"},
    190: {"color": (0X97, 0X27, 0XB5), "name": "floor-other-merged"},
    191: {"color": (0XA1, 0X13, 0X3B), "name": "pavement-merged"},
    192: {"color": (0XD6, 0X2E, 0X3B), "name": "mountain-merged"},
    193: {"color": (0X68, 0XFE, 0X2C), "name": "grass-merged"},
    194: {"color": (0XE7, 0X4A, 0X31), "name": "dirt-merged"},
    195: {"color": (0X4F, 0X9A, 0X86), "name": "paper-merged"},
    196: {"color": (0X38, 0XEA, 0XE0), "name": "food-other-merged"},
    197: {"color": (0XEF, 0X95, 0XD6), "name": "building-other-merged"},
    198: {"color": (0X97, 0X6D, 0X2C), "name": "rock-merged"},
    199: {"color": (0X56, 0X47, 0XE2), "name": "wall-other-merged"},
    200: {"color": (0X72, 0X0B, 0XF6), "name": "rug-merged"}}

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
    "BDD-100k": BDD_100k,
    "COCO": COCO
}


class _label_dict():
    def __init__(self, label_style):
        self.make_label_dict(label_style)

    def make_label_dict(self, label_style):
        self.label_style = label_style
        if label_style == "BDD-100k":
            self.label_dict = BDD_100k

        elif label_style == "CDnet-2014":
            self.label_dict = CD_net_2014

        elif label_style == "YTOVS":
            self.label_dict = YTOVS
        
        elif label_style == "COCO":
            self.label_dict = COCO

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

    def id_check(self, ID):
        return ID in self.label_dict.keys()

    def Class_to_Id(self, name):
        _is_exist = name in self.inverse_dict.keys()
        id_num = self.inverse_dict[name] if _is_exist else -1
        return _is_exist, id_num

    def Id_to_Calss(self, id_num):
        _is_exist = id_num in self.label_dict.keys()
        class_dict = self.label_dict[id_num] if _is_exist else \
            {"color": (0x00, 0x00, 0x00), "name": "Error"}
        return _is_exist, class_dict

    def make_label_image(self, class_channel_data):
        _h, _w, _ = np.shape(class_channel_data)

        _base = np.zeros((_h, _w, 3), np.uint8)
        for _ct, _key in enumerate(self.label_dict.keys()):
            _color = self.label_dict[_key]["color"]
            _base += (_color * class_channel_data[:, :, _ct][:, :, np.newaxis]).astype(np.uint8)

        return _base

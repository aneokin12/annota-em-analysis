

from model import dawid_skene
import numpy as np
from metrics import (
    macro_f1,
    positive_recall,
    positive_precision,
    negative_precision,
    negative_recall,
    accuracy
)
from labelprocessing import process_labels, get_valids

# ------------------------------------------------------------
#  Utility: load vanilla EM predictions (one entry per line)
# ------------------------------------------------------------

def load_vanilla_preds(concat = True):
    """Return the *hard‑coded* vanilla EM predictions for **all** 1323 lines."""
    o1 = [-1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 1, 1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 1, 1, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 0, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, 0, -1, -1, -1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1]
    o2 = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 1, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 0, 0, 0, 0, -1, -1, -1, 1, 1, 0, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, -1, -1, -1, -1, 0, 0, 0, 1, 1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1]
    
    o4 = [-1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, 1, 1, 1, 1, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, 1, 1, 1, 1, 0, -1, -1, -1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 0, 1, 1, 1, -1, -1, -1, 0, 0, -1, -1, -1, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, 1, 1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0]
    
    o3 = [-1, -1, -1, -1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 1, 1, 1, 1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1]
    
    # print(len(o1),len(o2), len(o3), len(o4))
    if concat:

        return np.concatenate([o1, o2, o3, o4])
    else:
        return [o1, o2, o3, o4]

ROW_RANGES = {0: (0, 40), 1: (40, 56), 2: (56, 94), 3: (94, 139)}
TASK_MAP = {0: (8, 1), 1: (9, 1), 2: (10, 1), 3: (67, 0), 4: (68, 0), 5: (78, 1), 6: (79, 1), 7: (80, 1), 8: (81, 1), 9: (82, 1), 10: (83, 1), 11: (87, 0), 12: (150, 1), 13: (151, 1), 14: (163, 1), 15: (164, 1), 16: (165, 1), 17: (166, 1), 18: (167, 1), 19: (233, 0), 20: (245, 1), 21: (278, 1), 22: (279, 1), 23: (280, 1), 24: (281, 1), 25: (282, 1), 26: (283, 1), 27: (284, 1), 28: (285, 1), 29: (286, 1), 30: (290, 1), 31: (291, 1), 32: (292, 1), 33: (301, 1), 34: (302, 1), 35: (303, 1), 36: (313, 1), 37: (314, 1), 38: (315, 1), 39: (316, 1), 40: (35, 1), 41: (81, 1), 42: (90, 1), 43: (110, 1), 44: (111, 1), 45: (173, 1), 46: (174, 1), 47: (251, 1), 48: (252, 1), 49: (253, 1), 50: (360, 1), 51: (361, 1), 52: (409, 1), 53: (410, 1), 54: (411, 1), 55: (412, 1), 56: (4, 1), 57: (5, 1), 58: (6, 1), 59: (7, 1), 60: (8, 1), 61: (20, 1), 62: (21, 1), 63: (24, 0), 64: (34, 1), 65: (35, 1), 66: (36, 1), 67: (37, 1), 68: (38, 1), 69: (39, 1), 70: (63, 1), 71: (64, 1), 72: (65, 1), 73: (66, 1), 74: (73, 1), 75: (74, 1), 76: (93, 1), 77: (94, 1), 78: (95, 1), 79: (96, 0), 80: (97, 1), 81: (98, 1), 82: (99, 1), 83: (100, 1), 84: (101, 1), 85: (102, 1), 86: (103, 1), 87: (114, 1), 88: (115, 1), 89: (116, 1), 90: (214, 1), 91: (215, 1), 92: (216, 1), 93: (217, 1), 94: (17, 1), 95: (18, 1), 96: (19, 1), 97: (20, 1), 98: (21, 1), 99: (22, 1), 100: (23, 1), 101: (24, 1), 102: (28, 1), 103: (29, 1), 104: (32, 1), 105: (33, 1), 106: (34, 1), 107: (35, 1), 108: (36, 1), 109: (37, 1), 110: (38, 1), 111: (58, 1), 112: (59, 1), 113: (60, 1), 114: (64, 1), 115: (65, 1), 116: (66, 1), 117: (70, 1), 118: (71, 1), 119: (72, 1), 120: (73, 1), 121: (74, 1), 122: (75, 1), 123: (76, 1), 124: (77, 1), 125: (78, 1), 126: (93, 1), 127: (94, 1), 128: (121, 1), 129: (122, 1), 130: (123, 1), 131: (124, 1), 132: (129, 1), 133: (149, 0), 134: (232, 1), 135: (233, 1), 136: (249, 1), 137: (250, 1), 138: (251, 1)}
# TASK_MAP = {0: (4, 0), 1: (5, 0), 2: (6, 0), 3: (8, 1), 4: (9, 1), 5: (10, 1), 6: (26, 0), 7: (30, 1), 8: (31, 1), 9: (32, 1), 10: (33, 1), 11: (38, 1), 12: (39, 1), 13: (40, 1), 14: (44, 1), 15: (45, 1), 16: (49, 1), 17: (53, 1), 18: (54, 1), 19: (55, 1), 20: (56, 1), 21: (57, 1), 22: (58, 1), 23: (59, 1), 24: (60, 1), 25: (61, 1), 26: (62, 1), 27: (63, 1), 28: (67, 0), 29: (68, 0), 30: (72, 0), 31: (76, 0), 32: (78, 1), 33: (79, 1), 34: (80, 1), 35: (81, 1), 36: (82, 1), 37: (83, 1), 38: (87, 0), 39: (88, 0), 40: (89, 0), 41: (90, 0), 42: (95, 0), 43: (96, 0), 44: (97, 0), 45: (102, 0), 46: (103, 0), 47: (107, 0), 48: (108, 0), 49: (109, 0), 50: (110, 0), 51: (111, 0), 52: (112, 0), 53: (113, 0), 54: (114, 0), 55: (120, 0), 56: (121, 0), 57: (122, 0), 58: (126, 0), 59: (131, 0), 60: (136, 0), 61: (137, 0), 62: (138, 0), 63: (140, 0), 64: (141, 0), 65: (145, 0), 66: (146, 0), 67: (150, 1), 68: (151, 1), 69: (155, 0), 70: (163, 1), 71: (164, 1), 72: (165, 1), 73: (166, 1), 74: (167, 1), 75: (172, 0), 76: (173, 0), 77: (184, 0), 78: (185, 0), 79: (186, 0), 80: (187, 0), 81: (195, 0), 82: (200, 0), 83: (201, 0), 84: (202, 0), 85: (206, 0), 86: (211, 0), 87: (212, 0), 88: (213, 0), 89: (217, 0), 90: (221, 0), 91: (229, 0), 92: (233, 0), 93: (237, 0), 94: (241, 0), 95: (245, 1), 96: (246, 0), 97: (250, 0), 98: (259, 0), 99: (260, 0), 100: (261, 0), 101: (262, 0), 102: (266, 0), 103: (267, 0), 104: (278, 1), 105: (279, 1), 106: (280, 1), 107: (281, 1), 108: (282, 1), 109: (283, 1), 110: (284, 1), 111: (285, 1), 112: (286, 1), 113: (290, 1), 114: (291, 1), 115: (292, 1), 116: (301, 1), 117: (302, 1), 118: (303, 1), 119: (307, 0), 120: (313, 1), 121: (314, 1), 122: (315, 1), 123: (316, 1), 124: (334, 0), 125: (335, 0), 126: (336, 0), 127: (4, 0), 128: (5, 0), 129: (6, 0), 130: (8, 0), 131: (9, 0), 132: (10, 0), 133: (11, 0), 134: (13, 0), 135: (14, 0), 136: (19, 0), 137: (20, 0), 138: (21, 0), 139: (24, 0), 140: (25, 0), 141: (27, 0), 142: (28, 0), 143: (29, 0), 144: (30, 0), 145: (31, 0), 146: (35, 1), 147: (36, 0), 148: (37, 0), 149: (42, 0), 150: (43, 0), 151: (44, 0), 152: (45, 0), 153: (46, 0), 154: (48, 0), 155: (52, 0), 156: (55, 0), 157: (56, 0), 158: (58, 0), 159: (59, 0), 160: (60, 0), 161: (61, 0), 162: (65, 0), 163: (66, 0), 164: (67, 0), 165: (68, 0), 166: (70, 0), 167: (74, 0), 168: (75, 0), 169: (81, 1), 170: (82, 0), 171: (83, 0), 172: (90, 1), 173: (92, 0), 174: (93, 0), 175: (94, 0), 176: (95, 0), 177: (96, 0), 178: (97, 0), 179: (101, 0), 180: (102, 0), 181: (103, 0), 182: (104, 0), 183: (105, 0), 184: (110, 1), 185: (111, 1), 186: (112, 0), 187: (121, 0), 188: (125, 0), 189: (126, 0), 190: (127, 0), 191: (128, 0), 192: (129, 0), 193: (130, 0), 194: (131, 0), 195: (147, 0), 196: (148, 0), 197: (149, 0), 198: (151, 0), 199: (152, 1), 200: (153, 1), 201: (154, 1), 202: (155, 1), 203: (159, 1), 204: (160, 1), 205: (161, 1), 206: (162, 1), 207: (163, 1), 208: (164, 1), 209: (165, 1), 210: (166, 1), 211: (173, 1), 212: (174, 1), 213: (192, 1), 214: (193, 1), 215: (194, 1), 216: (195, 1), 217: (196, 1), 218: (209, 1), 219: (210, 1), 220: (211, 0), 221: (212, 0), 222: (213, 0), 223: (217, 0), 224: (218, 0), 225: (219, 0), 226: (220, 0), 227: (221, 0), 228: (222, 0), 229: (224, 0), 230: (225, 0), 231: (227, 1), 232: (228, 1), 233: (232, 0), 234: (233, 0), 235: (234, 0), 236: (235, 0), 237: (236, 0), 238: (237, 0), 239: (238, 0), 240: (240, 0), 241: (241, 0), 242: (251, 1), 243: (252, 1), 244: (253, 1), 245: (262, 0), 246: (264, 0), 247: (265, 0), 248: (266, 0), 249: (267, 0), 250: (276, 1), 251: (277, 1), 252: (278, 1), 253: (279, 1), 254: (280, 1), 255: (282, 0), 256: (284, 0), 257: (285, 0), 258: (286, 0), 259: (287, 0), 260: (288, 0), 261: (293, 0), 262: (294, 0), 263: (295, 0), 264: (302, 0), 265: (303, 0), 266: (304, 0), 267: (305, 0), 268: (306, 0), 269: (313, 0), 270: (314, 0), 271: (315, 0), 272: (316, 0), 273: (318, 0), 274: (319, 0), 275: (320, 0), 276: (325, 1), 277: (326, 1), 278: (327, 1), 279: (328, 1), 280: (329, 1), 281: (330, 1), 282: (348, 0), 283: (350, 0), 284: (351, 0), 285: (352, 0), 286: (357, 0), 287: (358, 0), 288: (360, 1), 289: (361, 1), 290: (368, 0), 291: (369, 0), 292: (371, 1), 293: (386, 0), 294: (387, 0), 295: (388, 0), 296: (393, 0), 297: (394, 0), 298: (403, 0), 299: (404, 0), 300: (405, 0), 301: (406, 0), 302: (407, 0), 303: (408, 0), 304: (409, 1), 305: (410, 1), 306: (411, 1), 307: (412, 1), 308: (419, 0), 309: (420, 0), 310: (423, 0), 311: (424, 0), 312: (425, 0), 313: (426, 0), 314: (427, 0), 315: (432, 0), 316: (433, 0), 317: (434, 0), 318: (435, 0), 319: (4, 1), 320: (5, 1), 321: (6, 1), 322: (7, 1), 323: (8, 1), 324: (12, 0), 325: (13, 0), 326: (15, 0), 327: (20, 1), 328: (21, 1), 329: (22, 0), 330: (23, 0), 331: (24, 0), 332: (25, 0), 333: (31, 0), 334: (32, 0), 335: (34, 1), 336: (35, 1), 337: (36, 1), 338: (37, 1), 339: (38, 1), 340: (39, 1), 341: (44, 0), 342: (48, 0), 343: (51, 0), 344: (52, 0), 345: (53, 0), 346: (55, 0), 347: (63, 1), 348: (64, 1), 349: (65, 1), 350: (66, 1), 351: (73, 1), 352: (74, 1), 353: (75, 0), 354: (79, 0), 355: (80, 0), 356: (81, 0), 357: (83, 0), 358: (84, 0), 359: (85, 0), 360: (91, 1), 361: (92, 0), 362: (93, 1), 363: (94, 1), 364: (95, 1), 365: (96, 0), 366: (97, 1), 367: (98, 1), 368: (99, 1), 369: (100, 1), 370: (101, 1), 371: (102, 1), 372: (103, 1), 373: (108, 0), 374: (109, 0), 375: (110, 0), 376: (111, 0), 377: (112, 0), 378: (113, 0), 379: (114, 1), 380: (115, 1), 381: (116, 1), 382: (121, 0), 383: (122, 0), 384: (129, 0), 385: (130, 0), 386: (134, 0), 387: (135, 0), 388: (136, 0), 389: (141, 0), 390: (142, 0), 391: (163, 0), 392: (164, 0), 393: (165, 0), 394: (166, 0), 395: (167, 0), 396: (168, 0), 397: (172, 0), 398: (173, 0), 399: (174, 0), 400: (175, 0), 401: (176, 0), 402: (194, 0), 403: (198, 0), 404: (199, 0), 405: (200, 0), 406: (213, 0), 407: (214, 1), 408: (215, 1), 409: (216, 1), 410: (217, 1), 411: (218, 0), 412: (219, 0), 413: (220, 0), 414: (242, 0), 415: (248, 1), 416: (249, 1), 417: (254, 0), 418: (258, 0), 419: (6, 0), 420: (8, 0), 421: (9, 0), 422: (10, 0), 423: (17, 1), 424: (18, 1), 425: (19, 1), 426: (20, 1), 427: (21, 1), 428: (22, 1), 429: (23, 1), 430: (24, 1), 431: (28, 1), 432: (29, 1), 433: (32, 1), 434: (33, 1), 435: (34, 1), 436: (35, 1), 437: (36, 1), 438: (37, 1), 439: (38, 1), 440: (43, 1), 441: (44, 1), 442: (45, 1), 443: (46, 1), 444: (51, 0), 445: (52, 0), 446: (53, 0), 447: (54, 0), 448: (58, 1), 449: (59, 1), 450: (60, 1), 451: (64, 1), 452: (65, 1), 453: (66, 1), 454: (70, 1), 455: (71, 1), 456: (72, 1), 457: (73, 1), 458: (74, 1), 459: (75, 1), 460: (76, 1), 461: (77, 1), 462: (78, 1), 463: (83, 0), 464: (84, 0), 465: (85, 0), 466: (86, 0), 467: (90, 0), 468: (91, 0), 469: (93, 1), 470: (94, 1), 471: (95, 0), 472: (96, 0), 473: (101, 0), 474: (102, 0), 475: (103, 0), 476: (104, 0), 477: (109, 0), 478: (110, 0), 479: (111, 0), 480: (113, 0), 481: (114, 0), 482: (115, 0), 483: (121, 1), 484: (122, 1), 485: (123, 1), 486: (124, 1), 487: (129, 1), 488: (136, 0), 489: (137, 0), 490: (138, 0), 491: (140, 0), 492: (141, 0), 493: (145, 0), 494: (149, 0), 495: (155, 1), 496: (157, 1), 497: (163, 1), 498: (164, 0), 499: (165, 0), 500: (166, 0), 501: (170, 0), 502: (171, 0), 503: (175, 0), 504: (176, 0), 505: (177, 0), 506: (188, 0), 507: (189, 0), 508: (190, 1), 509: (191, 1), 510: (192, 1), 511: (196, 1), 512: (197, 1), 513: (198, 1), 514: (199, 1), 515: (200, 1), 516: (202, 1), 517: (203, 1), 518: (205, 1), 519: (206, 1), 520: (210, 0), 521: (212, 0), 522: (213, 1), 523: (217, 0), 524: (218, 0), 525: (222, 0), 526: (225, 1), 527: (226, 1), 528: (227, 1), 529: (228, 0), 530: (232, 1), 531: (233, 1), 532: (237, 0), 533: (240, 0), 534: (241, 0), 535: (242, 0), 536: (249, 1), 537: (250, 1), 538: (251, 1)}
# ROW_RANGES = {0: (0, 127), 1: (127, 319), 2: (319, 419), 3: (419, 539)}
# → e.g. [431, 556, 305, 31]  (check yours)



# ------------------------------------------------------------
#  EM‑related helpers
# ------------------------------------------------------------

def run_em(hint_tensor: np.ndarray) -> np.ndarray:
    """Run Dawid–Skene and return hard predictions (0=DIS, 1=AG)."""
    return dawid_skene(hint_tensor)


def determine_label(hint_type: int, em_vote: int) -> int:
    """Translate *agree/disagree* vote back to final relevance label."""
    if hint_type == 0:  # original hint said NOT_RELEVANT
        return 0 if em_vote == 1 else 1
    else:               # original hint said MISSED (i.e. relevant)
        return 1 if em_vote == 1 else 0


def integrate_predictions(
    raw_preds:   np.ndarray,
    task_map:    dict,
    vanilla:     np.ndarray,
    row_ranges:  dict = ROW_RANGES,
) -> np.ndarray:
    """
    Build a 1 323-long prediction vector that starts as 'vanilla' and
    is overwritten wherever a hint response exists.

    ───────────────────────────────────────────────────────────────
    • TASK_MAP values hold a *local* line_no (within-transcript).
    • We must therefore add an *offset in lines*, **not** in tasks.
    """

    # 1) figure out how many lines each transcript has ------------
    t1, t2, t3, t4 = process_labels()         # every line’s gold label
    line_counts      = [len(t1), len(t2), len(t3), len(t4)]
    line_offsets     = [0]                                  # cumulative sums → [0, L1, L1+L2, …]
    for n in line_counts[:-1]:
        line_offsets.append(line_offsets[-1] + n)

    # 2) easy lookup: task_idx → transcript_id --------------------
    #    e.g. if ROW_RANGES = {0:(0,127), 1:(127,319) …}
    task2transcript = {}
    for tid, (start, end) in row_ranges.items():            # 4 iterations
        for tk in range(start, end):
            task2transcript[tk] = tid

    # 3) start with vanilla, then overwrite ----------------------
    final = list(vanilla)      
    # print("final", final)                             # don’t mutate caller’s array

    for task_idx, (local_line, hint_type) in task_map.items():

        if task_idx >= len(raw_preds):                      # safety guard
            continue

        tid          = task2transcript[task_idx]            # which transcript?
        global_line  = line_offsets[tid] + local_line       # convert to 0-based global index
        new_label    = determine_label(hint_type, int(raw_preds[task_idx]))
        final[global_line] = new_label
        # print(global_line)

    # print("len of preds", len(final))
    return np.asarray(final)

def evaluate(preds_by_line: np.ndarray):
    """
    • `preds_by_line`  is the 1323-long vector produced by your pipeline.
    • Gold labels come from  process_labels(required=True).
    • Valid (speaker) line positions come from  get_valids().
    """
    gts_all            = process_labels(required=True)          # (t1, t2, t3, t4)
    valid_sets         = get_valids()                           # (po, ay, sm, mc)
    # make sure both tuples are in the same order --------------*
    # If your transcript ordering differs, simply rearrange the
    # elements in `valid_sets` to match gts_all.
    # ----------------------------------------------------------
    
    # cumulative line offsets to slice preds_by_line ------------
    line_counts  = [len(t) for t in gts_all]
    offsets      = [0]
    for n in line_counts[:-1]:
        offsets.append(offsets[-1] + n)

    for tid, (gt, valid_idx, start) in enumerate(zip(gts_all, valid_sets, offsets)):
        end          = start + len(gt)
        pred_slice   = preds_by_line[start:end]

        gt_valid     = [gt[i]          for i in valid_idx]
        pred_valid   = [pred_slice[i]  for i in valid_idx]
        vanilla = [(load_vanilla_preds(concat = False)[tid])[i] for i in valid_idx]
    
        
        # print("label",gt_valid)
        # print("vanilla", vanilla)
        # print("preds", pred_valid)
        print(f"\n=== Transcript {tid+1} ===")
        print(" #valid lines  :", len(gt_valid))
        print(" macro-F1      :", macro_f1(gt_valid, pred_valid))
        print(" pos recall    :", positive_recall(gt_valid, pred_valid))
        print(" pos precision :", positive_precision(gt_valid, pred_valid))
        print(" neg recall    :", negative_recall(gt_valid, pred_valid))
        print(" neg precision :", negative_precision(gt_valid, pred_valid))
        print(" acc :", accuracy(gt_valid, pred_valid))
        print(" vmacro-F1      :", macro_f1(gt_valid, vanilla))
        print(" vpos recall    :", positive_recall(gt_valid, vanilla))
        print(" vpos precision :", positive_precision(gt_valid, vanilla))
        print(" vneg recall    :", negative_recall(gt_valid, vanilla))
        print(" vneg precision :", negative_precision(gt_valid, vanilla))
        print(" vacc :", accuracy(gt_valid, vanilla))


# ------------------------------------------------------------
#  Main entry point
# ------------------------------------------------------------

def main():
    # print("????")

    vanilla = load_vanilla_preds()
    # print(vanilla)
    hint_tensor = np.load("/Users/saminthachandrasiri/Annota/annota-v2-cloud/functions/py-functions/hint_tensor.npy", allow_pickle=True)

    raw_agree_disagree = run_em(hint_tensor)

    blended_by_line = integrate_predictions(raw_agree_disagree, TASK_MAP, vanilla)
    # print(blended_by_line)
    # # 4. convert back to task order for eval -----------------------------
    # task_level_preds = extract_task_level(blended_by_line, TASK_MAP)

    # 5. evaluate --------------------------------------------------------
    evaluate(blended_by_line)

    print()


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    main()

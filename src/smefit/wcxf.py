# Dictionary translating from the smefit basis to the Warsaw basis in the WCxf
import numpy as np

alpha_s = 0.118
gs = np.sqrt(4 * np.pi * alpha_s)

wcxf_translate = {
    "OtG": {"wc": ["uG_33"], "value": [gs]},
    "OWWW": {"wc": ["W"]},
    # CpqMi = c1 - c3, C3pq = c3
    # c1 = CpqMi + C3pq, c3 = C3pq
    # (CpqMi = 1, C3pq = 0) => (c1 = 1, c3=0)
    # (CpqMi = 0, C3pq = 1) => (c1 = 1, c3 =1)
    "O3pq": {"wc": ["phiq1_11", "phiq1_22", "phiq3_11", "phiq3_22"]},
    "OpqMi": {
        "wc": ["phiq1_11", "phiq1_22"],
    },
}

# This creates a dictionary to invert the translation 1:1
# from the Warsaw basis to the SMEFiT basis
inverse_wcxf_translate = {
    "OtG": {"wc": ["uG_33"], "value": [1.0 / gs]},
    "OWWW": {"wc": ["W"]},
    "O3pq": {"wc": ["phiq3_11"]},
    "OpqMi": {"wc": ["phiq1_11", "phiq3_11"], "value": [1.0, -1.0]},
}

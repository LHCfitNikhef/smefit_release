# -*- coding: utf-8 -*-
import json
import sys
import traceback
import warnings

import numpy as np

# initializes global variables
all_operators = [
    "OpBox",
    "OpD",
    "O3pQ3",
    "O3pQ3prime",
    "O3pl1",
    "O3pl2",
    "O3pl3",
    "O3pq",
    "O3pqprime",
    "OWWW",
    "Obp",
    "Ocp",
    "Oll1221",
    "OpB",
    "OpBox",
    "OpD",
    "OpG",
    "OpQM",
    "OpW",
    "OpWB",
    "Opdi",
    "Ope",
    "Opl1",
    "Opl2",
    "Opl3",
    "Opmu",
    "Opq1",
    "Opq3",
    "OpqMi",
    "Opta",
    "Opui",
    "Otap",
    "Otp",
    "Omup",
]
# EW Scheme
Gf = 0.0000116638
MZ = 91.1876
MW = 80.385
vev = 1 / (2**0.5 * Gf) ** 0.5
alpha_s = (1.21772) ** 2 / 4 / np.pi
# Masses
MH = 125
MT = 172.5
MB = 4.7
MC = 1.27
MTAU = 1.777
MMU = 0.105
# Derived EW quantities
cw = MW / MZ
sw = (1 - (MW / MZ) ** 2) ** 0.5
g = 2 * MW / vev

# Scales
LambdaNP2 = 1000**2
v2_over_L2 = vev**2 / LambdaNP2


def particle_charges(particle):
    match particle:
        case "l":
            T3 = -0.5
            Q = -1
        case "nu":
            T3 = 0.5
            Q = 0
        case "u":
            T3 = 0.5
            Q = 2 / 3
        case "d":
            T3 = -0.5
            Q = -1 / 3
        case _:
            raise (ValueError("Unknown particle"))
    return T3, Q


def Higgs_gg(mq):  # SM Resolved loop, collecting a mq outside to isolate the yukawa
    t = 4 * mq**2 / MH**2
    if t > 1:
        func = np.arcsin(1 / np.sqrt(t))
        return 3 / 2 * t / mq * (1 + (1 - t) * func**2)
    else:
        radical = np.sqrt(1 - t)
        func = np.log((1 + radical) / (1 - radical))
        ipi = complex(0, np.pi)
        return 3 / 2 * t / mq * (1 - 1 / 4 * (1 - t) * (func - ipi) ** 2)


# Rotating function
def new_post(SMEFT_posteriors, smeft_coefficients, weights):
    n_live = len(SMEFT_posteriors[all_operators[0]])
    new_posteriors = np.zeros(n_live)
    if set(smeft_coefficients) <= set(SMEFT_posteriors.keys()):
        for coeff, w in zip(smeft_coefficients, weights):
            new_posteriors += w * np.array(SMEFT_posteriors[coeff])
    else:
        stack = traceback.extract_stack()
        caller = stack[-3].name
        warnings.warn(
            f"Setting 0 posteriors due to absence of {set(smeft_coefficients)-set(SMEFT_posteriors.keys())} needed in {caller}"
        )
    return new_posteriors


# operator check


def check_labels(SMEFT_posteriors):
    missing_operators = set(all_operators) - set(SMEFT_posteriors.keys())
    if len(missing_operators) > 0:
        warnings.warn(
            f"Missing operators: {list(missing_operators)}, setting to 0 the effective couplings in which they are involved"
        )
        return 0
    else:
        return 1


# Wavefunctions
def delta_ZA(SMEFT_posteriors):
    return -2 * v2_over_L2 * new_post(SMEFT_posteriors, ["OpWB"], [sw * cw])


def delta_ZAZ(SMEFT_posteriors):
    return v2_over_L2 * new_post(SMEFT_posteriors, ["OpWB"], [sw**2 - cw**2])


def delta_ZZ(SMEFT_posteriors):
    return 2 * sw * cw * new_post(SMEFT_posteriors, ["OpWB"], [v2_over_L2])


# Higgs contact
def zeta_A(SMEFT_posteriors):
    return (
        2
        * v2_over_L2
        * new_post(
            SMEFT_posteriors, ["OpB", "OpWB", "OpW"], [cw**2, -cw * sw, sw**2]
        )
    )


def zeta_W(SMEFT_posteriors):
    return 2 * new_post(SMEFT_posteriors, ["OpW"], [v2_over_L2])


def zeta_Z(SMEFT_posteriors):
    return (
        2
        * v2_over_L2
        * new_post(
            SMEFT_posteriors, ["OpB", "OpWB", "OpW"], [sw**2, +cw * sw, cw**2]
        )
    )


def zeta_ZA(SMEFT_posteriors):
    return v2_over_L2 * new_post(
        SMEFT_posteriors,
        ["OpB", "OpWB", "OpW"],
        [-2 * sw * cw, sw**2 - cw**2, 2 * cw * sw],
    )


# input
def cH(SMEFT_posteriors):
    return new_post(SMEFT_posteriors, ["OpBox", "OpD"], [1, +0.25]) * (2 * v2_over_L2)


def cT(SMEFT_posteriors):
    return -1 / 2 * new_post(SMEFT_posteriors, ["OpD"], [v2_over_L2])


def delta_v(SMEFT_posteriors):
    return (
        1
        / 2
        * new_post(SMEFT_posteriors, ["O3pl1", "O3pl2", "Oll1221"], [1, 1, -1])
        * v2_over_L2
    )


def delta_g(SMEFT_posteriors):
    return -delta_v(SMEFT_posteriors)


def delta_gprime(SMEFT_posteriors):
    return (
        -delta_v(SMEFT_posteriors)
        - new_post(SMEFT_posteriors, ["OpWB"], [cw / sw]) * v2_over_L2
        + cT(SMEFT_posteriors) / 2 / sw**2
    )


def delta_e(SMEFT_posteriors):
    cT_ = cT(SMEFT_posteriors)
    return (
        -delta_v(SMEFT_posteriors)
        + cw**2 / sw**2 * cT_ / 2
        - new_post(SMEFT_posteriors, ["OpWB"], [cw / sw]) * v2_over_L2
    )


def eta_W(SMEFT_posteriors):
    return -1 / 2 * cH(SMEFT_posteriors) - delta_v(SMEFT_posteriors)


def eta_Z(SMEFT_posteriors):
    return (
        -0.5 * cH(SMEFT_posteriors) - delta_v(SMEFT_posteriors) - cT(SMEFT_posteriors)
    )


# L and R definition
def delta_gl(SMEFT_posteriors, smeft1, smeft3, Q=-1, T3=-1 / 2):
    dg = delta_g(SMEFT_posteriors)
    dgprime = delta_gprime(SMEFT_posteriors)
    d_ZZ = delta_ZZ(SMEFT_posteriors)
    d_ZAZ = delta_ZAZ(SMEFT_posteriors)
    return (
        1
        / (T3 - Q * sw**2)
        * (
            cw**2 * dg * (T3 + Q * sw**2)
            + sw**2 * dgprime * (T3 - Q - cw**2 * Q)
            + new_post(SMEFT_posteriors, [smeft1, smeft3], [-1 / 2, T3]) * v2_over_L2
            + Q * sw * cw * d_ZAZ
        )
        + d_ZZ / 2
    )


def delta_gr(SMEFT_posteriors, smeft, Qf=-1):
    dg = delta_g(SMEFT_posteriors)
    dgprime = delta_gprime(SMEFT_posteriors)
    d_ZZ = delta_ZZ(SMEFT_posteriors)
    d_ZAZ = delta_ZAZ(SMEFT_posteriors)
    return (
        -(cw**2) * dg
        + (1 + cw**2) * dgprime
        + 1 / 2 / Qf / sw**2 * new_post(SMEFT_posteriors, [smeft], [1]) * v2_over_L2
        + (d_ZZ / 2 - cw / sw * d_ZAZ)
    )


# gW definition
def delta_gW(SMEFT_posteriors, smeft):
    return delta_g(SMEFT_posteriors) + new_post(SMEFT_posteriors, [smeft], [v2_over_L2])


# WW and ZZ decay definitions
QQ = (
    3 * (-1 / 2 + sw**2) ** 2
    + 3 * (1 / 2) ** 2
    + 3 * (sw**2) ** 2
    + 2 * 3 * (1 / 2 - 2 / 3 * sw**2) ** 2
    + 3 * 3 * (-1 / 2 + 1 / 3 * sw**2) ** 2
    + 3 * 3 * (1 / 3 * sw**2) ** 2
    + 3 * 2 * (-2 / 3 * sw**2) ** 2
)
Qqe = (
    3 * (-1 / 2 + sw**2) * (-1)
    + 3 * (sw**2) * (-1)
    + 2 * 3 * (1 / 2 - 2 / 3.0 * sw**2) * 2 / 3
    + 3 * 3 * (-1 / 2 + 1 / 3 * sw**2) * (-1 / 3)
    + 3 * 3 * (1 / 3 * sw**2) * (-1 / 3)
    + 2 * 3 * (-2 / 3 * sw**2) * 2 / 3
)
Qr = 0.529  # sticking to peskin


def compute_CZ(SMEFT_posteriors):
    pref = v2_over_L2 / 2
    left_SMEFT = {
        "el": -pref * new_post(SMEFT_posteriors, ["Opl1", "O3pl1"], [1, 1]),
        "mul": -pref * new_post(SMEFT_posteriors, ["Opl2", "O3pl2"], [1, 1]),
        "taul": -pref * new_post(SMEFT_posteriors, ["Opl3", "O3pl3"], [1, 1]),
        "vel": -pref * new_post(SMEFT_posteriors, ["Opl1", "O3pl1"], [1, -1]),
        "vmul": -pref * new_post(SMEFT_posteriors, ["Opl2", "O3pl2"], [1, -1]),
        "vtaul": -pref * new_post(SMEFT_posteriors, ["Opl3", "O3pl3"], [1, -1]),
        "ul": -pref * new_post(SMEFT_posteriors, ["Opq1", "O3pqprime"], [1, -1]),
        "dl": -pref * new_post(SMEFT_posteriors, ["Opq1", "O3pqprime"], [1, 1]),
        "sl": -pref * new_post(SMEFT_posteriors, ["Opq1", "O3pqprime"], [1, 1]),
        "cl": -pref * new_post(SMEFT_posteriors, ["Opq1", "O3pqprime"], [1, -1]),
        "bl": -pref * new_post(SMEFT_posteriors, ["Opq3", "O3pQ3prime"], [1, 1]),
    }
    right_SMEFT = {
        "er": -pref * new_post(SMEFT_posteriors, ["Ope"], [1]),
        "mur": -pref * new_post(SMEFT_posteriors, ["Opmu"], [1]),
        "taur": -pref * new_post(SMEFT_posteriors, ["Opta"], [1]),
        "ur": -pref * new_post(SMEFT_posteriors, ["Opui"], [1]),
        "dr": -pref * new_post(SMEFT_posteriors, ["Opdi"], [1]),
        "sr": -pref * new_post(SMEFT_posteriors, ["Opdi"], [1]),
        "cr": -pref * new_post(SMEFT_posteriors, ["Opui"], [1]),
        "br": -pref * new_post(SMEFT_posteriors, ["Opdi"], [1]),
    }
    uplcharge = 1 / 2 - sw**2 * 2 / 3
    downlcharge = -1 / 2 + sw**2 * 1 / 3
    llcharge = -1 / 2 + sw**2
    uprcharge = -(sw**2) * 2 / 3
    downrcharge = +(sw**2) * 1 / 3
    lrcharge = +(sw**2)
    nucharge = 1 / 2
    CZ = (
        3 * uplcharge * (left_SMEFT["ul"] + left_SMEFT["cl"])
        + 3 * downlcharge * (left_SMEFT["dl"] + left_SMEFT["sl"] + left_SMEFT["bl"])
        + 3 * uprcharge * (right_SMEFT["ur"] + right_SMEFT["cr"])
        + 3 * downrcharge * (right_SMEFT["dr"] + right_SMEFT["sr"] + right_SMEFT["br"])
        + nucharge * (left_SMEFT["vel"] + left_SMEFT["vmul"] + left_SMEFT["vtaul"])
        + llcharge * (left_SMEFT["el"] + left_SMEFT["mul"] + left_SMEFT["taul"])
        + lrcharge * (right_SMEFT["er"] + right_SMEFT["mur"] + right_SMEFT["taur"])
    ) / (QQ)
    return CZ


def compute_CW(SMEFT_posteriors):
    return (
        v2_over_L2
        * new_post(
            SMEFT_posteriors, ["O3pl1", "O3pl2", "O3pl3", "O3pqprime"], [1, 1, 1, 6]
        )
        / (1 + 1 + 1 + 3 + 3)
    )


def deltaGammaW(SMEFT_posteriors):
    CW = compute_CW(SMEFT_posteriors)
    return 2 * delta_g(SMEFT_posteriors) + 2 * CW


def deltaGammaZ(SMEFT_posteriors):
    CZ = compute_CZ(SMEFT_posteriors)
    dg = delta_g(SMEFT_posteriors)
    dgprime = delta_gprime(SMEFT_posteriors)
    d_ZZ = delta_ZZ(SMEFT_posteriors)
    d_ZAZ = delta_ZAZ(SMEFT_posteriors)
    return (
        2 * cw**2 * (1 + 2 * Qr * sw**2) * dg
        + 2 * sw**2 * (1 - 2 * Qr * cw**2) * dgprime
        + d_ZZ
        + Qr * sw * cw * d_ZAZ
        + 2 * CZ
    )


# effective coupling definitions
# Higgs loop
def delta_gHZZ(SMEFT_posteriors):
    dv = delta_v(SMEFT_posteriors)
    eta_Z_ = eta_Z(SMEFT_posteriors)
    zeta_Z_ = zeta_Z(SMEFT_posteriors)
    CZ = compute_CZ(SMEFT_posteriors)
    delta_gammaZ = deltaGammaZ(SMEFT_posteriors)
    return (
        1 / 2 * (2 * eta_Z_ - 2 * dv - 0.5 * zeta_Z_ - 1.02 * CZ + 1.18 * delta_gammaZ)
    )


def delta_gHZA(SMEFT_posteriors):
    dv = delta_v(SMEFT_posteriors)
    dg = delta_g(SMEFT_posteriors)
    dgprime = delta_gprime(SMEFT_posteriors)
    d_ZZ = delta_ZZ(SMEFT_posteriors)
    d_ZA = delta_ZA(SMEFT_posteriors)
    zeta_ZA_ = zeta_ZA(SMEFT_posteriors)
    cH_ = cH(SMEFT_posteriors)
    return (
        1
        / 2
        * (
            290 * zeta_ZA_
            - cH_
            - 2 * (1 - 3 * sw**2) * dg
            + 6 * cw**2 * dgprime
            + d_ZA
            + d_ZZ
            - 2 * dv
        )
    )


def delta_gHAA(SMEFT_posteriors):
    dv = delta_v(SMEFT_posteriors)
    d_e = delta_e(SMEFT_posteriors)
    d_ZA = delta_ZA(SMEFT_posteriors)
    zeta_A_ = zeta_A(SMEFT_posteriors)
    cH_ = cH(SMEFT_posteriors)
    return 1 / 2 * (526 * zeta_A_ + 2 * d_ZA - cH_ + 4 * d_e - 2 * dv)


def delta_gHWW(SMEFT_posteriors):
    dv = delta_v(SMEFT_posteriors)
    eta_W_ = eta_W(SMEFT_posteriors)
    zeta_W_ = zeta_W(SMEFT_posteriors)
    CW = compute_CW(SMEFT_posteriors)
    delta_gammaW = deltaGammaW(SMEFT_posteriors)
    return (
        1 / 2 * (2 * eta_W_ - 2 * dv - 0.75 * zeta_W_ - 0.88 * CW + 1.06 * delta_gammaW)
    )


def delta_gHgg(SMEFT_posteriors):
    Higgs_tt = Higgs_gg(MT)
    Higgs_bb = Higgs_gg(MB)
    Higgs_cc = Higgs_gg(MC)
    SM_pref = alpha_s**2 * MH**3 / (72 * np.pi**3)
    Higgsgg_SM_full = 0.0002092
    cH_ = cH(SMEFT_posteriors)
    dv = delta_v(SMEFT_posteriors)
    return (
        -cH_ / 2
        - dv
        - SM_pref
        * v2_over_L2
        * np.real(
            (MT / vev * Higgs_tt + MB / vev * Higgs_bb + MC / vev * Higgs_cc)
            * (
                (
                    new_post(SMEFT_posteriors, ["Otp"], [1 / np.sqrt(2)]) * Higgs_tt
                    + new_post(SMEFT_posteriors, ["Obp"], [1 / np.sqrt(2)]) * Higgs_bb
                    + new_post(SMEFT_posteriors, ["Ocp"], [1 / np.sqrt(2)]) * Higgs_cc
                )
                - (3 * 4 * np.pi / alpha_s / vev)
                * new_post(SMEFT_posteriors, ["OpG"], [1.0])
            )
        )
        / Higgsgg_SM_full
    )


# TGC
def delta_lambdaZ(SMEFT_posteriors):
    return new_post(SMEFT_posteriors, ["OWWW"], [3 / 2 * g]) * v2_over_L2


def delta_gz(SMEFT_posteriors):
    dg = delta_g(SMEFT_posteriors)
    dgprime = delta_gprime(SMEFT_posteriors)
    d_ZZ = delta_ZZ(SMEFT_posteriors)
    d_ZAZ = delta_ZAZ(SMEFT_posteriors)
    return dg * (1 + sw**2) - sw**2 * dgprime + 1 / 2 * d_ZZ + sw / cw * d_ZAZ


def delta_kgamma(SMEFT_posteriors):
    d_e = delta_e(SMEFT_posteriors)
    return d_e + cw / sw * new_post(SMEFT_posteriors, ["OpWB"], [v2_over_L2])


# gL coupling


def delta_geL(SMEFT_posteriors):
    return delta_gl(SMEFT_posteriors, "Opl1", "O3pl1")


def delta_gmuL(SMEFT_posteriors):
    return delta_gl(SMEFT_posteriors, "Opl2", "O3pl2")


def delta_gtaL(SMEFT_posteriors):
    return delta_gl(SMEFT_posteriors, "Opl3", "O3pl3")


def delta_guL(
    SMEFT_posteriors,
):
    return delta_gl(SMEFT_posteriors, "Opq1", "O3pqprime", Q=+2 / 3, T3=1 / 2)


def delta_gdL(SMEFT_posteriors):
    return delta_gl(SMEFT_posteriors, "Opq1", "O3pqprime", Q=-1 / 3, T3=-1 / 2)


# gR couplings


def delta_geR(SMEFT_posteriors):
    return delta_gr(SMEFT_posteriors, "Ope")


def delta_gmuR(SMEFT_posteriors):
    return delta_gr(SMEFT_posteriors, "Opmu")


def delta_gtaR(SMEFT_posteriors):
    return delta_gr(SMEFT_posteriors, "Opta")


def delta_guR(SMEFT_posteriors):
    return delta_gr(SMEFT_posteriors, "Opui", Qf=2 / 3)


def delta_gdR(SMEFT_posteriors):
    return delta_gr(SMEFT_posteriors, "Opdi", Qf=-1 / 3)


# W couplings
def delta_gWe(
    SMEFT_posteriors,
):
    return delta_gW(SMEFT_posteriors, "O3pl1")


def delta_gWmu(SMEFT_posteriors):
    return delta_gW(SMEFT_posteriors, "O3pl2")


def delta_gWta(SMEFT_posteriors):
    return delta_gW(SMEFT_posteriors, "O3pl3")


# Yukawas
def delta_gHff(SMEFT_posteriors, smeft, mf):
    cH_ = cH(SMEFT_posteriors)
    dv = delta_v(SMEFT_posteriors)
    return 0.5 * (
        -cH_
        - new_post(SMEFT_posteriors, [smeft], [2])
        * v2_over_L2
        / (np.sqrt(2) * mf / vev)
        - 2 * dv
    )


def delta_gHmumu(SMEFT_posteriors):
    return delta_gHff(SMEFT_posteriors, "Omup", MMU)


def delta_gHtata(SMEFT_posteriors):
    return delta_gHff(SMEFT_posteriors, "Otap", MTAU)


def delta_gHcc(SMEFT_posteriors):
    return delta_gHff(SMEFT_posteriors, "Ocp", MC)


def delta_gHbb(SMEFT_posteriors):
    return delta_gHff(SMEFT_posteriors, "Obp", MB)


def delta_gHtt(SMEFT_posteriors):
    return delta_gHff(SMEFT_posteriors, "Otp", MT)


def build_eff_coupling_dictionary(SMEFT_posteriors):
    eff_coupling_dictionary = {}
    eff_coupling_dictionary["geL"] = list(delta_geL(SMEFT_posteriors))
    eff_coupling_dictionary["gmuL"] = list(delta_gmuL(SMEFT_posteriors))
    eff_coupling_dictionary["gtaL"] = list(delta_gtaL(SMEFT_posteriors))
    eff_coupling_dictionary["guL"] = list(delta_guL(SMEFT_posteriors))
    eff_coupling_dictionary["gdL"] = list(delta_gdL(SMEFT_posteriors))
    eff_coupling_dictionary["geR"] = list(delta_geR(SMEFT_posteriors))
    eff_coupling_dictionary["gmuR"] = list(delta_gmuR(SMEFT_posteriors))
    eff_coupling_dictionary["gtaR"] = list(delta_gtaR(SMEFT_posteriors))
    eff_coupling_dictionary["guR"] = list(delta_guR(SMEFT_posteriors))
    eff_coupling_dictionary["gdR"] = list(delta_gdR(SMEFT_posteriors))
    eff_coupling_dictionary["gWe"] = list(delta_gWe(SMEFT_posteriors))
    eff_coupling_dictionary["gWmu"] = list(delta_gWmu(SMEFT_posteriors))
    eff_coupling_dictionary["gWta"] = list(delta_gWta(SMEFT_posteriors))
    eff_coupling_dictionary["gHZZ"] = list(delta_gHZZ(SMEFT_posteriors))
    eff_coupling_dictionary["gHWW"] = list(delta_gHWW(SMEFT_posteriors))
    eff_coupling_dictionary["gHAA"] = list(delta_gHAA(SMEFT_posteriors))
    eff_coupling_dictionary["gHZA"] = list(delta_gHZA(SMEFT_posteriors))
    eff_coupling_dictionary["gHgg"] = list(delta_gHgg(SMEFT_posteriors))
    eff_coupling_dictionary["gHmumu"] = list(delta_gHmumu(SMEFT_posteriors))
    eff_coupling_dictionary["gHtata"] = list(delta_gHtata(SMEFT_posteriors))
    eff_coupling_dictionary["gHcc"] = list(delta_gHcc(SMEFT_posteriors))
    eff_coupling_dictionary["gHbb"] = list(delta_gHbb(SMEFT_posteriors))
    eff_coupling_dictionary["gHtt"] = list(delta_gHtt(SMEFT_posteriors))
    eff_coupling_dictionary["lambdaZ"] = list(delta_lambdaZ(SMEFT_posteriors))
    eff_coupling_dictionary["g1Z"] = list(delta_gz(SMEFT_posteriors))
    eff_coupling_dictionary["kgamma"] = list(delta_kgamma(SMEFT_posteriors))
    return eff_coupling_dictionary


def main():
    # initialize reading PATH
    if len(sys.argv) == 1:
        PATH = "./"
    else:
        PATH = sys.argv[1]
    with open(PATH + "/fit_results.json", "r", encoding="utf8") as r:
        SMEFT_posteriors = json.load(r)["samples"]
    check_labels(SMEFT_posteriors)
    new_dict = build_eff_coupling_dictionary(SMEFT_posteriors)
    with open(PATH + "/fit_eff_coupling.json", "w", encoding="utf8") as w:
        json.dump(new_dict, w)
    return 1


if __name__ == "__main__":
    main()

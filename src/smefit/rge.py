import wilson
from smefit.wcxf import wcxf_translate, inverse_wcxf_translate
import pandas as pd


class RGE:
    def __init__(self, wc_names, init_scale, accuracy="integrate"):
        # order the Wilson coefficients alphabetically
        self.wc_names = sorted(wc_names)
        self.init_scale = init_scale
        self.accuracy = accuracy

    def RGEmatrix_dict(self, scale):
        # compute the RGE matrix at the scale `scale`
        rge_matrix_dict = {}
        for wc_name, wc_vals in self.RGEbasis.items():
            print(f"Computing RGE for {wc_name}")
            wc_init = wilson.Wilson(
                wc_vals, scale=self.init_scale, eft="SMEFT", basis="Warsaw"
            )
            wc_init.set_option("smeft_accuracy", self.accuracy)

            wc_final = wc_init.match_run(scale=scale, eft="SMEFT", basis="Warsaw")

            # Remove small values
            wc_final_vals = {
                key: value for key, value in wc_final.dict.items() if abs(value) > 1e-10
            }

            # check that imaginary values are small
            if any(abs(val.imag) > 1e-10 for val in wc_final_vals.values()):
                raise ValueError(
                    f"Imaginary values in Wilson coefficient for operator {wc_name}."
                )

            rge_matrix_dict[wc_name] = self.map_to_smefit(wc_final_vals)

        return rge_matrix_dict

    def RGEmatrix(self, scale):
        # compute the RGE matrix dict at the scale `scale`
        rge_matrix_dict = self.RGEmatrix_dict(scale)

        # create the RGE matrix as pandas dataframe
        rge_matrix = pd.DataFrame(
            columns=self.wc_names, index=self.all_ops, dtype=float
        )

        for wc_name, wc_dict in rge_matrix_dict.items():
            for op in self.all_ops:
                rge_matrix.loc[op, wc_name] = wc_dict.get(op, 0.0)
        # if there are rows with all zeros, remove them
        rge_matrix = rge_matrix.loc[(rge_matrix != 0).any(axis=1)]

        return rge_matrix

    @property
    def RGEbasis(self):
        # computes the translation from the smefit basis to the Warsaw basis
        # as expected by the Wilson package
        wc_basis = {}
        for wc_name in self.wc_names:
            try:
                wcxf_dict = wcxf_translate[wc_name]
            except KeyError:
                raise ValueError(
                    f"Wilson coefficient {wc_name} not present in the WCxf translation dictionary"
                )
            wc_warsaw_name = wcxf_dict["wc"]
            if "value" not in wcxf_dict:
                wc_warsaw_value = [1] * len(wcxf_dict["wc"])
            else:
                wc_warsaw_value = wcxf_dict["value"]

            # 1e-6 is because the Warsaw basis is in GeV^-2
            wc_value = {
                wc: val * 1e-6 for wc, val in zip(wc_warsaw_name, wc_warsaw_value)
            }
            wc_basis[wc_name] = wc_value

        return wc_basis

    def map_to_smefit(self, wc_final_vals):
        # TODO: missing a check that flavour structure is the one expected
        wc_dict = {}
        for wc_basis, wc_inv_dict in inverse_wcxf_translate.items():
            wc_warsaw_name = wc_inv_dict["wc"]
            if "value" not in wc_inv_dict:
                wc_warsaw_value = [1] * len(wc_warsaw_name)
            else:
                wc_warsaw_value = wc_inv_dict["value"]

            value = 0.0
            for wc, val in zip(wc_warsaw_name, wc_warsaw_value):
                if wc in wc_final_vals:
                    # 1e6 is to transform from GeV^-2 to TeV^2
                    value += 1e6 * wc_final_vals[wc].real * val
            wc_dict[wc_basis] = value
        return wc_dict

    @property
    def all_ops(self):
        return sorted(wcxf_translate.keys())

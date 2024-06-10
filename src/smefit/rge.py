import wilson
from smefit.wcxf import wcxf_translate, inverse_wcxf_translate
import pandas as pd


class RGE:
    def __init__(self, wc_names, init_scale):
        # order the Wilson coefficients alphabetically
        self.wc_names = sorted(wc_names)
        self.init_scale = init_scale

    def RGEmatrix(self, scale):
        # compute the RGE matrix at the scale `scale`
        rge_matrix_dict = {}
        for wc_name, wc_val in self.RGEbasis.items():
            print(f"Computing RGE for {wc_name}")
            wc_init = wilson.Wilson(
                wc_val, scale=self.init_scale, eft="SMEFT", basis="Warsaw"
            )

            wc_final = wc_init.match_run(scale=scale, eft="SMEFT", basis="Warsaw")

            # Assuming wc_final is a dictionary
            wc_final_vals = {
                key: value for key, value in wc_final.dict.items() if abs(value) > 1e-10
            }

            # check that imaginary values are small
            if any(abs(val.imag) > 1e-10 for val in wc_final_vals.values()):
                raise ValueError(
                    f"Imaginary values in Wilson coefficient for operator {wc_name}."
                )

            rge_matrix_dict[wc_name] = self.from_wcxf_to_smefit(wc_final_vals)

        # determine the union of the keys in rge_matrix_dict
        all_ops = set()
        for wc_dict in rge_matrix_dict.values():
            all_ops = all_ops.union(wc_dict.keys())
        # convert to a list, order alphabetically
        self.all_ops = sorted(list(all_ops))

        # create the RGE matrix as pandas dataframe
        rge_matrix = pd.DataFrame(
            columns=self.wc_names, index=self.all_ops, dtype=float
        )

        for wc_name, wc_dict in rge_matrix_dict.items():
            for op in self.all_ops:
                rge_matrix.loc[op, wc_name] = wc_dict.get(op, 0.0)

        return rge_matrix

    @property
    def RGEbasis(self):
        # compute the basis in the Warsaw basis expected by wilson
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

    def from_wcxf_to_smefit(self, wc_final_vals):
        # TODO: missing a check that flavour structure is the one expected
        wc_final_dict = {}
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
            wc_final_dict[wc_basis] = value
        return wc_final_dict

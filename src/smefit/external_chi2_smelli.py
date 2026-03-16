from smefit.rge import RGE


class smelli_chi2:
    # coefficients are mandatory, taken from smefit runcard. rge_dict is the rge dictionary in the runcard,
    # in this case we will only need it to read the initial scale, but we also need to build the RGE object.
    # everything else is optional and can be specified in the external_chi2 runcard
    def __init__(self, coefficients, rge_dict=None, dataset=[]):
        # this gives you the scale
        self.scale = rge_dict["init_scale"]

        # this gives you the names of the coeff in smefit
        self.names = coefficients.name

        # construct the RGE object. We don't actually need the RGE from smefit, but
        # it has inside of it a method to translate coefficients from smefit to wcxf.
        # Maybe not super ideal that it's inside that but ok for now.
        self.rge_obj = RGE(
            self.names,
            self.scale,
            rge_dict["smeft_accuracy"],
            rge_dict["adm_QCD"],
            rge_dict["yukawa"],
            # the last three are not actually needed
        )

        # This is a dictionary of dictionaries, associating to each of the smefit wc names, the wc_dict for smelli
        self.translation_basis = self.rge_obj.RGEbasis

    def compute_chi2(self, coefficient_values):
        # this receives an array of numbers, which order corresponds
        # to self.names

        # you need to take those values and construct the wc_dict for smelli
        # you have already the basis, which is telling you stuff like
        # self.translation_basis["OpD"] -> {"phiD": 1e-6}
        # so you need to take them all, multiply their values for the coefficien_values
        # then create a single dictionary (possible repeated keys should be summed)
        # something along these lines (adapted to the situation, this is copied from rge.RGEevolve)
        wc_dict = {}
        for op, values in self.RGEbasis.items():
            for key in values:
                if key not in wc_wilson:
                    wc_dict[key] = values[key] * wcs[op]
                else:
                    wc_dict[key] += values[key] * wcs[op]

        # now you should have the self.scale and the wc_dict needed for the smelli likelihood.
        # make sure you are returning a "chi2"
        return -2 smelli_loglikelihood

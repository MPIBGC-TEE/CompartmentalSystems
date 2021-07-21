# vim: set ff=unix expandtab ts=4 sw=4:

from copy import deepcopy
from .TsTpMassFields import TsTpMassFieldsPerPool


class CompatibleTsTpMassFieldsPerPool(TsTpMassFieldsPerPool):
    # This class is a list of mass Distributions with an entry for every
    # pool
    def __init__(self, normal_list):
        # make sure it has at least one entry
        if not (len(normal_list)) > 0:
            raise (Exception("There has to be at least one pool"))
        # check if all the Fields have the same tss
        for i, el in enumerate(self):
            if not (el.tss == self[0].tss):
                raise (
                    Exception(
                        Template(
                            "Element number ${i} had tts=${etss} while the first element of the list had tss=${first_tss}"
                        ).substitute(i=i, etss=el.tss, first_tss=self[0].tss)
                    )
                )
        # now check the sizes of the entries and adapt them as neccessary
        # to accomodate all possible transfers:
        # all pools must be able to receive  Material from any other pool
        # The maximum system Age for every pool is the maximum System age
        # of all pools
        # print("#############################")
        # print("self:=",normal_list)
        overall_number_of_Ts_entries = max(
            [field.number_of_Ts_entries for field in normal_list]
        )
        for el in normal_list:
            el.resize(overall_number_of_Ts_entries)
        # create a the dict like structure
        super().__init__({k: val for k, val in enumerate(normal_list)})

    def advanced(
        self, external_inputs, internal_death_rate_fields, outward_death_rate_fields
    ):

        # Note that the fields in res will become one tss bigger in
        # Ts size although we start with a copy..
        res = deepcopy(self)
        ol = res.external_losses(outward_death_rate_fields)
        res.remove(ol)

        il = self.internal_losses(internal_death_rate_fields)
        res.remove(il)
        gains = il.gains
        res.receive(gains)
        res.shift()  # move forward in time which increases size of the fields
        res.receive_external(external_inputs)
        return res

    @property
    def number_of_pools(self):
        return len(self)

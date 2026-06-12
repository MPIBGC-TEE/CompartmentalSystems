# vim:set ff=unix expandtab ts=4 sw=4
from copy import deepcopy
import numpy as np


class TimeStep:
    def __init__(
        self,
        time,
        rectangles,
        internal_death_rate_fields,
        external_death_rate_fields,
        external_input_numbers,
    ):
        self.time = time
        self.rectangles = rectangles
        self.internal_death_rate_fields = internal_death_rate_fields
        self.external_death_rate_fields = external_death_rate_fields
        self.external_input_numbers = external_input_numbers

    @property
    def updated_content(self):
        res = deepcopy(self.rectangles)
        external_losses = res.external_losses(self.external_death_rate_fields)
        internal_losses = res.internal_losses(self.internal_death_rate_fields)
        res.remove(external_losses)
        res.remove(internal_losses)
        res.shift()  # move forward in time which increases size of the fields
        # res.incorporate_gains(internal_losses,external_input_numbers)
        gains = internal_losses.gains
        res.receive(gains)
        res.receive_external(self.external_input_numbers)
        return res

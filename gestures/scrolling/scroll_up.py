"""ScrollUp: each fist → fully extended (pointing up) cycle pages up once."""


class ScrollUp:
    """Fires once when palm-reach grows from a fist to a fully raised open hand.

    After a page, the hand must return to a fist before the next page.
    """

    def __init__(self, fist_reach: float = 1.20, expanded_reach: float = 1.90):
        self.fist_reach = fist_reach
        self.expanded_reach = expanded_reach
        self.from_fist = False

    def reset(self) -> None:
        self.from_fist = False

    def update(self, *, in_line: bool, in_fist: bool, reach: float, pointing_up: bool) -> bool:
        if not in_line:
            return False

        if in_fist and reach <= self.fist_reach:
            self.from_fist = True
            return False

        if self.from_fist and pointing_up and reach >= self.expanded_reach:
            self.from_fist = False
            return True

        return False

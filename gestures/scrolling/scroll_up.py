"""ScrollUp: after a fist start, page-scroll up at the same rate the fingers rise."""


class ScrollUp:
    """Maps a slow fist-to-open raise onto a matching slow page-up.

    Starts only from a fist. While the fingertips stay touching, each increase
    in palm-reach scrolls the Mac up by a proportional amount. When the hand
    is fully open, scrolling stops. Separating the fingers cancels the raise;
    another fist is required to start again.
    """

    def __init__(
        self,
        fist_reach: float = 1.20,
        expanded_reach: float = 1.90,
        page_lines: float = 48.0,
    ):
        self.fist_reach = fist_reach
        self.expanded_reach = expanded_reach
        self.page_lines = page_lines
        self.from_fist = False
        self.finished = False
        self.last_reach = 0.0
        self.accrued = 0.0

    def reset(self) -> None:
        self.from_fist = False
        self.finished = False
        self.last_reach = 0.0
        self.accrued = 0.0

    def arm(self, reach: float) -> None:
        """Call when a qualifying fist is seen (mode just armed, or fist again)."""
        self.from_fist = True
        self.finished = False
        self.last_reach = min(reach, self.fist_reach)
        self.accrued = 0.0

    def update(self, *, touching: bool, in_fist: bool, reach: float) -> int:
        """Return how many lines to scroll up this frame (0 if none)."""
        if in_fist and touching and reach <= self.fist_reach:
            self.arm(reach)
            return 0

        if not self.from_fist or self.finished:
            return 0

        if not touching:
            self.reset()
            return 0

        span = max(self.expanded_reach - self.fist_reach, 1e-6)
        ceiling = self.expanded_reach
        if reach > self.last_reach:
            usable = min(reach, ceiling) - self.last_reach
            if usable > 0:
                self.accrued += (usable / span) * self.page_lines
            self.last_reach = min(reach, ceiling)

        lines = int(self.accrued)
        self.accrued -= lines

        if reach >= self.expanded_reach:
            self.finished = True
            self.from_fist = False
            leftover = self.accrued
            self.accrued = 0.0
            if leftover >= 0.5:
                lines += 1

        return max(0, lines)

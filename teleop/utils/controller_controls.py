"""Edge-triggered PICO controller commands for teleoperation lifecycle control."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ControllerButtons:
    """Normalized PICO face and grip button state for one XR sample."""

    x: bool = False
    y: bool = False
    a: bool = False
    b: bool = False
    left_grip: bool = False
    right_grip: bool = False

    @classmethod
    def from_tele_data(cls, tele_data):
        # Vuer names the left-controller X/Y buttons aButton/bButton.
        return cls(
            x=bool(tele_data.left_ctrl_aButton),
            y=bool(tele_data.left_ctrl_bButton),
            a=bool(tele_data.right_ctrl_aButton),
            b=bool(tele_data.right_ctrl_bButton),
            left_grip=bool(tele_data.left_ctrl_squeeze),
            right_grip=bool(tele_data.right_ctrl_squeeze),
        )


@dataclass(frozen=True)
class ControllerActions:
    arm_commands: tuple
    record_toggle_requested: bool
    quit_requested: bool
    left_grip_lock_toggled: bool
    right_grip_lock_toggled: bool
    left_grip_locked: bool
    right_grip_locked: bool


class ControllerControlMapper:
    """Convert held buttons into debounced lifecycle actions and grip locks.

    Single face buttons commit on release. Each overlapping pair is consumed
    until both buttons are up, including staggered releases.
    """

    def __init__(self):
        self.left_grip_locked = False
        self.right_grip_locked = False
        self._previous = ControllerButtons()
        self._consumed = set()

    def _pair(self, buttons, left, right):
        pair = (left, right)
        down = [name for name in pair if getattr(buttons, name)]
        if pair in self._consumed:
            if not down:
                self._consumed.remove(pair)
            return None
        if len(down) == 2:
            self._consumed.add(pair)
            return left + right
        for name in pair:
            if getattr(self._previous, name) and name not in down:
                return name
        return None

    def update(self, buttons: ControllerButtons) -> ControllerActions:
        left_grip_toggled = buttons.left_grip and not self._previous.left_grip
        right_grip_toggled = buttons.right_grip and not self._previous.right_grip
        if left_grip_toggled:
            self.left_grip_locked = not self.left_grip_locked
        if right_grip_toggled:
            self.right_grip_locked = not self.right_grip_locked

        reset = self._pair(buttons, "x", "a")
        take = self._pair(buttons, "y", "b")
        quit_requested = reset == "xa"
        record_toggle_requested = take == "yb" and not quit_requested
        commands = () if quit_requested or record_toggle_requested else tuple(
            key for key in (take, reset) if key is not None
        )
        self._previous = buttons
        return ControllerActions(
            arm_commands=commands,
            record_toggle_requested=record_toggle_requested,
            quit_requested=quit_requested,
            left_grip_lock_toggled=left_grip_toggled,
            right_grip_lock_toggled=right_grip_toggled,
            left_grip_locked=self.left_grip_locked,
            right_grip_locked=self.right_grip_locked,
        )

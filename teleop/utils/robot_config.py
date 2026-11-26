"""Robot model names shared by teleoperation entry points."""

ARM_ARGUMENT_CHOICES = ("G1", "G1_29", "G1_23", "H2", "R1")
SUPPORTED_ARM = "G1_29"


def normalize_arm_name(arm_name: str) -> str:
    """Return the canonical model name used by the implemented controller."""
    if arm_name == "G1":
        return SUPPORTED_ARM
    return arm_name


def unsupported_arm_message(arm_name: str) -> str:
    if arm_name == "G1_23":
        return (
            "G1_23 is not supported by this teleoperation entry point; its joint "
            "layout must not be controlled with the G1_29 implementation"
        )
    return f"{arm_name} arm is not supported yet"

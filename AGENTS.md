# Repository instructions

- Keep command-line argument parsing, workflow sequencing, task defaults, and entry-point-specific behavior with the entry point or in a helper used by those scripts.
- Keep shared modules such as `robot_ctrl`, `teleop/robot_control`, and transport, camera, and recording utilities independent of script entry points and flags. Pass hardware targets, standard data, or explicit callbacks from the script when it needs to configure a shared component.
- Put genuinely shared protocol adapters in a neutral module. Dependencies run from entry points to shared modules.

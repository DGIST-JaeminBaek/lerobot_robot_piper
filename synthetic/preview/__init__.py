"""Offline, hardware-free preview and validation for generated trajectories.

Ties together calibration (image<->board), `preprocessing/` (raw<->model
image), `transforms/` (board<->base), `trajectory/` (Cartesian segments),
and `kinematics/` (IK + normalized action) into one reviewable output
directory. Never opens CAN, RViz, or a live camera.
"""

"""Offline FK/IK and physical<->normalized action conversion for the Piper arm.

Every convention in this package (DH offset, FK units, calibration ranges,
motor ordering) is copied from already-validated project code
(`docs/kinematics/kinematics_check.md`,
`scripts/tools/piper_first_chunk_fk_analysis.py`,
`lerobot_robot_piper/motors/piper_motors_bus.py`,
`lerobot_robot_piper/piper_follower.py`) rather than re-derived or guessed.
"""

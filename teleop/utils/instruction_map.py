import numpy as np


class LowPassFilter:
    """Low-pass filter for smoothing data"""
    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self._value = 0.0
        self._last_value = 0.0

    def update(self, new_value, max_accel=1.5):
        delta = new_value - self._last_value
        delta = np.clip(delta, -max_accel, max_accel)
        filtered = self.alpha * (self._last_value + delta) + (1 - self.alpha) * self._value
        self._last_value = filtered
        self._value = filtered
        return self._value


class ControlDataMapper:
    """
    Control data mapper for mobile base and elevation
    """
    def __init__(self, current_waist_yaw=None):
        # Velocity filters
        self._filters = {
            'mobile_x_vel': LowPassFilter(alpha=0.15),
            'mobile_yaw_vel': LowPassFilter(alpha=0.15)
        }
        
        # Height accumulated value (remains unchanged after release)
        self.height_speed_value = 0
        self.mobile_x_vel = 0
        self.mobile_yaw_vel = 0
        self.current_waist_yaw_pos = current_waist_yaw if current_waist_yaw is not None else 0.0

    def update(self, lx=None, ly=None, rx=None, ry=None, current_waist_yaw=None):
        if lx is not None:
            # Map forward velocity 
            raw = self._map_forward_velocity(lx)
            mobile_x_vel = self._filters['mobile_x_vel'].update(raw, max_accel=1.0)
            self.mobile_x_vel = mobile_x_vel
        else:
            mobile_x_vel = self.mobile_x_vel

        if ly is not None:
            # Map lateral velocity
            raw = self._map_lateral_velocity(ly)
            mobile_yaw_vel = self._filters['mobile_yaw_vel'].update(raw, max_accel=1.0)
            self.mobile_yaw_vel = mobile_yaw_vel
        else:
            mobile_yaw_vel = self.mobile_yaw_vel

        # Update waist yaw position based on joystick input and current position
        if rx is not None and current_waist_yaw is not None:
            waist_yaw_pos = self._update_waist_position(rx, current_waist_yaw, max_velocity=0.05, min_position=-2.5, max_position=2.5)
        elif current_waist_yaw is not None:
            waist_yaw_pos = self.current_waist_yaw_pos
        else:
            waist_yaw_pos = 0.0

        self.height_speed_value = self._smooth_map(ry, -1.0, 1.0) if ry is not None else 0.0
        
        return {
            'mobile_x_vel': mobile_x_vel,
            'mobile_yaw_vel': mobile_yaw_vel,
            'waist_yaw_pos': waist_yaw_pos,
            'g1_height': self.height_speed_value,
        }

    def _update_waist_position(self, raw_value, current_position, max_velocity, min_position, max_position):
        velocity = self._smooth_map(raw_value, -max_velocity, max_velocity, deadzone=0.5)
        if velocity != 0.0:
            self.current_waist_yaw_pos = float(current_position) + velocity
            self.current_waist_yaw_pos = np.clip(self.current_waist_yaw_pos, min_position, max_position)

        return self.current_waist_yaw_pos
    
    def _map_forward_velocity(self, value):
        return self._smooth_map(value, -0.2, 0.2)
    
    def _map_lateral_velocity(self, value):
        return self._smooth_map(value, -0.6, 0.6)
    
    def _map_yaw_velocity(self, value, min_value, max_value):
        return self._smooth_map(value, min_value, max_value)
    
    def _smooth_map(self, value, out_min, out_max, deadzone=0.05):
        """
        Smooth mapping function
        Maps input value to output range using deadzone and smooth curve
        
        Args:
            value: Input value (-1 to 1)
            out_min: Output minimum value
            out_max: Output maximum value
            deadzone: Deadzone size
        """
        if abs(value) < deadzone:
            return 0.0
        t = (abs(value) - deadzone) / (1.0 - deadzone)
        t = np.clip(t, 0.0, 1.0)
        smooth = 6 * t**5 - 15 * t**4 + 10 * t**3
        return smooth * (out_max if value > 0 else out_min)

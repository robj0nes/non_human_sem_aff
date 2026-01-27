import numpy as np


class Vector:
    def __init__(self, x, y, z, w=None):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def to_numpy(self):
        return np.array([self.x, self.y, self.z, self.w]) \
            if self.w is not None else np.array([self.x, self.y, self.z])


class RobotTwist:
    def __init__(self, twist=None):
        if twist is None:
            self.angular = Vector(0, 0, 0)
            self.linear = Vector(0, 0, 0)
        else:
            self.angular = Vector(twist.angular.x, twist.angular.y, twist.angular.z)
            self.linear = Vector(twist.linear.x, twist.linear.y, twist.linear.z)


class RobotPose:
    def __init__(self, pose=None):
        if pose is None:
            self.position = Vector(0, 0, 0, 0)
            self.orientation = Vector(0, 0, 0, 1)
        else:
            self.position = Vector(pose.position.x, pose.position.y, pose.position.z)
            self.orientation = Vector(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)


class RobotLocalisation:
    def __init__(self, pose=None, twist=None):
        self.pose = RobotPose(pose)
        self.twist = RobotTwist(twist)

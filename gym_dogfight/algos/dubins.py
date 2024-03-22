"""
Calculate Dubins Curve between waypoints

author: Fischer, but I just copied the math from this paper:

fischergabbert@gmail.com

http://mems.eng.uci.edu/files/2014/04/Dubins_Set_Robotics_2001.pdf

Andrew Walker did this in C and I used that as a reference too..his github is out there somewhere

"""

"""
TODOS:

- Reduce computation time using the classification methods in the paper

"""

import matplotlib.pyplot as plt
import math
import numpy as np
from enum import Enum
from gym_dogfight.core.models import Waypoint


class DubinsTurnType(Enum):
    """
    直线（Straight, S）、左转弯（Left, L）、右转弯（Right, R）
    """
    LSL = 1  # 左转弯 - 直线 - 左转弯
    LSR = 2  # 左转弯 - 直线 - 右转弯
    RSL = 3  # 右转弯 - 直线 - 左转弯
    RSR = 4  # 右转弯 - 直线 - 右转弯
    RLR = 5  # 右转弯 - 左转弯 - 右转弯
    LRL = 6  # 左转弯 - 右转弯 - 左转弯


class DubinsParam:
    def __init__(self, p_init: Waypoint, seg_final, turn_radius, ):
        """

        :param p_init: 起始点
        :param seg_final: 存储计算出的Dubins路径的三个段（LSL, RSR等）的长度，（以转弯半径的倍数计量）
        :param turn_radius: 转弯半径
        """
        self.p_init = p_init
        self.seg_final = seg_final
        self.turn_radius = turn_radius
        self.type: DubinsTurnType = DubinsTurnType.LRL


def calc_dubins_path(wpt1, wpt2, turn_radius: float):
    """

    :param wpt1: Waypoint(x, y, psi)
    :param wpt2: Waypoint(x, y, psi)
    :param turn_radius: 转弯半径
    :return:
    """
    # Calculate a dubins path between two waypoints
    param = DubinsParam(wpt1, 0, 0)

    # 存储六种Dubins路径类型的参数
    tz = [0, 0, 0, 0, 0, 0]
    pz = [0, 0, 0, 0, 0, 0]
    qz = [0, 0, 0, 0, 0, 0]

    param.seg_final = [0, 0, 0]
    # Convert the headings from NED to standard unit cirlce, and then to radians
    psi1 = math.radians(_heading_to_standard(wpt1.psi))
    psi2 = math.radians(_heading_to_standard(wpt2.psi))

    # Do math
    param.turn_radius = turn_radius

    dx = wpt2.x - wpt1.x
    dy = wpt2.y - wpt1.y
    D = math.sqrt(dx * dx + dy * dy)  #
    d = D / param.turn_radius  # *** Normalize by turn radius...makes length calculation easier down the road.

    # Angles defined in the paper
    theta = math.atan2(dy, dx) % (2 * math.pi)  # 计算两点连线与x轴正方向的夹角，结果取模2 * math.pi确保角度在[0, 2π)范围内。
    alpha = (psi1 - theta) % (2 * math.pi)  # 计算起点方向与两点连线方向的相对角度
    beta = (psi2 - theta) % (2 * math.pi) # 计算终点方向与两点连线方向的相对角度
    best_word = -1
    best_cost = -1

    # Calculate all dubin's paths between points
    tz[0], pz[0], qz[0] = _dubins_LSL(alpha, beta, d)
    tz[1], pz[1], qz[1] = _dubins_LSR(alpha, beta, d)
    tz[2], pz[2], qz[2] = _dubins_RSL(alpha, beta, d)
    tz[3], pz[3], qz[3] = _dubins_RSR(alpha, beta, d)
    tz[4], pz[4], qz[4] = _dubins_RLR(alpha, beta, d)
    tz[5], pz[5], qz[5] = _dubins_LRL(alpha, beta, d)

    # Now, pick the one with the lowest cost
    for x in range(6):
        if (tz[x] != -1):
            cost = tz[x] + pz[x] + qz[x]
            if (cost < best_cost or best_cost == -1):
                best_word = x + 1
                best_cost = cost
                param.seg_final = [tz[x], pz[x], qz[x]]

    param.type = DubinsTurnType(best_word)
    return param


# Here's all of the dubins path math
def _dubins_LSL(alpha, beta, d):
    """

    :param alpha: 起点的方向与连接起点和终点的直线的夹角。
    :param beta: 终点的方向与连接起点和终点的直线的夹角。
    :param d: 起点和终点之间的距离，经过规范化处理，即除以了转弯半径。
    :return:
    """

    tmp0 = d + math.sin(alpha) - math.sin(beta) # 用于存储路径计算中的一个中间结果，这个结果是基于输入的距离和角度。
    tmp1 = math.atan2((math.cos(beta) - math.cos(alpha)), tmp0) # 计算的是连接起点和终点的直线与初始转弯圆的切线方向的角度。
    p_squared = 2 + d * d - (2 * math.cos(alpha - beta)) + (2 * d * (math.sin(alpha) - math.sin(beta)))
    if p_squared < 0:
        print('No LSL Path')
        p = -1
        q = -1
        t = -1
    else:
        t = (tmp1 - alpha) % (2 * math.pi)
        p = math.sqrt(p_squared)
        q = (beta - tmp1) % (2 * math.pi)
    return t, p, q


def _dubins_RSR(alpha, beta, d):
    tmp0 = d - math.sin(alpha) + math.sin(beta)
    tmp1 = math.atan2((math.cos(alpha) - math.cos(beta)), tmp0)
    p_squared = 2 + d * d - (2 * math.cos(alpha - beta)) + 2 * d * (math.sin(beta) - math.sin(alpha))
    if p_squared < 0:
        print('No RSR Path')
        p = -1
        q = -1
        t = -1
    else:
        t = (alpha - tmp1) % (2 * math.pi)
        p = math.sqrt(p_squared)
        q = (-1 * beta + tmp1) % (2 * math.pi)
    return t, p, q


def _dubins_RSL(alpha, beta, d):
    tmp0 = d - math.sin(alpha) - math.sin(beta)
    p_squared = -2 + d * d + 2 * math.cos(alpha - beta) - 2 * d * (math.sin(alpha) + math.sin(beta))
    if p_squared < 0:
        print('No RSL Path')
        p = -1
        q = -1
        t = -1
    else:
        p = math.sqrt(p_squared)
        tmp2 = math.atan2((math.cos(alpha) + math.cos(beta)), tmp0) - math.atan2(2, p)
        t = (alpha - tmp2) % (2 * math.pi)
        q = (beta - tmp2) % (2 * math.pi)
    return t, p, q


def _dubins_LSR(alpha, beta, d):
    tmp0 = d + math.sin(alpha) + math.sin(beta)
    p_squared = -2 + d * d + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) + math.sin(beta))
    if p_squared < 0:
        print('No LSR Path')
        p = -1
        q = -1
        t = -1
    else:
        p = math.sqrt(p_squared)
        tmp2 = math.atan2((-1 * math.cos(alpha) - math.cos(beta)), tmp0) - math.atan2(-2, p)
        t = (tmp2 - alpha) % (2 * math.pi)
        q = (tmp2 - beta) % (2 * math.pi)
    return t, p, q


def _dubins_RLR(alpha, beta, d):
    tmp_rlr = (6 - d * d + 2 * math.cos(alpha - beta) + 2 * d * (math.sin(alpha) - math.sin(beta))) / 8
    if (abs(tmp_rlr) > 1):
        print('No RLR Path')
        p = -1
        q = -1
        t = -1
    else:
        p = (2 * math.pi - math.acos(tmp_rlr)) % (2 * math.pi)
        t = (alpha - math.atan2((math.cos(alpha) - math.cos(beta)), d - math.sin(alpha) + math.sin(beta)) + p / 2 % (
                2 * math.pi)) % (2 * math.pi)
        q = (alpha - beta - t + (p % (2 * math.pi))) % (2 * math.pi)

    return t, p, q


def _dubins_LRL(alpha, beta, d):
    tmp_lrl = (6 - d * d + 2 * math.cos(alpha - beta) + 2 * d * (-1 * math.sin(alpha) + math.sin(beta))) / 8
    if (abs(tmp_lrl) > 1):
        print('No LRL Path')
        p = -1
        q = -1
        t = -1
    else:
        p = (2 * math.pi - math.acos(tmp_lrl)) % (2 * math.pi)
        t = (-1 * alpha - math.atan2((math.cos(alpha) - math.cos(beta)),
                                     d + math.sin(alpha) - math.sin(beta)) + p / 2) % (2 * math.pi)
        q = ((beta % (2 * math.pi)) - alpha - t + (p % (2 * math.pi))) % (2 * math.pi)
        print(t, p, q, beta, alpha)
    return t, p, q


def generate_dubins_traj(param, step):
    # Build the trajectory from the lowest-cost path
    x = 0
    i = 0
    length = (param.seg_final[0] + param.seg_final[1] + param.seg_final[2]) * param.turn_radius
    length = math.floor(length / step)
    path = -1 * np.ones((length, 3))

    while x < length:
        path[i] = _dubins_path(param, x)
        x += step
        i += 1
    return path


def _dubins_path(param, t):
    # Helper function for curve generation
    tprime = t / param.turn_radius
    p_init = np.array([0, 0, _heading_to_standard(param.p_init.psi) * math.pi / 180])
    #
    L_SEG = 1
    S_SEG = 2
    R_SEG = 3
    DIRDATA = np.array([[L_SEG, S_SEG, L_SEG], [L_SEG, S_SEG, R_SEG], [R_SEG, S_SEG, L_SEG], [R_SEG, S_SEG, R_SEG],
                        [R_SEG, L_SEG, R_SEG], [L_SEG, R_SEG, L_SEG]])
    #
    types = DIRDATA[param.type.value - 1][:]
    param1 = param.seg_final[0]
    param2 = param.seg_final[1]
    mid_pt1 = _dubins_segment(param1, p_init, types[0])
    mid_pt2 = _dubins_segment(param2, mid_pt1, types[1])

    if (tprime < param1):
        end_pt = _dubins_segment(tprime, p_init, types[0])
    elif (tprime < (param1 + param2)):
        end_pt = _dubins_segment(tprime - param1, mid_pt1, types[1])
    else:
        end_pt = _dubins_segment(tprime - param1 - param2, mid_pt2, types[2])

    end_pt[0] = end_pt[0] * param.turn_radius + param.p_init.x
    end_pt[1] = end_pt[1] * param.turn_radius + param.p_init.y
    end_pt[2] = end_pt[2] % (2 * math.pi)

    return end_pt


def _dubins_segment(seg_param, seg_init, seg_type):
    # Helper function for curve generation
    L_SEG = 1
    S_SEG = 2
    R_SEG = 3
    seg_end = np.array([0.0, 0.0, 0.0])
    if (seg_type == L_SEG):
        seg_end[0] = seg_init[0] + math.sin(seg_init[2] + seg_param) - math.sin(seg_init[2])
        seg_end[1] = seg_init[1] - math.cos(seg_init[2] + seg_param) + math.cos(seg_init[2])
        seg_end[2] = seg_init[2] + seg_param
    elif (seg_type == R_SEG):
        seg_end[0] = seg_init[0] - math.sin(seg_init[2] - seg_param) + math.sin(seg_init[2])
        seg_end[1] = seg_init[1] + math.cos(seg_init[2] - seg_param) - math.cos(seg_init[2])
        seg_end[2] = seg_init[2] - seg_param
    elif (seg_type == S_SEG):
        seg_end[0] = seg_init[0] + math.cos(seg_init[2]) * seg_param
        seg_end[1] = seg_init[1] + math.sin(seg_init[2]) * seg_param
        seg_end[2] = seg_init[2]

    return seg_end


def _wrap_angle_to_360(angle):
    posIn = angle > 0
    angle = angle % 360
    if angle == 0 and posIn:
        angle = 360
    return angle


def _wrap_angle_to_180(angle):
    q = (angle < -180) or (180 < angle)
    if (q):
        angle = _wrap_angle_to_360(angle + 180) - 180
    return angle


def _heading_to_standard(hdg):
    # Convert NED heading to standard unit cirlce...degrees only for now (Im lazy)
    thet = _wrap_angle_to_360(90 - _wrap_angle_to_180(hdg))
    return thet


def main():
    # User's waypoints: [x, y, heading (degrees)]
    pt1 = Waypoint(0, 0, 0)
    pt2 = Waypoint(6000, 7000, 270)
    # pt3 = Waypoint(1000,15000,180)
    # pt4 = Waypoint(0,0,270)
    Wptz = [pt1, pt2]
    # Run the code
    i = 0
    while i < len(Wptz) - 1:
        param = calc_dubins_path(Wptz[i], Wptz[i + 1], 500)
        path = generate_dubins_traj(param, 1)
        print(path.shape[0])
        # Plot the results
        plt.plot(Wptz[i].x, Wptz[i].y, 'kx')
        plt.plot(Wptz[i + 1].x, Wptz[i + 1].y, 'kx')
        plt.plot(path[:, 0], path[:, 1], 'b-')
        i += 1
    plt.grid(True)
    plt.axis("equal")
    plt.title('Dubin\'s Curves Trajectory Generation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == '__main__':
    main()

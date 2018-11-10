# 標準ライブラリ
import numpy as np
import matplotlib as plt
import sys
import math

# 関数
from ..functions.data.read import read_csv_with_header
from ..functions.math.angle import fit_angle_in_range
from ..models.robot import TwoWheeledRobot
from ..functions.math.normalization import normalize_by_min_max

# DWA
# path
class Path():
    """ Structure of Path

    Attributes
    ----------
    x : list of float in meters
        position x
    y : list of float in meters
        position y
    th : list of float in radians 
        angle th
    u_v : float in [m/s] 
        input velocity u_v
    u_th : float in [rad/s]
        input angular velocity u_th 
    """

    def __init__(self, u_v, u_th): 
        self.xs = None
        self.ys = None
        self.ths = None
        self.u_v = u_v
        self.u_th = u_th

class Obstacle():
    """ Structure of Obstacle

    Attributes
    ----------
    x : float in meters
        obstacle position coordinate x
    y : float in meters
        obstacle position coordinate x
    size : float in meters
        obstacles size
    """

    def __init__(self, x, y, size, pre_step):
        self.x = x
        self.y = y
        self.size = size
        self.score = [0.0 for _ in range(pre_step)]
        self.history_score = []
    
    def update(self, x, y, size=0.5):
        """update the obstacles positions

        Parameters
        -----------
        x : float in meters
        obstacle position coordinate x
        y : float in meters
            obstacle position coordinate x
        size : float in meters
            obstacles size
        """
        
        self.x = x
        self.y = y
        self.size = size
    
    def add_score(self, h_f, step):
        """
        add score of nmpc
        
        Parameters
        -------------
        score : float
            calculated by nmpc method
        step : int 
            step of prestep time
        """
        self.score[step] = h_f

    def save_score(self):
        """
        save score
        """
        self.history_score.append(self.score)

class Simulator_DWA_two_wheeled_robot(): # DWA
    """ Simulator robot class of Two_wheeled robot model in DWA predictiton 

    Attributes
    ----------
    max_accelation : float in meters
        initial position x
    max_ang_accelation : float in meters
        initial position y [m]
    init_th : float in radians
        initial position th [m]
    history_x : float of list
        time history position x 
    history_y : float of list
        time history position y
    history_th : float of list
        time history th
    """

    def __init__(self):
        # two_wheeled
        # limit of acceleration
        self.max_acceleration = 1.0
        self.max_ang_acceleration = 100 * math.pi /180
        # limit of velocity
        self.lim_max_velo = 1.6 # m/s
        self.lim_min_velo = 0.0 # m/s
        self.lim_max_ang_velo = math.pi # rad/s
        self.lim_min_ang_velo = -math.pi # rad/s

    def make_predict_state(self, velo, ang_velo, x, y, th, dt, pre_step):
        """ Make predict states

        Parameters
        ----------
        velo : float in [m/s]
            velocity input of the DWA
        ang_velo : float in [rad/s]
            angular velocity input of the DWA
        x : float in [m]
            now position x of the robot
        y : float in [m]
            now position y of the robot
        th : float in radians
            now angle th of the robot
        dt : float [s]
            sampling time
        pre_step : int
            predict length (Simulation time)
        
        Returns
        ----------
        next_xs : list of float
            length is pre_step
        next_ys : list of float
            length is pre_step
        next_ths : list of float
            length is pre_step
        """
        
        next_xs = []
        next_ys = []
        next_ths = []

        for i in range(pre_step):
            temp_x = velo * math.cos(th) * dt + x
            temp_y = velo * math.sin(th) * dt + y
            temp_th = ang_velo * dt + th

            next_xs.append(temp_x)
            next_ys.append(temp_y)
            next_ths.append(temp_th)

            x = temp_x
            y = temp_y
            th = temp_th

        return next_xs, next_ys, next_ths # 予想した軌跡

# DWA Dynamic WIndow Approach
class DWA():
    '''
    Path planning class using Dynamic Window Approach Algorithm

    Attributes
    -----------
    simulator_robot : Simulator_DWA_two_wheeled_robot class
        this class should have make_predict_state method

    pre_time : float in second
        prediction time length
    
    pre_step : int in time steps
        prediction time step length
    
    sampling time : float in seconds
    
    weight_angle : float
        DWA parameter this is related to the tracking angle
    
    weight_velo : float
        DWA parameter this is related to the speed of the robot
    
    weight_obs : float
        DWA parameter this is ralated to the obstacle avoidance
    
    area_dis_to_obs : float in meters
        DWA parameter this is the minimum distance to the obstacles

    history_paths : list of the time list of path class
        time history of the path which made by the DWA
        this variable keep all time step's path
    
    history_opt_paths : list of path class
        time history of the opt_path which made by the DWA
        this variable keep all time step's opt_path

    '''

    def __init__(self):
        # simulationter of DWA
        self.simulator_robot = Simulator_DWA_two_wheeled_robot()

        # prediction time
        self.pre_time = 3
        self.pre_step = 30

        # resolution length
        self.delta_velo = 0.02
        self.delta_ang_velo = 0.02

        # XXX : if you change you should check the other sampling time
        self.samplingtime = 0.065

        # DWA parameteres
        self.weight_angle = 0.1
        self.weight_velo = 0.2
        self.weight_obs = 0.1

        self.area_dis_to_obs = 5 # how far do we think the obstacles

        # history of the paths
        self.history_paths = []
        self.history_opt_paths = []

    def calc_input(self, g_x, g_y, g_th, state, obstacles, human):
        """ To make optimization path class
 
        Parameters
        ------------
        g_x : float in meters
            goal position x
        g_y : float in meters
            goal position y
        g_th : float in radians
            goal angle th
        state : Robot class
            Robot class should have positions and angles's variables like x, y, th
        obstacles : list of obstacle class
            Obstacle class should have positions and angles's variables like x, y, th
        human : class of human class
            Obstacle class should have positions and angles's variables like x, y, th, v_x, v_y, v_th

        Returns
        -----------
        paths : list of path class
            all paths
        opt_path : path class
            optimized path
        history_paths : list of path class
            all time history path
        history_opt_paths : list of opt_path class
            all time history optimized path
        """
        paths = self._make_path(state)

        opt_path = self._eval_path(paths, g_x, g_y, state, obstacles)
        self.history_opt_paths.append(opt_path)
        
        
        return paths, opt_path, self.history_paths, self.history_opt_paths

    def _make_path(self, state): 
        """ making candidate of the paths

        Parameters
        ----------
        state : robot class
            this variable should have positions and angles and inputs
        
        Returns
        ---------
        paths : list of path class
            this variable is composed of all candidate path
        """

        # range calculate
        min_velo, max_velo, min_ang_velo, max_ang_velo = self._calc_range_velo(state)

        paths = []

        # Search the path in velocity and angular velocity range
        for ang_velo in np.arange(min_ang_velo, max_ang_velo, self.delta_ang_velo):
            for velo in np.arange(min_velo, max_velo, self.delta_velo):

                path = Path(velo, ang_velo) # making path class

                next_x, next_y, next_th \
                    = self.simulator_robot.make_predict_state(velo, ang_velo, state.x, state.y, state.th, self.samplingtime, self.pre_step)

                path.xs = next_x
                path.ys = next_y
                path.ths = next_th
            
                paths.append(path) # all paths are appended in path class
        
        # save time histories of path's candidate
        self.history_paths.append(paths)

        return paths

    def _calc_range_velo(self, state):
        """ Calculate the range of possivle inputs

        Parameters
        ----------
        state : robot class
            this variable should have positions and angles and inputs
        
        Returns
        ----------
        min_velo : float in [m/s]
            minimum possible input angular velocity
        max_velo : float in [m/s]
            maximum possible input angular velocity
        min_ang_velo : float in [rad/s]
            minimum possible input angular velocity
        max_ang_velo : float in [rad/s]
            maximum possible input angular velocity
        """

        # velocity
        range_velo = self.samplingtime * self.simulator_robot.max_acceleration
        min_velo = state.u_v - range_velo
        max_velo = state.u_v + range_velo
        # min
        if min_velo < self.simulator_robot.lim_min_velo:
            min_velo = self.simulator_robot.lim_min_velo
        # max
        if max_velo > self.simulator_robot.lim_max_velo:
            max_velo = self.simulator_robot.lim_max_velo

        # angular velocity
        range_ang_velo = self.samplingtime * self.simulator_robot.max_ang_acceleration
        min_ang_velo = state.u_th - range_ang_velo
        max_ang_velo = state.u_th + range_ang_velo
        # min
        if min_ang_velo < self.simulator_robot.lim_min_ang_velo:
            min_ang_velo = self.simulator_robot.lim_min_ang_velo
        # max
        if max_ang_velo > self.simulator_robot.lim_max_ang_velo:
            max_ang_velo = self.simulator_robot.lim_max_ang_velo

        return min_velo, max_velo, min_ang_velo, max_ang_velo

    def _eval_path(self, paths, g_x, g_y, state, obastacles):
        """ Evaluate the path

        Parameters
        ----------
        paths : list of path class
            candidate paths
        g_x : float in meters
            goal position x
        g_y : float in meters
            goal postion y
        state : Robot class
            now time step sate of robot
        obstacles : Obstacle class
            now time step obstacles
        
        Returns
        ---------
        opt_path : path class
            optimized path

        """

        nearest_obs = self._calc_nearest_obs(state, obastacles)

        # score list
        score_heading_angles = [] 
        score_heading_velos = []
        score_obstacles = []

        # Search the all path
        for path in paths:
            # (1) heading_angle
            score_heading_angles.append(self._heading_angle(path, g_x, g_y))
            # (2) heading_velo
            score_heading_velos.append(self._heading_velo(path))
            # (3) obstacle
            score_obstacles.append(self._judge_obstacle(path, nearest_obs))

        for scores in [score_heading_angles, score_heading_velos, score_obstacles]:
            scores = normalize_by_min_max(scores)
            # print(scores)
        score = 0.0
        # optimized the path
        for k in range(len(paths)):
            temp_score = 0.0

            temp_score = self.weight_angle * score_heading_angles[k] + \
                         self.weight_velo * score_heading_velos[k] + \
                         self.weight_obs * score_obstacles[k]
        
            if temp_score > score:
                opt_path = paths[k]
                score = temp_score
                
        return opt_path

    def _heading_angle(self, path, g_x, g_y):
        """ Calculate heading angle

        Parameters
        -----------
        path : path class
            candidate path
        g_x : float in meteres
            goal postion x
        g_y : float in meters
            goal position y

        Returns
        ---------
        score_angle : float
            score of angle, how much is the last path way point heading to goal
        """
        
        # last position and state of the candidate path
        last_x = path.xs[-1]
        last_y = path.ys[-1]
        last_th = path.ths[-1]

        # the angle to the goal
        angle_to_goal = math.atan2(g_y-last_y, g_x-last_x)

        # calc score
        score_angle = angle_to_goal - last_th

        # format the angle
        score_angle = abs(fit_angle_in_range(score_angle, min_angle=-math.pi, max_angle=math.pi))
        
        # considering the evaluation (max -> good)
        score_angle = math.pi - score_angle

        return score_angle

    def _heading_velo(self, path): # 速く進んでいるか（直進）
        """ Calc heading angle

        Parameters
        -----------
        path : path class
            candidate of path
        
        Returns
        ----------
        score_heading_velo : float
            score of heading angle, velocity speed

        """
        score_heading_velo = path.u_v

        return score_heading_velo

    def _calc_nearest_obs(self, state, obstacles):
        """ Calc nearest obstacles 
        if the obstacles are far, we dont think about the obstacles

        Parameters
        -----------
        state : Robot class
            Now time step state of the robot
        obstacles : list of Obstacle class
            This variable is including recognized obstacle

        Returns
        ----------
        nearest_obs : list of Obstacle class
            This list composed of the obstacles which is within some_distance
        """
        nearest_obs = []

        for obs in obstacles:
            temp_dis_to_obs = math.sqrt((state.x - obs.x) ** 2 + (state.y - obs.y) ** 2)

            if temp_dis_to_obs < self.area_dis_to_obs :
                nearest_obs.append(obs)

        return nearest_obs

    def _judge_obstacle(self, path, nearest_obs):
        """Judge the path in distance to obstacle
        if its near, the path is exclude

        Parameters 
        -----------
        path : path class

        Returns
        ----------
        score_obstacle : float
            score about obstacle, nearest distance from the path's way point to the obstacles
         
        """
        score_obstacle = 2.0 # this parameter is for avoiding the divergence of score
        temp_dis_to_obs = 0.0

        for i in range(len(path.xs)):
            for obs in nearest_obs: 
                temp_dis_to_obs = math.sqrt((path.xs[i] - obs.x) * (path.xs[i] - obs.x) + (path.ys[i] - obs.y) *  (path.ys[i] - obs.y))
                
                if temp_dis_to_obs < score_obstacle:
                    score_obstacle = temp_dis_to_obs # nearest obs

                if temp_dis_to_obs < obs.size + 0.15: # margin
                    score_obstacle = 0.0
                    break

            else:
                continue
            
            break

        return score_obstacle
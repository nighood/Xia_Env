
import copy
import itertools
from typing import Any, Union, Dict, Tuple, List, SupportsFloat
import gymnasium
from matplotlib import pyplot as plt
import numpy as np

from entities import UAV, People


class Xia1(gymnasium.Env):
    def __init__(self) -> None:
        super().__init__()
        config, uav_config, people_config = self.default_config()
        # uavs = [UAV(uav_id, config["uav_action_lst"]) for uav_id in range(config["num_uavs"])]
        self.uavs = [UAV(uav_id, **uav_config)
                     for uav_id in range(config["num_uavs"])]
        self.people = [People(p_id, **people_config)
                       for p_id in range(config["num_people"])]
        # self.BeliefMap = np.zeros((config["width"], config["height"]))

        self.time = None
        self.closed = False

        self.action_space = gymnasium.spaces.MultiDiscrete(
            [len(config["uav_action_lst"]) for _ in self.uavs])
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(config["observation_size"],))
        self.config = config

        self.fig = plt.figure(figsize=(10, 5))
        self.ax = self.fig.add_subplot(121, projection='3d')
        # self.fig1, self.ax1 = plt.subplots()
        self.ax1 = self.fig.add_subplot(122)

    @classmethod
    def default_config(cls):
        width, height, h_3d = 80, 80, 100   # 图大小
        uav_h_min = 10  # uav 限制高度大于10
        search_banjing_max = 10
        num_uavs, num_people = 2, 10
        # uav_action_lst = list(itertools.product([-10, 0, 10], [-10, 0, 10], [-5, 0, 5]))
        # uav_action_lst = [(0, 0, 0), (5, 0, 0), (0, 5, 0),
        #                   (0, 0, 2), (-5, 0, 0), (0, -5, 0), (0, 0, -2)]
        uav_action_lst = [(x, y, z)
                          for x in [-5, 0, 5]
                          for y in [-5, 0, 5]
                          for z in [-2, 0, 2]]
        # people_action_lst = list(itertools.product([-1, 0, 1], [-1, 0, 1]))
        ep_time = 200   # 结束时间
        # observation_size = 90015
        observation_size = width*height + num_uavs*3
        seed = 202401
        if seed != None:
            seed_people = seed + num_uavs + 10
        else:
            seed_people = None
        uav_config = {
            "width": width,
            "height": height,
            "h_3d": h_3d,
            "seed": seed,
            "h_min": uav_h_min,
            "uav_action_lst": uav_action_lst,

            "search_banjing_max": search_banjing_max,
        }
        people_config = {
            "width": width,
            "height": height,
            "h_3d": h_3d,
            "seed": seed_people,

            "search_banjing_max": search_banjing_max,
        }
        config = {
            # environment parameters:
            "width": width,
            "height": height,
            "h_3d": h_3d,
            "seed": seed,
            "num_uavs": num_uavs,
            "num_people": num_people,
            "EP_MAX_TIME": ep_time,

            "uav_action_lst": uav_action_lst,

            "observation_size": observation_size,

            "search_banjing_max": search_banjing_max,
            "save_replay": False,
        }
        return config, uav_config, people_config

    def reset(self, *, seed: Union[int, None] = None, options: Union[Dict[str, Any], None] = None) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.UAV_traj_render = []
        self.time = 0
        self.BMap_previous = None   # Modify
        self.closed = False
        self.rew_render = 0
        self.rew_render_total = 0
        self.BeliefMap = np.ones(
            (self.config["width"], self.config["height"])) * 0.5

        for i in self.uavs:
            i.reset()
        for i in self.people:
            i.reset()

        self.groundTruth, self.groundTruth0 = self.GroundTruth()     # gT -1,1矩阵 gT0 0,1矩阵

        obs = self._get_obs()
        info = self.ObsDict

        if self.config['save_replay']:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection='3d')
            self.fig1, self.ax1 = plt.subplots()

        return obs, info

    def _get_uav_map(self):
        # 先不用，因为要三维
        uav_map = np.zeros(
            (self.config["width"], self.config["height"], self.config["h_3d"]))
        for _ in self.uavs:
            uav_map[int(_.x), int(_.y), int(_.z)] = 1
        return uav_map

    def _get_obs(self):
        _UAV_location = []
        _People_location = []
        for _ in self.uavs:
            _UAV_location.append([_.x, _.y, _.z])
        for _ in self.people:
            _People_location.append([_.x, _.y])
        BeliefMap = self.BeliefMap

        self.ObsDict = {"_UAV_location": _UAV_location,
                        "_People_location": _People_location,
                        "BeliefMap": BeliefMap, }

        MMmax = max(self.config["width"],
                    self.config["height"], self.config["h_3d"])
        _UAV_location = self.MinMaxNorm(_UAV_location, 0, MMmax)
        _People_location = self.MinMaxNorm(_People_location, 0, MMmax)

        NormObsDict = {"_UAV_location": _UAV_location,
                       # "_People_location": _People_location, # 实际没有人的观测，只有BeliefMap
                       "BeliefMap": BeliefMap, }

        start = True
        for i in NormObsDict.values():
            if start:
                arr = i.flatten()
                start = False
                continue
            arr = np.concatenate((arr, i.flatten()))
        return arr.astype(np.float32)

    def MinMaxNorm(self, l, min, max):
        l = np.clip(np.array(l), min, max)
        return (l - min) / (max - min)

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        OutFlag = False
        self.CollisionFlag = 0
        assert not self.closed, "step() called on terminated episode"
        self.time += 1
        self.BMap_previous = copy.deepcopy(self.BeliefMap)  # Modify

        for i in range(len(self.uavs)):
            uav_1 = self.uavs[i]
            action_i = self.config["uav_action_lst"][action[i]]
            (uav_1.x, uav_1.y, uav_1.z), self.CollisionFlag = uav_1.move(
                action_i, self.CollisionFlag)
            if self.UAV_out(uav_1):
                # OutFlag = True
                continue
            self.BMapChange(uav_1)

        rewards = self._get_reward()
        self.rew_render = rewards
        self.rew_render_total += rewards

        obs = self._get_obs()
        info = self.ObsDict

        if self.time >= self.config['EP_MAX_TIME'] or OutFlag:
            self.closed = True

        return obs, rewards, self.closed, self.closed, info

    def BMapChange(self, uav):
        '''
        BBBBBB
        B#####   半径2 x=y=3
        B#####
        B##X##
        B#####
        B#####
        '''
        search_banjing = int(
            uav.z * self.config['search_banjing_max'] / self.config['h_3d'])

        xm = max(0, int(uav.x) - search_banjing)
        xl = min(self.config['width'] - 1, int(uav.x) + search_banjing)
        ym = max(0, int(uav.y) - search_banjing)
        yl = min(self.config['width'] - 1, int(uav.y) + search_banjing)

        AgT0 = self.groundTruth0[xm:(xl+1), ym:(yl+1)]  # 0,1矩阵
        AgT = self.groundTruth[xm:(xl+1), ym:(yl+1)]  # -1,1矩阵

        # noise = np.clip(1/np.random.normal(uav.z, 10), -1, 1) # 0-1, z越低值越大
        # noise = np.random.normal(
        #     uav.z/1000.0, uav.z/200, size=AgT0.shape) * (-AgT)
        noise = np.random.normal(-2.206/(uav.z-5.191) +
                                 0.470, uav.z/1000, size=AgT0.shape) * (-AgT)
        noise = np.clip(noise, 0, 1)
        y_belief = AgT0 + noise    # -1，1矩阵
        # y_belief = AgT0   # noise free
        observation_prob = np.clip(y_belief, 0, 1)
        prior_prob = self.BeliefMap[xm:(xl+1), ym:(yl+1)]

        posterior_prob = (observation_prob * prior_prob) / \
            ((observation_prob * prior_prob) +
             ((1 - observation_prob) * (1 - prior_prob)) + 1e-7)

        self.BeliefMap[xm:(xl+1), ym:(yl+1)] = posterior_prob
        self.BeliefMap = np.clip(self.BeliefMap, 0, 1)

        return

    def GroundTruth(self):
        '''
        1为有人
        gT -1,1矩阵
        gT0 0,1矩阵
        '''
        gT = - np.ones((self.config["width"], self.config["height"]))
        gT0 = np.zeros((self.config["width"], self.config["height"]))
        for _ in self.people:
            gT[int(_.x), int(_.y)] = 1
            gT0[int(_.x), int(_.y)] = 1
        return gT, gT0

    def calculate_entropy(self, belief_map):
        # 计算信息熵
        # 加入小量避免对0取对数
        entropy = -np.sum(belief_map * np.log(belief_map + 1e-7))
        return entropy

    def _get_reward(self):  # Modify
        entropy_now = self.calculate_entropy(self.BeliefMap)
        entropy_previous = self.calculate_entropy(self.BMap_previous)
        self.rew1 = - (entropy_now - entropy_previous)

        BMap_difference = self.BeliefMap - self.BMap_previous   # 空地负数，人正数
        Factor = self.groundTruth0 * \
            (200 - 1) + self.groundTruth   # 9,0矩阵+1，-1矩阵=10，-1矩阵
        # self.rew2 = np.sum(BMap_difference * self.groundTruth)
        self.rew2 = np.sum(BMap_difference * Factor)

        reward = self.rew1*5 + self.rew2
        self.rew3 = 0

        for i in self.uavs:
            self.UAV_traj_render.append([i.x, i.y, i.z])
            # if i.x < 1 or i.x > (self.config['width'] - 2) or i.y < 1 or i.y > (self.config['height'] - 2):
            #     # self.rew3 -= (reward + self.rew_render_total)
            #     self.rew3 -= 50
        # if self.CollisionFlag:
        self.rew3 -= (200 * self.CollisionFlag)

        # people_confidence = self.groundTruth0 * self.BeliefMap
        # self.rew4 = np.count_nonzero(rewards > 1e-6)

        return reward + self.rew3

    def UAV_out(self, uav_i):
        i = uav_i
        if i.x < 1 or i.x > (self.config['width'] - 2) or i.y < 1 or i.y > (self.config['height'] - 2):
            return True
        return False

    def render(self) -> None:
        # self.ax.cla()
        # self.ax1.cla()
        self.fig.clf()
        # self.ax1 = self.fig1.add_subplot()
        self.ax = self.fig.add_subplot(121, projection='3d')
        # self.fig1, self.ax1 = plt.subplots()
        self.ax1 = self.fig.add_subplot(122)
        plt.subplots_adjust(left=None, bottom=None,
                            right=None, top=None, wspace=0.5)

        for i in self.ObsDict['_UAV_location']:
            self.ax.scatter(int(i[0]), int(i[1]), int(i[2]), marker='x')

        for i in self.ObsDict['_People_location']:
            self.ax.scatter(int(i[0]), int(i[1]), 0, marker='o')

        if self.time > 1:
            for i in range(self.time - 1):
                self.ax.plot3D([self.UAV_traj_render[i*2][0], self.UAV_traj_render[(i+1)*2][0]],
                               [self.UAV_traj_render[i*2][1],
                                   self.UAV_traj_render[(i+1)*2][1]],
                               [self.UAV_traj_render[i*2][2], self.UAV_traj_render[(i+1)*2][2]], color="red")
                self.ax.plot3D([self.UAV_traj_render[i*2+1][0], self.UAV_traj_render[(i+1)*2+1][0]],
                               [self.UAV_traj_render[i*2+1][1],
                                   self.UAV_traj_render[(i+1)*2+1][1]],
                               [self.UAV_traj_render[i*2+1][2], self.UAV_traj_render[(i+1)*2+1][2]], color="blue")

        self.ax1.matshow(self.BeliefMap + 1, cmap=plt.cm.Blues)
        caxes = self.ax1.matshow(self.BeliefMap, cmap=plt.cm.Blues)
        for i in self.ObsDict['_People_location']:
            self.ax1.scatter(int(i[1]), int(
                i[0]), marker='.', s=1, color="red")
        for i in self.ObsDict['_UAV_location']:
            self.ax1.scatter(int(i[1]), int(i[0]), marker='x', s=5)
        self.fig.colorbar(caxes)
        # self.fig.suptitle('My Figure')
        self.fig.suptitle("Step = " + str(self.time) +
                          ", Step Reward = " + str(round(self.rew_render, 2)) +
                          ", Total Reward = " + str(round(self.rew_render_total, 2)) +
                          ", r1 = " + str(round(self.rew1, 2)) +
                          ", r2 = " + str(round(self.rew2, 2)) +
                          ", r3 = " + str(round(self.rew3, 2))
                          )
        # plt.pause(0.0001)
        self.ax.view_init(elev=0, azim=0)
        plt.savefig('imgs/' + str(self.time))
        print(self.time, "Render")
        # return self.render_to_frame()

    def render_to_frame(self):
        # 返回 Matplotlib 图形对象
        self.fig.canvas.draw()
        frame_3d = np.array(self.fig.canvas.renderer.buffer_rgba())

        self.fig1.canvas.draw()
        frame_2d = np.array(self.fig1.canvas.renderer.buffer_rgba())

        # 将两个图像拼接成一个帧
        frame_combined = np.concatenate((frame_3d, frame_2d), axis=1)

        return frame_combined

    def seed(self, seed: int = 0):
        np.random.seed(seed)

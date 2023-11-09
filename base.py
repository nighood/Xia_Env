
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
        self.uavs = [UAV(uav_id, **uav_config) for uav_id in range(config["num_uavs"])]
        self.people = [People(p_id, **people_config) for p_id in range(config["num_people"])]
        self.BeliefMap = np.zeros((config["width"], config["height"]))

        self.time = None
        self.closed = False

        self.action_space = gymnasium.spaces.MultiDiscrete([len(config["uav_action_lst"]) for _ in self.uavs])
        self.observation_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(config["observation_size"],))
        self.config = config

        self.ax = plt.figure().add_subplot(projection='3d')
        self.fig1, self.ax1 = plt.subplots()

    @classmethod
    def default_config(cls):
        width, height, h_3d = 300, 300, 100   # 图大小
        uav_h_min = 10  # uav 限制高度大于10
        search_banjing_max = 10
        num_uavs, num_people = 5, 15
        # uav_action_lst = list(itertools.product([-10, 0, 10], [-10, 0, 10], [-5, 0, 5]))
        uav_action_lst = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 10), (-10, 0, 0), (0, -10, 0), (0, 0, -10)]
        # people_action_lst = list(itertools.product([-1, 0, 1], [-1, 0, 1]))
        ep_time = 1000   # 结束时间
        # observation_size = 90015
        observation_size = width*height + num_uavs*3
        seed = 0
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
        }
        return config, uav_config, people_config
    
    def reset(self, *, seed: Union[int, None] = None, options: Union[Dict[str, Any], None] = None) -> Tuple[Any, Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.time = 0

        for i in self.uavs:
            i.reset()
        for i in self.people:
            i.reset()
        
        obs = self._get_obs()
        info = self.ObsDict

        return obs, info
    
    def _get_uav_map(self):
        # 先不用，因为要三维
        uav_map = np.zeros((self.config["width"], self.config["height"], self.config["h_3d"]))
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
                        "BeliefMap": BeliefMap,}

        MMmax = max(self.config["width"], self.config["height"], self.config["h_3d"])
        _UAV_location = self.MinMaxNorm(_UAV_location, 0, MMmax)
        _People_location = self.MinMaxNorm(_People_location, 0, MMmax)

        NormObsDict = {"_UAV_location": _UAV_location,
                        # "_People_location": _People_location, # 实际没有人的观测，只有BeliefMap
                        "BeliefMap": BeliefMap,}
        
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
        self.time += 1

        for i in range(len(self.uavs)):
            uav_1 = self.uavs[i]
            action_i = self.config["uav_action_lst"][action[i]]
            uav_1.x, uav_1.y, uav_1.z = uav_1.move(action_i)
            self.BMapChange(uav_1)

        rewards = self._get_reward()

        obs = self._get_obs()
        info = self.ObsDict

        if self.time >= self.config['EP_MAX_TIME']:
            self.closed = True

        return obs, rewards, self.closed, self.closed, info
    
    def BMapChange1(self, uav):
        x, y, z = uav.x, uav.y, uav.z
        search_banjing = uav.z * self.config['search_banjing_max'] / self.config['h_3d']
        search_banjing = int(search_banjing)
        A = np.ones((search_banjing * 2 + 1, search_banjing * 2 + 1))
        for r in range(1, search_banjing + 1):
            A[r:A.shape[0]-r, r:A.shape[0]-r] += 1
        # print(A)
        # A = A / (np.max(A) * uav.z * 10)
        A = (A / (np.max(A))) * (1/np.random.normal(uav.z, 10)) * 1
        # print(A)

        xm = max(0, int(x) - search_banjing)
        xl = min(self.config['width'], int(x) + search_banjing + 1)
        ym = max(0, int(y) - search_banjing)
        yl = min(self.config['width'], int(y) + search_banjing + 1)
        self.BeliefMap[xm:xl, ym:yl] += A
        self.BeliefMap = np.clip(self.BeliefMap, 0, 1)

        # B = self.BeliefMap[xm:xl, ym:yl]
        # print((A == B).all())

        return
    
    def BMapChange(self, uav):
        x, y, z = uav.x, uav.y, uav.z
        search_banjing = uav.z * self.config['search_banjing_max'] / self.config['h_3d']
        search_banjing = int(search_banjing)

        xm = max(0, int(x) - search_banjing)
        xl = min(self.config['width'], int(x) + search_banjing + 1)
        ym = max(0, int(y) - search_banjing)
        yl = min(self.config['width'], int(y) + search_banjing + 1)

        A = np.ones((search_banjing * 2 + 1, search_banjing * 2 + 1)) * (-1)
        for _ in self.people:
            if _.x <= xl and _.x >= xm and _.y <= yl and _.y >= ym:
                A[int(_.x) - xm, int(_.y) - ym] = 1
        # A为-1，1矩阵
        # noise = np.clip(1/np.random.normal(uav.z, 10), -1, 1) # 0-1, z越低值越大
        noise = np.clip(np.random.normal(80/uav.z, 2, size=A.shape)/10, -1, 1)
        y_belief = A * noise    # -1，1矩阵

        self.BeliefMap[xm:xl, ym:yl] += y_belief / self.config['EP_MAX_TIME']
        # self.BeliefMap = np.clip(self.BeliefMap, 0, 1)

        # B = self.BeliefMap[xm:xl, ym:yl]
        # print((A == B).all())

        return

    def _get_reward(self):
        reward = 0
        for i in self.people:
            reward += self.BeliefMap[int(i.x), int(i.y)]
        return reward

    def render(self) -> None:
        self.ax.cla()
        # self.ax1.cla()
        self.fig1.clf()
        self.ax1 = self.fig1.add_subplot()

        for i in self.ObsDict['_UAV_location']:
            self.ax.scatter(int(i[0]), int(i[1]), int(i[2]), marker='x')
        
        for i in self.ObsDict['_People_location']:
            self.ax.scatter(int(i[0]), int(i[1]), 0, marker='o')
        
        # self.ax1.matshow(self.BeliefMap + 1, cmap=plt.cm.Blues)
        caxes = self.ax1.matshow(self.BeliefMap, cmap=plt.cm.Blues)
        for i in self.ObsDict['_People_location']:
            self.ax1.scatter(int(i[0]), int(i[1]), marker='.', s=1)
        self.fig1.colorbar(caxes)
        plt.pause(0.01)
        return None
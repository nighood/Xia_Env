from typing import List, Tuple

import numpy as np

class People:
    def __init__(self, PeopleID: int,
                 width: int, 
                 height: int, 
                 h_3d: int,
                 seed: int,
                 search_banjing_max,):
        self.id = PeopleID
        self.x: float = None
        self.y: float = None

        self.width, self.height, self.h_3d = width, height, h_3d # 图的总大小
        self.seed = None
        if seed != None:
            self.seed = seed + self.id
        self.rng = None

        self.search_banjing_max = search_banjing_max

    def __str__(self) -> str:
        return f"People ID: {self.id}"
    
    def reset(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed) # 重置随机数生成器

        self.x = int(self.rng.uniform(self.search_banjing_max, self.width - self.search_banjing_max))    # 考虑探索半径
        self.y = int(self.rng.uniform(self.search_banjing_max, self.height - self.search_banjing_max))

class UAV:
    def __init__(self, 
                 UAV_ID: int, 
                 width: int, 
                 height: int, 
                 h_3d: int,
                 seed: int,
                 search_banjing_max,
                 h_min: int=10,
                 uav_action_lst: List=None,
                 ):
        self.id = UAV_ID
        self.uav_action_lst = uav_action_lst

        self.x: float = None
        self.y: float = None
        self.z: float = None
        self.h_min = h_min

        self.width, self.height, self.h_3d = width, height, h_3d # 图的总大小
        self.seed = None
        if seed != None:
            self.seed = seed + self.id
        self.rng = None

        self.search_banjing_max = search_banjing_max

    def __str__(self) -> str:
        return f"UAV ID: {self.id}"
    
    def reset(self) -> None:
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed) # 重置随机数生成器

        self.x = int(self.rng.uniform(self.search_banjing_max, self.width - self.search_banjing_max - 1))    # 考虑探索半径
        self.y = int(self.rng.uniform(self.search_banjing_max, self.height - self.search_banjing_max - 1))
        self.z = int(self.rng.uniform(self.h_min, self.h_3d))        # uav 限制高度大于self.h_min

    def move(self, action: Tuple[float, float, float] = None) -> None:
        if action == None:
            x_change, y_change, z_change = self.rng.choice(self.uav_action_lst)
        else:
            x_change, y_change, z_change = action

        position = np.array([self.x, self.y, self.z])
        position = position + [x_change, y_change, z_change]
        position = np.clip(position, self.search_banjing_max, max(self.width, self.height) - self.search_banjing_max - 1)
        position[2] = np.clip(position[2], self.h_min, self.h_3d)
        return tuple(position)




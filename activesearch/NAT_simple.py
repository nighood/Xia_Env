import numpy as np
import gymnasium
import imageio
import activesearch
from easydict import EasyDict

width, height, h_3d = 80, 80, 100
uav_action_lst = [(0, 0, 0), (5, 0, 0), (0, 5, 0), (0, 0, 2), (-5, 0, 0), (0, -5, 0), (0, 0, -2)]
num_uavs = 2
num_people = 15
search_banjing_max = 10
max_steps = 200

mcfg=EasyDict(
        env_config= EasyDict({
            # "width": width,
            # "height": height,
            # "h_3d": h_3d,
            # "seed": seed,
            # "num_uavs": num_uavs,
            # "num_people": num_people,
            "EP_MAX_TIME": max_steps,
            # "uav_action_lst": uav_action_lst,
            # "observation_size": width*height+3*num_uavs,
            # "search_banjing_max": search_banjing_max,
            "save_replay": True,
        })
        # uav_config = EasyDict({
        #     "width": width,
        #     "height": height,
        #     "h_3d": h_3d,
        #     "seed": seed,
        #     "h_min": uav_h_min,
        #     "uav_action_lst": uav_action_lst,
        #     "search_banjing_max": search_banjing_max,
        # })
        # people_config = EasyDict({
        #     "width": width,
        #     "height": height,
        #     "h_3d": h_3d,
        #     "seed": seed,
        #     "search_banjing_max": search_banjing_max,
        # })
        )

# def uncovered_area_count(fov, uncovered_area):
#     # 计算未覆盖区域
#     for i in range(80):
#         for j in range(80):
#             # 计算点到当前无人机的距离
#             distance = np.linalg.norm(np.array([i, j]) - np.array([x + 5, y]))

#             # 如果距离小于 FOV，则认为该点被覆盖
#             if distance < fov:
#                 uncovered_area[i, j] = 0
#     return uncovered_area

# 修改水平移动规则
def horizontal_move(current_position, horizontal_direction, fov):
    x = int(current_position)

    # 选择水平方向
    direction = horizontal_direction

    # 未到边界
    if direction == 1 and x + 5 <= 80:
        action_index = 1
        # uncovered_area = uncovered_area_count(fov, uncovered_area)
    elif direction == -1 and x - 5 >= 0:
        action_index = 4
        # uncovered_area = uncovered_area_count(fov, uncovered_area)
    # 到达边界
    elif (direction == 1 and x + 5 > 80) or (direction == -1 and x - 5 < 0):
        action_index = 0

    return action_index


# 修改垂直移动规则
def vertical_move(current_position, vertical_direction, fov):
    y = int(current_position)

    # 选择水平方向
    direction = vertical_direction

    # 未到边界
    if direction == 1 and y + 5 <= 80:
        action_index = 2
        # uncovered_area = uncovered_area_count(fov, uncovered_area)
    elif direction == -1 and y - 5 >= 0:
        action_index = 5
        # uncovered_area = uncovered_area_count(fov, uncovered_area)
    # 到达边界
    elif (direction == 1 and y + 5 > 80) or (direction == -1 and y - 5 < 0):
        action_index = 0

    return action_index

def is_valid_action(current_position, action, boundary):
    new_position = np.array(current_position) + np.array(action[:2])
    return all(0 <= coord < bound for coord, bound in zip(new_position, boundary))

def rule_based_algorithm(num_uav, uav_action_lst, env, max_steps=200, p_up_down=0.5):
    uav_actions = []
    frames = []
    obs, _ = env.reset()
    episode_reward = 0
    # uncovered_area = np.ones((80, 80))
    #获取初始方向
    horizontal_direction = []
    vertical_direction = []
    for uav_id in range(num_uav):
        pos = obs[3*uav_id:3*uav_id+3]
        horizontal_direction.append(1 if pos[0] < 40 else -1)  # 初始水平方向，往右是1
        vertical_direction.append(1 if pos[1] < 40 else -1) # 初始垂直方向，往上是1

    for step in range(max_steps):
        actions = []
        for uav_id in range(num_uav):
            # 获取当前无人机的观测值
            pos = obs[3*uav_id:3*uav_id+3]
            if np.random.rand() < p_up_down:
                # 进行上下移动
                if pos[2] >= 98:  # 如果当前高度已经是最大高度，则下降
                    action_index = uav_action_lst.index((0, 0, -2))
                elif pos[2] <=12:  # 如果当前高度已经是最小高度，则上升
                    action_index = uav_action_lst.index((0, 0, 2))
                else:  # 否则，随机选择上升或下降
                    action_index = np.random.choice([uav_action_lst.index((0, 0, 2)), uav_action_lst.index((0, 0, -2))])

            else:
                # 计算当前无人机的 FOV
                fov_max = 10
                fov = int(pos[2] * fov_max / 100)

                # # 计算当前无人机的位置
                # current_position = np.array([pos[0], pos[1]])

                if 0 <= pos[0] <= 80:
                    # 如果当前位置在水平范围内，先走水平方向
                    action_index = horizontal_move(pos[0], horizontal_direction[uav_id], fov)
                    # 水平到达边界
                    if action_index == 0:
                        horizontal_direction[uav_id] *= -1
                        action_index = vertical_move(pos[1], vertical_direction[uav_id], fov)
                        # 垂直到达边界
                        # if action_index == 0:
                            # 需要判断具体是哪个边界
        
                # # 计算未覆盖区域
                # for i in range(80):
                #     for j in range(80):
                #         # 计算点到当前无人机的距离
                #         distance = np.linalg.norm(np.array([i, j]) - current_position)

                #         # 如果距离小于 FOV，则认为该点被覆盖
                #         if distance < fov:
                #             uncovered_area[i, j] = 0

                # # 找到未覆盖区域中距离当前位置最远的点
                # max_distance = 0
                # target_position = None

                # for i in range(80):
                #     for j in range(80):
                #         if uncovered_area[i, j] == 1:
                #             distance = np.linalg.norm(np.array([i, j]) - current_position)
                #             if distance > max_distance:
                #                 max_distance = distance
                #                 target_position = np.array([i, j])

                # # 计算移动方向
                # direction = target_position - current_position

                # # 选择最接近的动作
                # action_index = np.argmin([np.linalg.norm(direction - action[:2]) for action in uav_action_lst])

                # # 找到所有合法动作的索引
                # valid_action_indices = [i for i, action in enumerate(uav_action_lst) if is_valid_action([pos[0], pos[1]], action, [80, 80])]
                
                # if not valid_action_indices:
                #     # 如果没有合法动作，则选择原地不动
                #     action_index = uav_action_lst.index((0, 0, 0))
                # elif action_index not in valid_action_indices:
                #     # 如果选择的动作没在合法动作中，在合法动作中随机选择一个
                #     action_index = np.random.choice(valid_action_indices)
            actions.append(action_index)


        # 执行动作并获取下一个观测值
        obs, rewards, _, _, _ = env.step(np.array(actions))
        frames.append(env.render())
        episode_reward += rewards
        print(np.array(actions))
        uav_actions.append(actions)


    path = '~/VS_Project/pic/bot2'
    imageio.mimsave(path +'.gif', frames, duration=20)
    return uav_actions

env = gymnasium.make("active-search-v0", custom_config=mcfg)
uav_actions = rule_based_algorithm(num_uavs, uav_action_lst, env, max_steps)
# print(uav_actions)

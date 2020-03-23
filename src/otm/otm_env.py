import numpy as np
from OTM4RL import OTM4RL
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as pltc
from random import sample

class otmEnvDiscrete:

    def __init__(self, env_info, configfile):

        self.otm4rl = OTM4RL(configfile)
        self.num_states = env_info["num_states"]
        self.num_actions = env_info["num_actions"]
        self.controllers = self.otm4rl.get_controller_infos()
        self.num_intersections = len(self.controllers)
        self.action_space = range(self.num_actions ** self.num_intersections)
        self.state_space = range(self.num_states ** (self.num_intersections * 2))
        self.max_queues = self.otm4rl.get_max_queues()
        self.time_step = env_info["time_step"]
        # self.seed()

    # def seed(self, seed=None):
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def encode_state(self, state):
        encoded_state = 0
        state_vec = []
        road_connection_info = self.otm4rl.get_road_connection_info()
        i = 0
        for c_id, controller in self.controllers.items():
            stages = controller["stages"]
            for stage in stages:
                in_link_ids = []
                agg_queue = 0
                max_queue = 0
                phase_ids = stage["phases"]
                for phase_id in phase_ids:
                    road_connections = self.otm4rl.get_signals()[c_id]["phases"][phase_id]["road_conns"]
                    for road_connection in road_connections:
                        in_link_ids.append(road_connection_info[road_connection]["in_link"])
                in_link_ids = list(set(in_link_ids))
                for link_id in in_link_ids:
                    agg_queue += state[link_id]["waiting"]
                    max_queue += self.max_queues[link_id]
                encoded_stage_state = int(agg_queue * self.num_states / max_queue) if agg_queue != max_queue else self.num_states - 1
                state_vec.append(encoded_stage_state)
                encoded_state += encoded_stage_state * (self.num_states ** i)
                i += 1
        state_vec.reverse()
        return encoded_state, np.array(state_vec)

    def decode_action(self, action):
        a = action
        signal_command = dict(list(zip(self.controllers.keys(), np.zeros(self.num_intersections).astype(int))))
        i = self.num_intersections - 1
        while a != 0:
            controller_id = list(self.controllers.keys())[i]
            signal_command[controller_id] = a % self.num_actions
            a = a // self.num_actions
            i -= 1

        return signal_command

    def set_state(self, state):
        self.otm4rl.set_queues(state)
        self.state = self.encode_state(state)

    def reset(self):
         state = self.max_queues.copy()
         in_links = set([rc_info["in_link"] for rc_info in self.otm4rl.get_road_connection_info().values()])
         out_links = set([rc_info["out_link"] for rc_info in self.otm4rl.get_road_connection_info().values()])
         out_links = list(out_links - in_links)
         for link_id in state.keys():
            if link_id in out_links:
                state[link_id] = {"waiting": int(0), "transit": int(0)}
            else:
                p = np.random.random()
                transit_queue = p*state[link_id]
                q = np.random.random()
                waiting_queue = q*(state[link_id] - transit_queue)
                state[link_id] = {"waiting": round(waiting_queue), "transit": round(transit_queue)}
         self.otm4rl.initialize()
         self.set_state(state)
         return self.state

    def step(self, action, render = False):
        assert action in self.action_space, "%r (%s) invalid" % (action, type(action))

        self.otm4rl.set_control(self.decode_action(action))

        self.otm4rl.advance(self.time_step)

        next_state = self.otm4rl.get_queues()

        self.state, state_vec = self.encode_state(next_state)
        reward = -state_vec.sum()

        return self.state, reward

    def plot_queues(self, queue_dynamics, signal_dynamics, link_id):
        # Ex: queue_dynamics = {1: {"waiting": [20, 30, 40, 50, 40, 30],
        #                           "transit": [100, 110, 100, 99, 90, 95]}
        #                      }
        # Ex: signal_dynamics = {1: [0, 0, 1], 2: [1, 0, 0], 3: [1, 1, 1]}
        # plot a graph of number of vehicles in waiting queue over time, given a link_id
        # plot a vetical line: green if the signal turned green and red otherwise
        pass

    def plot_environment(self):
        fig, ax = plt.subplots()

        nodes = {}
        for node_id in self.otm4rl.otmwrapper.otm.scenario().get_node_ids():
            node_info = self.otm4rl.otmwrapper.otm.scenario().get_node_with_id(node_id)
            nodes[node_id] = {'x': node_info.getX(), 'y': node_info.getY()}

        lines = []
        norms = []
        minX = float('Inf')
        maxX = -float('Inf')
        minY = float('Inf')
        maxY = -float('Inf')

        state = self.otm4rl.get_queues()

        for link_id in self.otm4rl.get_link_ids():
            link_info = self.otm4rl.otmwrapper.otm.scenario().get_link_with_id(link_id)

            start_point = nodes[link_info.getStart_node_id()]
            end_point = nodes[link_info.getEnd_node_id()]

            x0 = start_point['x']
            y0 = start_point['y']
            x1 = end_point['x']
            y1 = end_point['y']

            if x1-x0 > 0:
                y0 -= 150
                y1 -= 150

            if x1-x0 < 0:
                y0 += 150
                y1 += 150

            if y1-y0 > 0:
                x0 += 100
                x1 += 100

            if y1-y0 < 0:
                x0 -= 100
                x1 -= 100

            p0 = (x0, y0)
            p1 = (x1, y1)

            lines.append([p0, p1])
            norms.append(state[link_id]["waiting"]/self.max_queues[link_id])

            minX = min([minX, p0[0], p1[0]])
            maxX = max([maxX, p0[0], p1[0]])
            minY = min([minY, p0[1], p1[1]])
            maxY = max([maxY, p0[1], p1[1]])

        cmap = plt.get_cmap('Wistia')
        all_colors = [cmap(z) for z in norms]
        lc = LineCollection(lines, colors = all_colors)
        lc.set_linewidths(15)
        ax.add_collection(lc)

        dY = maxY - minY
        dX = maxX - minX

        if (dY > dX):
            ax.set_ylim((minY, maxY))
            c = (maxX + minX) / 2
            ax.set_xlim((c - dY / 2, c + dY / 2))
        else:
            ax.set_xlim((minX, maxX))
            c = (maxY + minY) / 2
            ax.set_ylim((c - dX / 2, c + dX / 2))

        return plt
        # color gradient for links
        # plot traffic lights
        # show time

    # def render(self, mode='human'):
    #     #plot the queue profile over time
    #     #render the network
    #     pass
    #
    # def close(self):
    #     #stop rendering
    #     pass

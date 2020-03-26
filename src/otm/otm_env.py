import numpy as np
from OTM4RL import OTM4RL
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as pltc
from random import sample

class otmEnvDiscrete:

    def __init__(self, env_info, configfile):

        self.time_step = env_info["time_step"]
        self.plot_precision = env_info["plot_precision"]

        assert (type(self.plot_precision) == int and self.plot_precision >= 1), "plot_precision must be an integer greater than or equal to 1"

        self.otm4rl = OTM4RL(configfile)
        self.num_states = env_info["num_states"]
        self.num_actions = env_info["num_actions"]
        self.controllers = self.otm4rl.get_controller_infos()
        self.num_intersections = len(self.controllers)
        self.action_space = range(self.num_actions ** self.num_intersections)
        self.state_space = range(self.num_states ** (self.num_intersections * 2))
        self.max_queues = self.otm4rl.get_max_queues()
        self.buffer = env_info["buffer"]
        self.queue_buffer = dict(list(zip(self.otm4rl.get_link_ids(), [{"waiting": [], "transit": []} for i in self.otm4rl.get_link_ids()])))
        self.signal_buffer = dict(list(zip(self.controllers.keys(), [[] for i in self.controllers.keys()])))
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
         self.add_queue_buffer()

         return self.state

    def step(self, action):
        assert action in self.action_space, "%r (%s) invalid" % (action, type(action))

        self.otm4rl.set_control(self.decode_action(action))
        self.add_signal_buffer()

        self.otm4rl.advance(self.time_step)

        next_state = self.otm4rl.get_queues()
        self.add_queue_buffer()

        self.state, state_vec = self.encode_state(next_state)
        reward = -state_vec.sum()

        return self.state, reward

    def add_queue_buffer(self):

        if self.buffer == True:
            queues = self.otm4rl.get_queues()
            for link_id in queues.keys():
                self.queue_buffer[link_id]["waiting"].append(queues[link_id]["waiting"])
                self.queue_buffer[link_id]["transit"].append(queues[link_id]["transit"])
        else:
            pass

    def add_signal_buffer(self):

        if self.buffer == True:
            signals = self.otm4rl.get_control()
            for c_id in signals:
                self.signal_buffer[c_id].append(signals[c_id])
        else:
            pass

    def plot_queues(self, link_id, queue_type, from_time = 0, to_time = 10):

        road_connection_info = self.otm4rl.get_road_connection_info()

        link_rc = []
        link_controller = None
        link_stages = []
        for rc, rc_info in road_connection_info.items():
            if link_id == rc_info["in_link"]:
                link_rc.append(rc)

        for c_id in self.controllers.keys():
            for stage in range(len(self.controllers[c_id]["stages"])):
                phase_ids = self.controllers[c_id]["stages"][stage]["phases"]
                for phase_id in phase_ids:
                    road_connections = self.otm4rl.get_signals()[c_id]["phases"][phase_id]["road_conns"]
                    if set.intersection(set(link_rc),set(road_connections)) != set():
                        link_stages.append(stage)
            if len(link_stages) != 0:
                link_controller = c_id
                break

        if link_controller == None:
            print("This link is leaving the network or it is a demand link, so it is not impacted by traffic lights")
            return

        fig, ax = plt.subplots()
        queues = self.queue_buffer[link_id][queue_type]
        step = self.time_step/self.plot_precision
        ax.plot([i*step for i in range(len(queues))], queues)

        stages = np.array(self.signal_buffer[link_controller])
        stage_times = np.array(range(len(stages)))*self.time_step
        aux = np.array([stages[i] if (i == 0 or stages[i-1] != stages[i]) else -1 for i in range(len(stages))])
        changing_stages = np.array([aux[i] if (i == 0 or aux[i] in link_stages or (aux[i-1] in link_stages and aux[i] not in link_stages)) else -1 for i in range(len(aux))])
        stages = np.extract(changing_stages >= 0, stages)
        stage_times = np.extract(changing_stages >=0, stage_times)
        colors = ["g" if stages[i] in link_stages else "r" for i in range(len(stages))]
        for i in range(len(colors)):
            ax.axvline(x=stage_times[i], color = colors[i])
            y = (ax.get_ylim()[1] - ax.get_ylim()[0])*0.96 + ax.get_ylim()[0]
            ax.text(stage_times[i] + 0.05*self.time_step, y, stages[i] if stages[i] in link_stages else "")

        plt.title("Link " + str(link_id) + " - Queue dynamics (" + queue_type + " queue)")
        plt.show()

    def build_network_lines(self, state):

        nodes = {}
        for node_id in self.otm4rl.get_node_ids():
            node_info = self.otm4rl.get_node_with_id(node_id)
            nodes[node_id] = {'x': node_info.getX(), 'y': node_info.getY()}

        lines = []
        norms = []
        minX = float('Inf')
        maxX = -float('Inf')
        minY = float('Inf')
        maxY = -float('Inf')

        for link_id in self.otm4rl.get_link_ids():
            link_info = self.otm4rl.get_link_with_id(link_id)

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

        return lines, norms, minX, maxX, minY, maxY

    def get_signal_positions(self, lines, control):

        link_coords = dict(zip(self.otm4rl.get_link_ids(), lines))
        road_connection_info = self.otm4rl.get_road_connection_info()
        signal_positions = dict()
        for c_id, stage in control.items():
            phase_ids = self.controllers[c_id]["stages"][stage]["phases"]
            for phase_id in phase_ids:
                road_connections = self.otm4rl.get_signals()[c_id]["phases"][phase_id]["road_conns"]
                for road_connection in road_connections:
                    in_link_id = road_connection_info[road_connection]["in_link"]
                    out_link_id = road_connection_info[road_connection]["out_link"]
                    signal_positions[road_connection] = {"in_link": link_coords[in_link_id], "out_link": link_coords[out_link_id]}
        return signal_positions

    def plot_environment(self, state, control):
        fig, ax = plt.subplots()

        lines, norms, minX, maxX, minY, maxY = self.build_network_lines(state)

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

        signal_positions = self.get_signal_positions(lines, control)

        for rc in signal_positions.values():
            p0 = rc["in_link"][0]
            p1 = rc["in_link"][1]
            ax.annotate(s='', xy=p1, xytext=p0, arrowprops=dict(arrowstyle='-'))
            p0 = rc["out_link"][0]
            p1 = rc["out_link"][1]
            ax.annotate(s='', xy=p1, xytext=p0, arrowprops=dict(arrowstyle='->'))

        plt.show()
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

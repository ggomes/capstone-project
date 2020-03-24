import os
import inspect
import numpy as np
from otm_env import otmEnvDiscrete

def get_config():
	this_folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
	root_folder = os.path.dirname(os.path.dirname(this_folder))
	configfile = os.path.join(root_folder,'cfg', 'network_v6.xml')
	return configfile

def get_env():
    return otmEnvDiscrete({"num_states": 2, "num_actions": 2, "time_step": 200}, get_config())

def test_decode_action():
    env = get_env()
    for i in range(8):
    	print(env.decode_action(i))
    del env

def test_encode_state():
    env = get_env()
    env.otm4rl.initialize()
    env.otm4rl.advance(600)
    state = env.otm4rl.get_queues()
    print(state)
    print(env.encode_state(state))
    del env

def test_set_state():
	env = get_env()
	env.otm4rl.initialize()
	print(env.otm4rl.get_queues())
	state = env.otm4rl.get_max_queues()
	in_links = set([rc_info["in_link"] for rc_info in env.otm4rl.get_road_connection_info().values()])
	out_links = set([rc_info["out_link"] for rc_info in env.otm4rl.get_road_connection_info().values()])
	out_links = list(out_links - in_links)
	for link_id in state.keys():
		if link_id in out_links:
			state[link_id] = {"waiting": int(0), "transit": int(0)}
		else:
			p = np.random.random()
			q = np.random.random()
			state[link_id] = {"waiting": round(p*state[link_id]), "transit": round(q*(1-p)*state[link_id])}
	print(state)
	env.set_state(state)
	print(env.state)
	print(env.otm4rl.get_queues())
	del env

def test_reset():
	env = get_env()
	print(env.reset())
	del env

def test_step():
	env = get_env()

	env.reset()
	print("Initial state:", env.encode_state(env.otm4rl.get_queues()))
	print(env.otm4rl.get_queues())

	action = np.random.choice(env.action_space)
	print("Action 1: ", env.decode_action(action))
	state, reward = env.step(action)
	print("Next state:", env.encode_state(env.otm4rl.get_queues()))
	print("Reward:", reward)
	print(env.otm4rl.get_queues())

	action = np.random.choice(env.action_space)
	print("Action 1: ", env.decode_action(action))
	state, reward = env.step(action)
	print("Next state:", env.encode_state(env.otm4rl.get_queues()))
	print("Reward:", reward)
	print(env.otm4rl.get_queues())

	del env

def test_get_signal_positions():
	env = get_env()
	env.reset()
	env.otm4rl.advance(100)
	control = env.otm4rl.get_control()
	state = env.otm4rl.get_queues()
	lines = env.build_network_lines(state)[0]
	print(env.get_signal_positions(lines, control))

def test_plot_environment():
	env = get_env()

	env.reset()
	action = np.random.choice(env.action_space)
	state, reward = env.step(action)
	action = np.random.choice(env.action_space)
	state = env.otm4rl.get_queues()
	print(env.decode_action(action))
	env.plot_environment(state, env.decode_action(action)).show()
	state, reward = env.step(action)
	action = np.random.choice(env.action_space)
	state = env.otm4rl.get_queues()
	print(env.decode_action(action))
	env.plot_environment(state, env.decode_action(action)).show()
	state, reward = env.step(action)
	action = np.random.choice(env.action_space)
	state = env.otm4rl.get_queues()
	print(env.decode_action(action))
	env.plot_environment(state, env.decode_action(action)).show()

if __name__ == '__main__':
	# test_encode_state()
	# test_decode_action()
	# test_set_state()
	# test_reset()
	# test_step()
	# test_get_signal_positions()
	test_plot_environment()

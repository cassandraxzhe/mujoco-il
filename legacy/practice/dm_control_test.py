from dm_control import suite

# Load cartpole task
env = suite.load('cartpole', 'swingup')
time_step = env.reset()

for _ in range(10):
    action = env.action_spec().generate_value()
    time_step = env.step(action)
    print(time_step.reward, time_step.observation)

import time 
import ur5_push_env
env = ur5_push_env.ur5(render_mode="human")
env.reset()

while True:
    env.render()
    time.sleep(1/30)
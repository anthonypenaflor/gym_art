import numpy as np
import time
from quadrotor import QuadrotorEnv

def main():
    # Create the environment with NonlinearPositionController
    env = QuadrotorEnv(
        dynamics_params="DefaultQuad",  # Use default quadrotor parameters
        raw_control=False,  # Use NonlinearPositionController instead of raw control
        dim_mode='3D',     # Full 3D control
        sim_freq=200.,     # Simulation frequency
        ep_time=10.,       # Episode time in seconds
        room_size=10,      # Room size
        init_random_state=True,  # Start with random state
        obs_repr="xyz_vxyz_R_omega"  # Observation representation
    )

    # Set a custom destination (x, y, z coordinates)
    destination = np.array([2.0, 1.0, 1.5])
    env.set_destination(destination)

    # Reset the environment
    state = env.reset()

    # Run the environment
    done = False
    episode_reward = 0
    step_count = 0

    while not done:
        # First, let the controller compute the action
        env.controller.step_func(
            dynamics=env.dynamics,
            action=None,  # We don't need to provide an action
            goal=env.goal,
            dt=env.dt,
            observation=np.expand_dims(env.state_vector(env), axis=0)
        )
        
        # Now we can use the computed action
        state, reward, done, info = env.step(env.controller.action)
        
        # Accumulate reward
        episode_reward += reward
        
        # Print information about the current state
        print(f"Step {step_count}")
        print(f"Position: {env.dynamics.pos}")
        print(f"Distance to destination: {info['distance_to_destination']:.2f}")
        print(f"Destination reached: {info['destination_reached']}")
        print(f"Reward: {reward:.2f}")
        print("---")
        
        # Optional: Render the environment
        env.render()
        
        # Small delay to make visualization more manageable
        # time.sleep(0.01)
        
        step_count += 1

    print(f"Episode finished after {step_count} steps")
    print(f"Total reward: {episode_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()
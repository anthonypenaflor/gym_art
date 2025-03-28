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
        ep_time=60.,       # Increased episode time to 60 seconds
        room_size=10,      # Room size
        init_random_state=True,  # Start with random state
        obs_repr="xyz_vxyz_R_omega"  # Observation representation
    )

    # Define multiple destination points (x, y, z coordinates) with more spacing
    destinations = [
        np.array([3.0, 2.0, 2.0]),    # First destination - far right and up
        np.array([-2.0, -2.0, 1.5]),  # Second destination - far left and down
        np.array([0.0, 0.0, 1.0]),    # Third destination - center
        np.array([2.0, -2.0, 2.0])    # Fourth destination - right and down
    ]
    
    # Define destination tolerance (in meters)
    destination_tolerance = 0.1  # 10cm tolerance
    
    # Reset environment once at the start
    state = env.reset()
    
    # Set initial destination
    current_dest_idx = 0
    env.set_destination(destinations[current_dest_idx])
    
    # Run the environment
    done = False
    total_reward = 0
    step_count = 0
    max_steps = int(env.ep_time / env.dt)  # Maximum steps per episode
    stabilization_steps = 50  # Number of steps to wait after reaching a destination
    
    print(f"\nStarting navigation to destination {current_dest_idx + 1}: {destinations[current_dest_idx]}")
    print(f"Maximum steps per episode: {max_steps}")
    print(f"Destination tolerance: {destination_tolerance}m")
    print(f"Stabilization steps between destinations: {stabilization_steps}")
    
    stabilization_counter = 0
    while current_dest_idx < len(destinations):
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
        total_reward += reward
        
        # Print information about the current state
        print(f"Step {step_count}")
        print(f"Position: {env.dynamics.pos}")
        print(f"Distance to destination: {info['distance_to_destination']:.2f}")
        print(f"Destination reached: {info['distance_to_destination'] < destination_tolerance}")
        print(f"Reward: {reward:.2f}")
        print("---")
        
        # Optional: Render the environment
        env.render()
        
        step_count += 1
        
        # Check if current destination is reached (using tolerance)
        if info['distance_to_destination'] < destination_tolerance:
            if stabilization_counter == 0:
                print(f"\nDestination {current_dest_idx + 1} reached!")
                print(f"Final distance: {info['distance_to_destination']:.3f}m")
                stabilization_counter = stabilization_steps
            else:
                stabilization_counter -= 1
                if stabilization_counter == 0:
                    # Move to next destination if available
                    current_dest_idx += 1
                    if current_dest_idx < len(destinations):
                        # Just update the destination without resetting
                        env.set_destination(destinations[current_dest_idx])
                        print(f"Proceeding to destination {current_dest_idx + 1}: {destinations[current_dest_idx]}")
                    else:
                        print("\nAll destinations reached successfully!")
                        break
        
        # Check if we've exceeded maximum steps
        if step_count >= max_steps:
            print(f"\nEpisode ended due to maximum steps ({max_steps})")
            print(f"Current destination {current_dest_idx + 1} not reached")
            print(f"Final distance to destination: {info['distance_to_destination']:.2f}")
            break
    
    print(f"\nEpisode finished!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final destination index: {current_dest_idx + 1}/{len(destinations)}")
    env.close()

if __name__ == "__main__":
    main()
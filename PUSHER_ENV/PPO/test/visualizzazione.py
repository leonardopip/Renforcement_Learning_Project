import gymnasium as gym
import time
import sys
import os
import logging
from stable_baselines3 import PPO

# =================================================================
# 1. ENVIRONMENT AND PATH CONFIGURATION
# =================================================================

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import PUSHER_ENV.PPO.pusher_PPO
    logging.info("Module 'PUSHER_ENV' successfully imported.")
except ImportError as e:
    logging.error(f"Failed to import local modules: {e}")
    sys.exit(1)

# =================================================================
# 2. RESOURCE DISCOVERY
# =================================================================

def find_file_recursive(target_filename, search_directory):
    """
    Traverses the directory tree to locate a specific filename.
    """
    for root, _, files in os.walk(search_directory):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None

# Locate XML configuration file
XML_FILENAME = "pusher-v2.xml"
XML_PATH = find_file_recursive(XML_FILENAME, project_root)

if not XML_PATH:
    logging.error(f"Critical resource missing: {XML_FILENAME} not found.")
    sys.exit(1)

# CORRECTED: Use only the FILENAME for the recursive search
MODEL_FILENAME = "PPO_Pusher-MassFric-Uni-20-v0_s90161.zip"
MODEL_PATH = find_file_recursive(MODEL_FILENAME, project_root)

if not MODEL_PATH:
    logging.error(f"Model weights not found for filename: {MODEL_FILENAME}")
    sys.exit(1)

logging.info(f"XML Path: {XML_PATH}")
logging.info(f"Model Path: {MODEL_PATH}")

# =================================================================
# 3. SIMULATION EXECUTION
# =================================================================

def run_inference_test():
    env = None
    try:
        # Note: Ensure "Pusher-Target-Mass-Easy-v0" matches your registration
        env = gym.make(
            "Pusher-Target-Mass-Medium-v0", 
            render_mode="human", 
            xml_file=XML_PATH
        )
        
        model = PPO.load(MODEL_PATH)
        observation, info = env.reset()
        
        logging.info("Simulation initiated. Press Ctrl+C to terminate.")
        
        while True:
            action, _states = model.predict(observation, deterministic=True)
            
            # DIAGNOSTIC: Uncomment these to see if the model is outputting zero or if rewards are flat
            # logging.info(f"Action: {action} | Obs Sample: {observation[:3]}")
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            time.sleep(0.01)
            
            if terminated or truncated:
                logging.info(f"Episode Finished. Final Reward: {reward}")
                observation, info = env.reset()

    except KeyboardInterrupt:
        logging.info("Simulation terminated by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if env is not None:
            env.close()
            logging.info("Environment resources released.")

if __name__ == "__main__":
    run_inference_test()
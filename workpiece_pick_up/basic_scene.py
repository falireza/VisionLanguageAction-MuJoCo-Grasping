import mujoco
import mujoco.viewer
import time
import numpy as np

google_robot_xml_path = r"C:\Users\alrfa\OneDrive - Eotvos Lorand Tudomanyegyetem Informatikai Kar\VLAs\google_robot\scene.xml"

# 2. Load the model from the XML file path
try:
    print(f"Attempting to load model from: {google_robot_xml_path}")
    # Use from_xml_path() when loading from a file path
    model = mujoco.MjModel.from_xml_path(google_robot_xml_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the following:")
    print(f"1. The path '{google_robot_xml_path}' is correct and points to a valid MuJoCo XML/MJCF model file.")
    print("2. The model file and any associated assets (like meshes, usually in an 'assets' folder nearby or referenced correctly in the XML) are correctly located and accessible.")
    exit()

# 3. Create the data structure for the simulation state
data = mujoco.MjData(model)
print("Data structure created for the loaded model.")

# 4. Launch the interactive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
  start_time = time.time()
  # Limit simulation time for this example
  simulation_duration_seconds = 300
  print(f"Simulation will run for approximately {simulation_duration_seconds} seconds or until viewer is closed.")

  while viewer.is_running() and (time.time() - start_time < simulation_duration_seconds):
    step_start_time = time.time()

    # Advance the simulation by one step
    mujoco.mj_step(model, data)

    # Synchronize the viewer with the new simulation state
    viewer.sync()

    # Optional: Regulate the simulation speed to be closer to real-time
    time_until_next_step = model.opt.timestep - (time.time() - step_start_time)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)

  if not viewer.is_running():
      print("Viewer was closed.")
  else:
      print(f"Simulation time of {simulation_duration_seconds} seconds ended.")

print("Script finished.")
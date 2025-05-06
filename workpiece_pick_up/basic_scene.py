import mujoco
import mujoco.viewer
import time
import numpy as np # Often useful, though not strictly for this basic scene

# 1. Define the MJCF model as an XML string
# MJCF (MuJoCo XML Format) is how scenes and objects are defined.
# This is a very simple model: a ground plane and a red box.
xml_model_string = """
<mujoco>
  <worldbody>
    <body>
      <geom name="ground" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"/>
    </body>
    <body>
      <joint name="free_joint" type="free"/> <geom name="box" type="box" size="0.1 0.1 0.1" rgba="1 0 0 1" pos="0 0 1" mass="0.1"/>
    </body>
  </worldbody>
  <actuator>
      </actuator>
</mujoco>
"""

# 2. Load the model from the XML string
try:
    model = mujoco.MjModel.from_xml_string(xml_model_string)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 3. Create the data structure for the simulation state
data = mujoco.MjData(model)
print("Data structure created.")

# 4. Launch the interactive viewer
# This will open a window and run the simulation.
# 'launch_passive' is one way to do this; it gives you a handle to the viewer.
print("Launching viewer...")
with mujoco.viewer.launch_passive(model, data) as viewer:
  # Keep the simulation running while the viewer is open
  start_time = time.time()
  while viewer.is_running() and (time.time() - start_time < 30): # Run for 30 seconds
    step_start_time = time.time()

    # Advance the simulation by one step
    mujoco.mj_step(model, data)

    # Synchronize the viewer with the new simulation state
    # This updates what you see in the window.
    viewer.sync()

    # Optional: Regulate the simulation speed (roughly)
    # MuJoCo's default timestep is 0.002s (500Hz).
    # viewer.sync() can sometimes slow things down appropriately,
    # but you can add a small sleep if it runs too fast for viewing.
    time_until_next_step = model.opt.timestep - (time.time() - step_start_time)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)

  print("Simulation time ended or viewer closed.")

print("Script finished.")
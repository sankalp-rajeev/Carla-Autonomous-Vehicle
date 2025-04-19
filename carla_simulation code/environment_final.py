'''
Credits : https://github.com/vadim7s/SelfDrive/blob/master/RL_Full_Tutorial
Modified to include Adaptive Cruise Control functionality and RGB camera for object detection
'''

import random
import time
import numpy as np
import math 
import cv2
import gymnasium
from gymnasium import spaces
import carla
from keras.models import load_model

SECONDS_PER_EPISODE = 15

N_CHANNELS = 3
HEIGHT = 240
WIDTH = 320

SPIN = 10 #angle of random spin

HEIGHT_REQUIRED_PORTION = 0.5 #bottom share, e.g. 0.1 is take lowest 10% of rows
WIDTH_REQUIRED_PORTION = 0.9

SHOW_PREVIEW = True

SEED = 123

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0

    def compute(self, error, dt):
        # Proportional term
        p_term = self.Kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        d_term = self.Kd * derivative
        
        # Update previous error
        self.previous_error = error
        
        # Calculate total output
        output = p_term + i_term + d_term
        
        return output

class CarEnv(gymnasium.Env):
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = WIDTH
    im_height = HEIGHT
    front_camera = None
    rgb_camera_image = None  # Added for RGB camera image
    CAMERA_POS_Z = 1.3 
    CAMERA_POS_X = 1.4
    PREFERRED_SPEED = 30 # what it says
    SPEED_THRESHOLD = 2 #defines when we get close to desired speed so we drop the
    
    # ACC parameters
    ACC_ACTIVE = True
    SAFE_DISTANCE = 10.0  # Safe distance in meters
    MIN_DISTANCE = 5.0    # Minimum distance before emergency braking
    
    # New control smoothing parameters
    LAST_THROTTLE = 0.0
    LAST_BRAKE = 0.0
    MAX_THROTTLE_CHANGE = 0.2  # Max change per step
    MAX_BRAKE_CHANGE = 0.3     # Max change per step
    DISTANCE_FILTER_ALPHA = 0.3  # For smoothing distance measurements
    FILTERED_DISTANCE = float('inf')
    
    def __init__(self):
        super(CarEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([9, 3])
        self.height_from = int(HEIGHT * (1 -HEIGHT_REQUIRED_PORTION))
        self.width_from = int((WIDTH - WIDTH * WIDTH_REQUIRED_PORTION) / 2)
        self.width_to = self.width_from + int(WIDTH_REQUIRED_PORTION * WIDTH)
        self.new_height = HEIGHT - self.height_from
        self.new_width = self.width_to - self.width_from
        self.image_for_CNN = None
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(7, 18, 8), dtype=np.float32)
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = not self.SHOW_CAM
        self.world.apply_settings(self.settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.cnn_model = load_model('model_saved_from_CNN.h5',compile=False)
        self.cnn_model.compile()
        if self.SHOW_CAM:
            self.spectator = self.world.get_spectator()

        # Initialize Traffic Manager for spawn vehicles
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        self.step_counter = 0
        
        # Initialize ACC controller
        self.speed_controller = PIDController(0.5, 0.1, 0.05)
        self.distance_controller = PIDController(0.5, 0.02, 0.1)
        self.last_time = time.time()

    def _smooth_transition(self, current, target, max_change):
        """Helper method to smooth transitions between control values"""
        difference = target - current
        if abs(difference) > max_change:
            return current + math.copysign(max_change, difference)
        return target

    def cleanup(self):
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        cv2.destroyAllWindows()
        if hasattr(self, 'lead_vehicle') and self.lead_vehicle is not None:
            self.lead_vehicle.destroy()
    
    def get_distance_to_lead_vehicle(self):
        if not hasattr(self, 'lead_vehicle') or self.lead_vehicle is None:
            return float('inf')
        
        try:
            ego_location = self.vehicle.get_location()
            lead_location = self.lead_vehicle.get_location()
        
            # Calculate Euclidean distance
            distance = math.sqrt(
                (ego_location.x - lead_location.x) ** 2 +
                (ego_location.y - lead_location.y) ** 2
            )
        
            return distance
        except RuntimeError:
            # The lead vehicle was likely destroyed
            self.lead_vehicle = None
            return float('inf')
           
    def get_relative_speed(self):
        if not hasattr(self, 'lead_vehicle') or self.lead_vehicle is None:
            return 0.0
        
        try:
            ego_velocity = self.vehicle.get_velocity()
            lead_velocity = self.lead_vehicle.get_velocity()
        
            ego_speed = math.sqrt(ego_velocity.x ** 2 + ego_velocity.y ** 2)
            lead_speed = math.sqrt(lead_velocity.x ** 2 + lead_velocity.y ** 2)
        
            # Positive if ego is faster, negative if lead is faster
            return ego_speed - lead_speed
        except RuntimeError:
            # The lead vehicle was likely destroyed
            self.lead_vehicle = None
            return 0.0

    def maintain_speed(self, current_speed):
        ''' 
        Enhanced function to maintain desired speed and safe distance
        with smoother transitions and filtered distance measurements
        '''
        # Get raw distance and apply simple low-pass filter
        raw_distance = self.get_distance_to_lead_vehicle()
        self.FILTERED_DISTANCE = (self.DISTANCE_FILTER_ALPHA * raw_distance + 
                                 (1 - self.DISTANCE_FILTER_ALPHA) * self.FILTERED_DISTANCE)
        distance = self.FILTERED_DISTANCE
        
        relative_speed = self.get_relative_speed()
        
        # Add small delay to prevent over-reaction
        time.sleep(0.05)  # Small delay to prevent over-reaction
        
        # Default to target speed if no lead vehicle or it's far away
        if distance > 50.0 or not hasattr(self, 'lead_vehicle') or self.lead_vehicle is None:
            # Regular speed control with smoother transitions
            if current_speed >= self.PREFERRED_SPEED:
                new_throttle = 0.2
                new_brake = 0.0
            elif current_speed < 5.0:
                new_throttle = 0.8
                new_brake = 0.0
            elif current_speed < self.PREFERRED_SPEED - self.SPEED_THRESHOLD:
                new_throttle = 0.7
                new_brake = 0.0
            else:
                new_throttle = 0.3
                new_brake = 0.0
        else:
            # ACC logic with smoother transitions
            desired_distance = max(self.MIN_DISTANCE, 2.0 * current_speed / 3.6)
            distance_error = distance - desired_distance
            
            # Emergency braking if too close
            if distance < self.MIN_DISTANCE:
                new_throttle = 0.0
                new_brake = 1.0
            elif distance < 0.7 * desired_distance:
                new_throttle = 0.0
                new_brake = 0.5
            elif distance < 0.9 * desired_distance:
                new_throttle = 0.0
                new_brake = 0.3
            elif 0.9 * desired_distance < distance < 1.1 * desired_distance:
                if relative_speed < 2.0:
                    new_throttle = 0.0
                    new_brake = 0.2
                elif relative_speed > 2.0:
                    new_throttle = 0.4
                    new_brake = 0.0
                else:
                    new_throttle = 0.2
                    new_brake = 0.0
            elif distance > 1.5 * desired_distance:
                if current_speed < 5.0:
                    new_throttle = 0.9
                    new_brake = 0.0
                else:
                    new_throttle = 0.7
                    new_brake = 0.0
            else:
                new_throttle = 0.4
                new_brake = 0.0
        
        # Smooth throttle and brake transitions
        self.LAST_THROTTLE = self._smooth_transition(self.LAST_THROTTLE, new_throttle, self.MAX_THROTTLE_CHANGE)
        self.LAST_BRAKE = self._smooth_transition(self.LAST_BRAKE, new_brake, self.MAX_BRAKE_CHANGE)
        
        return self.LAST_THROTTLE, self.LAST_BRAKE

    def apply_cnn(self, im):
        img = np.float32(im)
        img = img /255
        img = np.expand_dims(img, axis=0)
        aux_input = np.array([[0]], dtype=np.float32)  
        cnn_applied = self.cnn_model([img, aux_input], training=False)
        cnn_applied = np.squeeze(cnn_applied)
        return cnn_applied

    def step(self, action):
        self.world.tick()  # Important for sync mode
        self.step_counter += 1  # For PPO callback/debug compatibility
        
        # Calculate delta time for PID controllers
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        trans = self.vehicle.get_transform()
        if self.SHOW_CAM:
            self.spectator.set_transform(carla.Transform(trans.location + carla.Location(z=20),carla.Rotation(yaw =-180, pitch=-90)))

        # Extract RL actions
        steer_action = action[0]
        brake_override = action[1]  # 0: Use ACC, 1: Light brake, 2: Full brake
        
        # Map steering actions from discrete to continuous
        if steer_action == 0:
            steer = - 0.9
        elif steer_action == 1:
            steer = -0.25
        elif steer_action == 2:
            steer = -0.1
        elif steer_action == 3:
            steer = -0.05
        elif steer_action == 4:
            steer = 0.0 
        elif steer_action == 5:
            steer = 0.05
        elif steer_action == 6:
            steer = 0.1
        elif steer_action == 7:
            steer = 0.25
        elif steer_action == 8:
            steer = 0.9

        #  print steer and throttle every 50 steps
        if self.step_counter % 50 == 0:
            print('steer input from model:', steer)
        
        # Get current speed
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        # Get ACC throttle and brake values
        if self.ACC_ACTIVE:
            acc_throttle, acc_brake = self.maintain_speed(kmh)
            
            # Allow RL to override ACC with brake actions
            if brake_override == 1:  # Light brake
                throttle = 0.0
                brake = 0.3
            elif brake_override == 2:  # Full brake
                throttle = 0.0
                brake = 1.0
            else:  # Use ACC
                throttle = acc_throttle
                brake = acc_brake
                
            # Display ACC status
            if self.step_counter % 50 == 0:
                distance = self.get_distance_to_lead_vehicle()
                relative_speed = self.get_relative_speed()
                print(f"ACC: Speed={kmh}km/h, Distance={distance:.1f}m, RelSpeed={relative_speed:.1f}m/s, Throttle={throttle:.2f}, Brake={brake:.2f}")
        else:
            # Original throttle-brake coordination logic if ACC disabled
            estimated_throttle = self.maintain_speed(kmh)
            if brake_override == 2:
                throttle = 0.0
                brake = 1.0
            elif brake_override == 1:
                throttle = 0.0
                brake = 0.3
            else:
                throttle = estimated_throttle
                brake = 0.0

        # Apply vehicle control with smoothed values
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake))
        

        # Small delay to prevent over-control
        time.sleep(0.02)

        self.vehicle.apply_control(control)

        distance_travelled = self.initial_location.distance(self.vehicle.get_location())

        # Store camera to return at the end in case the clean-up function destroys it
        cam = self.front_camera
        rgb_cam = self.rgb_camera_image  # Store RGB image
        
        # Show image
        if self.SHOW_CAM:
            cv2.imshow('Sem Camera', cam)
            if rgb_cam is not None:
                cv2.imshow('RGB Camera', rgb_cam)
            cv2.waitKey(1)

        # Track steering lock duration to prevent "chasing its tail"
        lock_duration = 0
        if self.steering_lock == False:
            if steer<-0.6 or steer>0.6:
                self.steering_lock = True
                self.steering_lock_start = time.time()
        else:
            if steer<-0.6 or steer>0.6:
                lock_duration = time.time() - self.steering_lock_start
        
        # Start defining reward from each step
        reward = 0
        done = False
        
        # Punish for collision
        if len(self.collision_hist) != 0:
            done = True
            reward = reward - 50
            self.cleanup()
            
        if len(self.lane_invade_hist) != 0:
            done = True
            reward = reward - 50
            self.cleanup()
            
        # Punish for steer lock up
        if lock_duration > 3:
            reward = reward - 100
            done = True
            self.cleanup()
        elif lock_duration > 1:
            reward = reward - 20

        # ACC-specific rewards
        if hasattr(self, 'lead_vehicle') and self.lead_vehicle is not None:
            distance_to_lead = self.get_distance_to_lead_vehicle()
            
            #calculate ideal following distance (2-second rule)
            ideal_distance = max(5.0, 2.0 * kmh / 3.6)  # in meters
            distance_error = abs(distance_to_lead - ideal_distance)
            
            #Reward for maintaining perfect distance (inversely proportional to distance error)
            if distance_error < 1.0:
                reward += 20 #perfect following distance
            elif distance_error < 3.0:
                reward += 1.0 #Good following distance
            elif distance_error < 5.0:
                reward += 5.0 #Acceptable following distance
            
            
            # Penalties for unsafe following
            if distance_to_lead < 5.0:
                reward -= 10.0
            if distance_to_lead < 3:
                reward -= 50.0
            if distance_to_lead < 1.5:
                reward -= 80.0
                done = True
                self.cleanup()
            
            # Reward for appropriate brake usage
            if brake > 0.5 and distance_to_lead < ideal_distance - 2.0:
                reward += 2.0  # Good reaction to close distance
                
            # Penalize unnecessary hard braking when far from lead vehicle
            elif brake > 0.7 and distance_to_lead > ideal_distance + 10.0:
                reward -= 3.0  # Unnecessary hard braking

        # Reward for making progress (distance travelled)
        if kmh < 5:
            reward -= 1.0
        elif distance_travelled < 30:
            reward -= 1.0
        elif distance_travelled < 50:
            reward += 1.0
        else:
            reward = reward + 2
            
        # Check for episode duration
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
            self.cleanup()
            
        try:
            self.image_for_CNN = self.apply_cnn(self.front_camera[self.height_from:, self.width_from:self.width_to])
            return self.image_for_CNN, reward, done, done, {}
        except Exception as e:
            print(f"[ERROR] Failed to apply CNN or return observation: {e}")
            return self.observation_space.sample(), -100, True, True, {}

    def reset(self, seed=SEED):
        
        #first clean up existing resources
        try:
            self.cleanup()
        except Exception as e:
            print(f"[WARNING] Cleanup failed: {e}")
        
        self.collision_hist = []
        self.lane_invade_hist = []
        self.actor_list = []
        
        # Get spawn points properly
        spawn_points = self.world.get_map().get_spawn_points()
        
        #try multiple times to spawn the vehicle
        # self.transform = random.choice(self.world.get_map().get_spawn_points())
        attempts = 0
        max_attempts = 10
        
        
        self.vehicle = None
        while self.vehicle is None and attempts < max_attempts:
            try:
                self.transform = random.choice(spawn_points)
                # Connect
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
                break
            except:
                attempts += 1
                time.sleep(0.1)  # Wait a bit before retrying
         
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle after multiple attempts")     
                
        self.actor_list.append(self.vehicle)
        self.initial_location = self.vehicle.get_location()

        # Set spectator to follow from top-down for visibility
        if self.SHOW_CAM:
            spectator = self.world.get_spectator()
            ego_transform = self.vehicle.get_transform()
            top_down = carla.Transform(
                ego_transform.location + carla.Location(z=50),
                carla.Rotation(pitch=-90)
            )
            spectator.set_transform(top_down)

        # Improved lead vehicle spawn with better positioning for ACC testing
        spawn_attempts = 0
        self.lead_vehicle = None
        
        # Get ego vehicle's transform and forward vector
        ego_transform = self.vehicle.get_transform()
        forward_vector = ego_transform.get_forward_vector()
        ego_location = ego_transform.location
        
        # Try to spawn lead vehicle directly in front of ego vehicle first
        for test_distance in [20, 15, 25, 30]:
            try:
                # Calculate spawn position in front of ego vehicle
                spawn_location = ego_location + carla.Location(
                    x=forward_vector.x * test_distance,
                    y=forward_vector.y * test_distance,
                    z=0.5  # Height offset
                )
                
                # Create transform for lead vehicle
                spawn_transform = carla.Transform(
                    spawn_location,
                    ego_transform.rotation
                )
                
               
                vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.*'))
                vehicle_bp.set_attribute('color', '255,0,0')  # Red color for visibility
                
                # Try to spawn the lead vehicle
                self.lead_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
                
                if self.lead_vehicle:
                    self.actor_list.append(self.lead_vehicle)
                    # Set lead vehicle to move at a slower speed than ego vehicle's target
                    random_speed = np.random.uniform(20, 25)  # km/h, slightly slower than ego
                    self.tm.set_desired_speed(self.lead_vehicle, random_speed)
                    self.lead_vehicle.set_autopilot(True, self.tm.get_port())
                    self.world.tick()  # Ensure CARLA updates the state
                    time.sleep(0.2)
                    print(f"[INFO] Lead vehicle spawned at {test_distance}m with speed {random_speed:.2f} km/h")
                    break
            except Exception as e:
                print(f"[WARNING] Failed to spawn lead vehicle at {test_distance}m: {e}")
                continue
        
        # Fallback to existing spawn points if direct positioning fails
        if self.lead_vehicle is None:
            # Get available spawn points and sort by distance and direction from ego
            spawn_points = self.world.get_map().get_spawn_points()
            
            # Find suitable points ahead of ego vehicle
            suitable_points = []
            for sp in spawn_points:
                # Vector from ego to spawn point
                to_spawn = sp.location - ego_location
                
                # Check if point is in front (dot product with forward vector)
                forward_projection = to_spawn.x * forward_vector.x + to_spawn.y * forward_vector.y
                
                if forward_projection > 0:  # Point is in front
                    # Check lateral offset
                    right_vector = carla.Vector3D(x=-forward_vector.y, y=forward_vector.x, z=0)
                    lateral_offset = abs(to_spawn.x * right_vector.x + to_spawn.y * right_vector.y)
                    
                    # Distance from ego to spawn point
                    distance_to_spawn = to_spawn.length()
                    
                    # Add points that are reasonably aligned with ego's path
                    if lateral_offset < 3.0 and 10.0 < distance_to_spawn < 40.0:
                        suitable_points.append((sp, distance_to_spawn))
            
            # Sort by distance
            suitable_points.sort(key=lambda x: x[1])
            
            # Try suitable points in order
            for spawn_point, dist in suitable_points:
                try:
                    vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.*'))
                    vehicle_bp.set_attribute('color', '255,0,0')  # Red for visibility
                    
                    self.lead_vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                    
                    if self.lead_vehicle:
                        self.actor_list.append(self.lead_vehicle)
                        random_speed = np.random.uniform(20, 25)
                        self.tm.set_desired_speed(self.lead_vehicle, random_speed)
                        self.lead_vehicle.set_autopilot(True, self.tm.get_port())
                        self.world.tick()
                        print(f"[INFO] Lead vehicle spawned at spawn point, distance: {dist:.1f}m, speed: {random_speed:.1f} km/h")
                        break
                except Exception as e:
                    print(f"[WARNING] Failed to spawn at point {spawn_point}: {e}")
                    continue
        
        if self.lead_vehicle is None:
            print("[WARNING] Failed to spawn lead vehicle after multiple attempts")
        
        # Apply random yaw so RL doesn't guess to go straight
        angle_adj = random.randrange(-SPIN, SPIN, 1)
        trans = self.vehicle.get_transform()
        trans.rotation.yaw = trans.rotation.yaw + angle_adj
        self.vehicle.set_transform(trans)
        
        # Setup camera, collision and lane sensors
        
        # First create the semantic segmentation camera blueprint (keep for CNN processing)
        camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(self.im_width))
        camera_bp.set_attribute('image_size_y', str(self.im_height))
        
        # Set camera position
        camera_init_trans = carla.Transform(carla.Location(x=self.CAMERA_POS_X, z=self.CAMERA_POS_Z))
        self.camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda data: self.process_img(data))
        
        # Add RGB camera for object detection
        rgb_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute('image_size_x', str(self.im_width))
        rgb_camera_bp.set_attribute('image_size_y', str(self.im_height))
        rgb_camera_bp.set_attribute('fov', '110')  # Same FOV as in YOLO detection script
        
        # Set up RGB camera - same position as semantic camera
        self.rgb_camera = self.world.spawn_actor(rgb_camera_bp, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_camera)
        self.rgb_camera.listen(lambda data: self.process_rgb_img(data))
        
        # Collision sensor
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # Lane invasion sensor
        lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lanesensor = self.world.spawn_actor(lanesensor, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.lanesensor)
        self.lanesensor.listen(lambda event: self.lane_data(event))

        # Wait for camera to be ready
        while self.front_camera is None:
            time.sleep(0.01)
        
        # Initialize episode variables
        self.episode_start = time.time()
        self.steering_lock = False
        self.steering_lock_start = None 
        self.step_counter = 0
        self.last_time = time.time()
        
        # Reset control smoothing variables
        self.LAST_THROTTLE = 0.0
        self.LAST_BRAKE = 0.0
        self.FILTERED_DISTANCE = float('inf')
        
        # Reset vehicle controls
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        # Apply initial throttle to get the vehicle moving with gradual application
        initial_throttle = 0.8
        for i in range(5):  # Gradual application over 5 steps
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=initial_throttle * (i+1)/5, 
                brake=0.0))
            time.sleep(0.1)  # Small delay between steps
            self.world.tick()
        
        try:
            self.image_for_CNN = self.apply_cnn(self.front_camera[self.height_from:, self.width_from:self.width_to])
            return self.image_for_CNN, {}
        except Exception as e:
            print(f"[ERROR] Reset failed: {e}")
            return self.observation_space.sample(), {}

    def process_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3] # this is to ignore the 4th Alpha channel - up to 3
        self.front_camera = i

    def process_rgb_img(self, image):
        # Process RGB camera image
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3]  # Remove alpha channel
        self.rgb_camera_image = i

    def collision_data(self, event):
        self.collision_hist.append(event)
        
    def lane_data(self, event):
        self.lane_invade_hist.append(event)
import carla
import time
import numpy as np
import cv2
import threading

SHOW_SENSOR_OUTPUT = False

class VehicleSetup:
    def __init__(self):
        self.running = True  # Control spectator thread loop
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town04')
        self.blueprint_library = self.world.get_blueprint_library()
        self.lead_vehicle = None 

        # Holders for vehicle and sensors
        self.vehicle = None
        self.sensors = {}

        # LiDAR points
        self.semantic_lidar_points = None
        self.lidar_points = None
        self.front_camera = None

        # Spectator camera thread
        self.spectator_thread = None

        print("[INFO] Vehicle setup initialized.")

    def spawn_vehicle(self):
        available_vehicles = [bp.id for bp in self.blueprint_library.filter('vehicle.*')]
        vehicle_bp = self.blueprint_library.find(available_vehicles[0])  # Pick the first available vehicle
        spawn_points = self.world.get_map().get_spawn_points()

        for i, spawn_point in enumerate(spawn_points):
            try:
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

                if self.vehicle:
                    print(f"[INFO] Vehicle '{vehicle_bp.id}' spawned at spawn point {i}")
                    self.vehicle.set_simulate_physics(True)
                    
                    # ✅ Enable Traffic Manager and autopilot
                    self.tm = self.client.get_trafficmanager(8000)

                    # Enable synchronous mode for deterministic simulation
                    settings = self.world.get_settings()
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 0.05
                    self.world.apply_settings(settings)

                    # Traffic Manager needs synchronous mode too
                    self.tm.set_synchronous_mode(True)

                    # Traffic Manager vehicle settings
                    self.tm.ignore_lights_percentage(self.vehicle, 0)
                    self.tm.ignore_signs_percentage(self.vehicle, 0)
                    self.tm.auto_lane_change(self.vehicle, True)
                    self.tm.set_desired_speed(self.vehicle, 20)  # Speed in km/h
                    self.tm.distance_to_leading_vehicle(self.vehicle, 5.0)

                    # ✅ Enable autopilot with TM controlling the vehicle
                    # self.vehicle.set_autopilot(True, self.tm.get_port())
                    self.vehicle.set_autopilot(False)
                    print("[INFO] Autopilot enabled. Traffic Manager is controlling the vehicle.")

                    # Optional: Start the spectator view (follow vehicle)
                    self.set_spectator_camera()

                    return  # Exit after successful spawn
            except RuntimeError as e:
                print(f"[WARNING] Spawn attempt {i} failed: {e}")

        print("[ERROR] All spawn attempts failed!")
        exit(1)


    def attach_sensors(self, camera_callback=None):
        if not self.vehicle:
            print("[ERROR] No vehicle found.")
            return

        # Front RGB Camera
        # self.add_camera(
        #     name='front_camera',
        #     transform=carla.Location(x=1.5, z=2.0),
        #     width='800',
        #     height='600',
        #     fov='90',
        #     callback=camera_callback
        # )
        self.add_camera(
            name='front_camera',
            transform=carla.Transform(
                carla.Location(x=1.6, y=0.0, z=1.7),
                carla.Rotation(pitch=-10.0)  # tilt downward
            ),
            width='800',
            height='600',
            fov='70',
            callback=camera_callback
        )



        # LiDAR
        # lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        # lidar_bp.set_attribute('range', '100')
        # lidar_bp.set_attribute('points_per_second', '1000000')
        # lidar_bp.set_attribute('rotation_frequency', '20')

        # lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        # lidar_actor = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        # lidar_actor.listen(self.lidar_callback)
        # self.sensors['lidar'] = lidar_actor
        
        
        sem_lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast_semantic')
        sem_lidar_bp.set_attribute('range', '100')
        sem_lidar_bp.set_attribute('points_per_second', '1000000')
        sem_lidar_bp.set_attribute('rotation_frequency', '20')
        sem_lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        sem_lidar_actor = self.world.spawn_actor(sem_lidar_bp, sem_lidar_transform, attach_to=self.vehicle)
        sem_lidar_actor.listen(self.semantic_lidar_callback)
        self.sensors['semantic_lidar'] = sem_lidar_actor

        # IMU Sensor
        imu_bp = self.blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=0, z=1.0))
        imu_actor = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle)
        imu_actor.listen(lambda imu: print(f"[IMU] Accel: {imu.accelerometer}, Gyro: {imu.gyroscope}") if SHOW_SENSOR_OUTPUT else None)
        self.sensors['imu'] = imu_actor

        # GNSS Sensor
        gnss_bp = self.blueprint_library.find('sensor.other.gnss')
        gnss_transform = carla.Transform(carla.Location(x=0, z=2.0))
        gnss_actor = self.world.spawn_actor(gnss_bp, gnss_transform, attach_to=self.vehicle)
        gnss_actor.listen(lambda gnss: print(f"[GNSS] Lat: {gnss.latitude}, Lon: {gnss.longitude}") if SHOW_SENSOR_OUTPUT else None)
        self.sensors['gnss'] = gnss_actor

        print("[INFO] All sensors attached successfully.")

    def add_camera(self, name, transform, width, height, fov, sensor_type='sensor.camera.rgb', callback=None):
        camera_bp = self.blueprint_library.find(sensor_type)
        camera_bp.set_attribute('image_size_x', width)
        camera_bp.set_attribute('image_size_y', height)
        camera_bp.set_attribute('fov', fov)

        camera_actor = self.world.spawn_actor(
            camera_bp,
            transform,
            attach_to=self.vehicle
        )

        if callback:
            print(f"[INFO] External callback provided for {name}.")
            camera_actor.listen(lambda img: callback(img, name))  # External callback
        else:
            print(f"[INFO] No external callback for {name}. Using default.")
            camera_actor.listen(lambda img: self.camera_callback(img, name))  # Default internal

        self.sensors[name] = camera_actor

    def camera_callback(self, image, name):
        """
        Default camera callback: Converts the raw image data to a numpy array and stores it.
        """
        # Convert image using CityScapesPalette (or choose another converter if desired)
        image.convert(carla.ColorConverter.CityScapesPalette)
        # Create numpy array from raw data
        i = np.array(image.raw_data)
        # Reshape to image dimensions. 
        # (Assuming the sensor's image size matches what's set; adjust as necessary.)
        # CARLA images often have an alpha channel, so we take only the first 3 channels.
        # You might need to set these dimensions explicitly if not provided.
        height = image.height
        width = image.width
        i = i.reshape((height, width, 4))[:, :, :3]
        self.front_camera = i


    def lidar_callback(self, point_cloud):
        """
        Callback for LiDAR data.
        """
        points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
        self.lidar_points = points
        # Optional: Print info
        # print(f"[LIDAR] Received {len(self.lidar_points)} points")
    def semantic_lidar_callback(self, sem_lidar_measurement):
        # Semantic LiDAR data: points are stored as [X, Y, Z, Cosine, ObjIdx, SemanticTag]
        data = np.frombuffer(sem_lidar_measurement.raw_data, dtype=np.float32)
        if data.size == 0:
            self.semantic_lidar_points = None
        else:
            data = data.reshape(-1, 6)
            self.semantic_lidar_points = []
            for d in data:
                detection = type("SemanticLidarDetection", (), {})()
                detection.point = type("Point", (), {})()
                detection.point.x = d[0]
                detection.point.y = d[1]
                detection.point.z = d[2]
                detection.cosine = d[3]
                detection.obj_index = int(d[4])
                detection.semantic_tag = int(d[5])
                self.semantic_lidar_points.append(detection)

    def set_spectator_camera(self):
        """
        Start following the vehicle from top-down.
        """
        spectator = self.world.get_spectator()
        vehicle_transform = self.vehicle.get_transform()

        spectator_transform = carla.Transform(
            vehicle_transform.location + carla.Location(z=50),  # Top-down view
            carla.Rotation(pitch=-90)
        )
        spectator.set_transform(spectator_transform)

        print("[INFO] Spectator camera set.")

        # Start follow_vehicle thread
        self.spectator_thread = threading.Thread(target=self.follow_vehicle, daemon=True)
        self.spectator_thread.start()

    def follow_vehicle(self):
        """
        Continuously follow the vehicle with spectator.
        """
        while self.running and self.vehicle is not None:
            self.world.tick()

            try:
                vehicle_transform = self.vehicle.get_transform()

                spectator_transform = carla.Transform(
                    vehicle_transform.location + carla.Location(z=50),
                    carla.Rotation(pitch=-90)
                )
                self.world.get_spectator().set_transform(spectator_transform)

            except RuntimeError:
                print("[WARNING] Vehicle actor no longer valid. Exiting follow thread.")
                break

            time.sleep(0.05)

    def shutdown(self):
        print("[INFO] Shutting down vehicle setup...")
        self.running = False
        if self.spectator_thread is not None:
            self.spectator_thread.join(timeout=1)
        # Destroy additional vehicles
        if hasattr(self, 'additional_vehicles'):
            for veh in self.additional_vehicles:
                try:
                    veh.destroy()
                except Exception as e:
                    print(f"[WARNING] Error destroying additional vehicle: {e}")
        # Destroy sensors
        for sensor in self.sensors.values():
            try:
                if sensor.is_listening:
                    sensor.stop()
                sensor.destroy()
            except Exception as e:
                print(f"[WARNING] Error destroying sensor: {e}")
        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
            except Exception as e:
                print(f"[WARNING] Error destroying vehicle: {e}")
        cv2.destroyAllWindows()
        print("[INFO] Vehicle setup shutdown complete.")

    def spawn_lead_vehicle(self):
        """Spawn a lead vehicle in front of the ego vehicle."""
        if self.vehicle is None:
            print("[ERROR] Ego vehicle not spawned yet.")
            return
        
        # Get ego vehicle transform and offset 20 meters ahead.
        ego_transform = self.vehicle.get_transform()
        lead_location = ego_transform.location + carla.Location(x=20)
        lead_transform = carla.Transform(lead_location, ego_transform.rotation)
        
        # Choose a vehicle blueprint (different from ego if desired)
        vehicle_bp = np.random.choice(self.blueprint_library.filter('vehicle.*'))
        self.lead_vehicle = self.world.try_spawn_actor(vehicle_bp, lead_transform)
        
        if self.lead_vehicle:
            print("[INFO] Lead vehicle spawned.")
            # Option 1: Let Traffic Manager control it with a random speed.
            tm = self.client.get_trafficmanager(8000)
            tm.set_synchronous_mode(True)
            random_speed = np.random.uniform(10, 40)  # Speed between 10 and 40 km/h
            tm.set_desired_speed(self.lead_vehicle, random_speed)
            self.lead_vehicle.set_autopilot(True, tm.get_port())
        else:
            print("[WARNING] Failed to spawn lead vehicle.")

    def destroy_lead_vehicle(self):
        """Destroy the lead vehicle if it exists."""
        if self.lead_vehicle is not None:
            self.lead_vehicle.destroy()
            self.lead_vehicle = None
            print("[INFO] Lead vehicle destroyed.")

    def update_lead_vehicle(self):
        """
        Randomly decide whether to spawn or remove the lead vehicle,
        and occasionally update its desired speed.
        """
        # Example: Every call, with a small probability change the lead vehicle state.
        if self.lead_vehicle is None:
            if np.random.rand() < 0.05:  # 5% chance to spawn a lead vehicle
                self.spawn_lead_vehicle()
        else:
            if np.random.rand() < 0.02:  # 2% chance to remove it
                self.destroy_lead_vehicle()
            else:
                # Occasionally update the lead vehicle's speed to simulate traffic dynamics.
                if np.random.rand() < 0.05:
                    tm = self.client.get_trafficmanager(8000)
                    random_speed = np.random.uniform(10, 40)
                    tm.set_desired_speed(self.lead_vehicle, random_speed)
                    print(f"[INFO] Lead vehicle desired speed updated to {random_speed:.1f} km/h.")


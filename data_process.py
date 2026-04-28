import os
import argparse
import json
from tqdm import tqdm

from flow_planner.data.data_process.data_processor import DataProcessor

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping


def get_filter_parameters(num_scenarios_per_type=None, limit_total_scenarios=None, shuffle=True, scenario_tokens=None, log_names=None):
    scenario_types = None
    map_names = None
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None
    expand_scenarios = True
    remove_invalid_goals = False
    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None

    return scenario_types, scenario_tokens, log_names, map_names, num_scenarios_per_type, limit_total_scenarios, timestamp_threshold_s, ego_displacement_minimum_m, \
           expand_scenarios, remove_invalid_goals, shuffle, ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance


def process_scenario_wrapper(args_tuple):
    """Wrapper function for parallel processing."""
    scenario, save_dir = args_tuple
    from flow_planner.data.data_process.data_processor import DataProcessor
    import numpy as np
    from flow_planner.data.data_process.agent_process import agent_past_process, agent_future_process, sampled_tracked_objects_to_array_list, sampled_static_objects_to_array_list
    from flow_planner.data.data_process.map_process import get_neighbor_vector_set_map, map_process
    from flow_planner.data.data_process.roadblock_utils import route_roadblock_correction
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
    
    try:
        # Settings
        past_time_horizon = 2
        num_past_poses = 20
        future_time_horizon = 8
        num_future_poses = 80
        num_agents = 32
        num_static = 5
        max_ped_bike = 10
        
        map_features = ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES', 'ROUTE_POLYGON', 'CROSSWALK']
        max_elements = {'LANE': 70, 'LEFT_BOUNDARY': 70, 'RIGHT_BOUNDARY': 70, 'ROUTE_LANES': 25, 'ROUTE_POLYGON': 5, 'CROSSWALK': 5}
        max_points = {'LANE': 20, 'LEFT_BOUNDARY': 20, 'RIGHT_BOUNDARY': 20, 'ROUTE_LANES': 20, 'ROUTE_POLYGON': 10, 'CROSSWALK': 10}
        vehicle_parameters = get_pacifica_parameters()
        radius = 100
        interpolation_method = 'linear'
        
        map_name = scenario._map_name
        token = scenario.token
        map_api = scenario.map_api
        
        # Get ego agent past
        anchor_ego_state = scenario.initial_ego_state
        past_ego_states = scenario.get_ego_past_trajectory(
            iteration=0, num_samples=num_past_poses, time_horizon=past_time_horizon
        )
        
        output = np.zeros((len(list(past_ego_states)) + 1, 13), dtype=np.float64)
        past_ego_list = list(past_ego_states) + [anchor_ego_state]
        for i, state in enumerate(past_ego_list):
            output[i, 0] = state.rear_axle.x
            output[i, 1] = state.rear_axle.y
            output[i, 2] = state.rear_axle.heading
            output[i, 3] = state.dynamic_car_state.rear_axle_velocity_2d.x
            output[i, 4] = state.dynamic_car_state.rear_axle_velocity_2d.y
            output[i, 5] = state.dynamic_car_state.rear_axle_acceleration_2d.x
            output[i, 6] = state.dynamic_car_state.rear_axle_acceleration_2d.y
            output[i, 7] = vehicle_parameters.width
            output[i, 8] = vehicle_parameters.length
        output[:, 9] = 1
        ego_agent_past = output
        
        past_time_stamps = list(
            scenario.get_past_timestamps(
                iteration=0, num_samples=num_past_poses, time_horizon=past_time_horizon
            )
        ) + [scenario.start_time]
        time_stamps_past = np.array([t.time_us for t in past_time_stamps], dtype=np.int64)
        
        # Get neighbor agents
        present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=0, time_horizon=past_time_horizon, num_samples=num_past_poses
            )
        ]
        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        past_tracked_objects_array_list, past_tracked_objects_types = \
            sampled_tracked_objects_to_array_list(sampled_past_observations)
        current_static_objects_array_list, current_static_objects_types = sampled_static_objects_to_array_list(present_tracked_objects)
        
        ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects = \
            agent_past_process(ego_agent_past, past_tracked_objects_array_list, past_tracked_objects_types, num_agents, current_static_objects_array_list, current_static_objects_types, num_static, max_ped_bike)
        
        # Get map
        ego_state = scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        traffic_light_data = list(scenario.get_traffic_light_status_at_iteration(0))
        
        if route_roadblock_ids != ['']:
            route_roadblock_ids = route_roadblock_correction(ego_state, map_api, route_roadblock_ids)
        
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, map_features, ego_coords, radius, traffic_light_data
        )
        vector_map = map_process(map_api, route_roadblock_ids, ego_state.rear_axle, coords, traffic_light_data, speed_limit, lane_route, map_features, 
                                 max_elements, max_points, interpolation_method)
        
        # Get ego future
        current_absolute_state = scenario.initial_ego_state
        trajectory_absolute_states = scenario.get_ego_future_trajectory(
            iteration=0, num_samples=num_future_poses, time_horizon=future_time_horizon
        )
        ego_agent_future = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
        )
        
        # Get neighbor future
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=0, time_horizon=future_time_horizon, num_samples=num_future_poses
            )
        ]
        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_array_list(sampled_future_observations)
        neighbor_agents_future = agent_future_process(current_absolute_state, future_tracked_objects_tensor_list, num_agents, neighbor_indices)
        
        # Calculate additional ego states
        current_state = ego_agent_past[-1]
        prev_state = ego_agent_past[-2]
        dt = (time_stamps_past[-1] - time_stamps_past[-2]) * 1e-6
        cur_velocity = current_state[3]
        angle_diff = current_state[2] - prev_state[2]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
        yaw_rate = angle_diff / dt
        
        if abs(cur_velocity) < 0.2:
            steering_angle = 0.0
            yaw_rate = 0.0
        else:
            steering_angle = np.arctan(yaw_rate * get_pacifica_parameters().wheel_base / abs(cur_velocity))
            steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
            yaw_rate = np.clip(yaw_rate, -0.95, 0.95)
        
        past = np.zeros((ego_agent_past.shape[0], ego_agent_past.shape[1]+1), dtype=np.float32)
        past[:, :2] = ego_agent_past[:, :2]
        past[:, 2] = np.cos(ego_agent_past[:, 2])
        past[:, 3] = np.sin(ego_agent_past[:, 2])
        past[:, 4:] = ego_agent_past[:, 3:]
        
        current = np.zeros((ego_agent_past.shape[1]+3), dtype=np.float32)
        current[:2] = current_state[:2]
        current[2] = np.cos(current_state[2])
        current[3] = np.sin(current_state[2])
        current[4:8] = current_state[3:7]
        current[8] = steering_angle
        current[9] = yaw_rate
        current[10:] = current_state[7:]
        
        ego_agent_past = past
        ego_current_state = current
        
        # Gather data
        data = {"map_name": map_name, "token": token, "ego_agent_past": ego_agent_past, "ego_current_state": ego_current_state, "ego_agent_future": ego_agent_future,
                "neighbor_agents_past": neighbor_agents_past, "neighbor_agents_future": neighbor_agents_future, "static_objects": static_objects}
        data.update(vector_map)
        
        # Save to disk
        np.savez(f"{save_dir}/{map_name}_{token}.npz", **data)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--data_path', default='/data/nuplan-v1.1/trainval', type=str, help='path to raw data')
    parser.add_argument('--map_path', default='/data/nuplan-v1.1/maps', type=str, help='path to map data')
    parser.add_argument('--save_path', default='./cache', type=str, help='path to save processed data')
    parser.add_argument('--scenarios_per_type', type=int, default=None, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', type=int, default=10, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=bool, default=True, help='shuffle scenarios')
    parser.add_argument('--num_workers', type=int, default=8, help='number of parallel workers')
    args = parser.parse_args()

    # Create save folder
    os.makedirs(args.save_path, exist_ok=True)

    # Load log names
    with open('./nuplan_train.json', "r", encoding="utf-8") as file:
        log_names = json.load(file)

    print(f"Building scenarios...")
    map_version = "nuplan-maps-v1.0"
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, None, map_version)
    scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios, log_names=log_names))

    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    # Clean up
    del worker, builder, scenario_filter

    # Process scenarios using multiprocessing
    print(f"Processing scenarios with {args.num_workers} workers...")
    
    from multiprocessing import Pool
    
    scenario_list = list(scenarios)
    process_args = [(s, args.save_path) for s in scenario_list]
    
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap(process_scenario_wrapper, process_args), total=len(scenario_list)))
    
    success_count = sum(results)
    print(f"Successfully processed {success_count}/{len(scenario_list)} scenarios")

    # Generate npz file list
    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]

    with open('./diffusion_planner_training.json', 'w') as json_file:
        json.dump(npz_files, json_file, indent=4)

    print(f"Saved {len(npz_files)} .npz file names")


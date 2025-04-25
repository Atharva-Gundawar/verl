import json
from rich import print
from copy import deepcopy
from tqdm import tqdm
import boto3

DEBUG = False
def parse_response_maze(response: list[str], DEBUG: bool = False) -> list[str]:
    """
    Parse the response from the model
    """
    
    
    # Get each action from the response till "eos"
    operations = []
    plan = []
    string_response = []
    i = 0
    operation_syntax_error = False
    plan_syntax_error = False
    while i < len(response):
        # print(f"[*] Parsing action: {response[i]}")
        if response[i] == "bos":
            i += 1
            continue
        if response[i] == "eos":
            break
        if response[i] == "plan":
            action = ' '.join(response[i:i+3])
            if "create" in action or "close" in action or "eos" in action:
                extra_action = [j for j in response[i:i+3] if "create" in j or "close" in j or "eos" in j][0]
                if DEBUG:   
                    print(f"[-] Syntax error: {action}")
                plan_syntax_error = True
                i=i+action.split(" ").index(extra_action)
            else:
                string_response.append(action)
                i += 3
        elif response[i] == "create":
            action = ' '.join(response[i:i+5])
            if  "close" in action or "eos" in action or "plan" in action:
                extra_action = [j for j in response[i:i+5] if "close" in j or "eos" in j or "plan" in j][0]
                if DEBUG:   
                    print(f"[-] Syntax error: {action}")
                operation_syntax_error = True
                i=i+action.split(" ").index(extra_action)
            else:
                string_response.append(action)
                try:
                    action = {
                        'action': 'create',
                        'coordinates': (int(response[i+1]), int(response[i+2])),
                        'cost': int(response[i+3][1:]),
                        'heuristic': int(response[i+4][1:])
                    }
                    operations.append(action)
                except Exception as e:
                    operation_syntax_error = True
                
                i += 5
        elif response[i] == "close":
            action = ' '.join(response[i:i+5])
            if "create" in action or "eos" in action or "plan" in action:
                extra_action = [j for j in response[i:i+5] if "create" in j or "eos" in j or "plan" in j][0]
                if DEBUG:   
                    print(f"[-] Syntax error: {action}")
                operation_syntax_error = True
                i=i+action.split(" ").index(extra_action)
            else:
                string_response.append(action)
                try:
                    action = {
                        'action': 'close',
                        'coordinates': (int(response[i+1]), int(response[i+2])),
                        'cost': int(response[i+3][1:]),
                        'heuristic': int(response[i+4][1:])
                    }
                    operations.append(action)
                except Exception as e:
                    operation_syntax_error = True
                i += 5
        else:
            if DEBUG:
                print(f"Unknown action: {response[:i]} {response[i:]}")
            operation_syntax_error = True
            i += 1
    if not plan_syntax_error:
        while len(string_response) > 0 and "plan" in string_response[-1]:
            list_action = string_response[-1].split(' ')
            try:
                action = {
                    'action': list_action[0],
                    'coordinates': (int(list_action[1]), int(list_action[2])),
                }
                plan.append(action)
            except Exception as e:
                plan_syntax_error = True
                break
            string_response.pop()
    plan.reverse()
    if len(plan) == 0:
        plan_syntax_error = True
    return operations, plan, string_response, operation_syntax_error, plan_syntax_error

def parse_prompt_maze(prompt: list[str], DEBUG: bool = False, maze_size: tuple[int, int] = (10, 10)) -> list[str]:
    """
    Parse the prompt from the prompt. Get the start and goal coordinates. and the walls of the maze.
    """
    walls = []
    for i in range(len(prompt)):
        if prompt[i] == "start":
            start = (int(prompt[i+1]), int(prompt[i+2]))
        elif prompt[i] == "goal":
            goal = (int(prompt[i+1]), int(prompt[i+2]))
        elif prompt[i] == "wall":
            walls.append((int(prompt[i+1]), int(prompt[i+2])))

    # Create maze matrix with walls as 1 and empty spaces as 0
    maze = [[0 for _ in range(maze_size[0])] for _ in range(maze_size[1])]
    for wall in walls:
        maze[wall[0]][wall[1]] = 1
    maze[start[0]][start[1]] = 2
    maze[goal[0]][goal[1]] = 3
    if DEBUG:
        print(maze)
    return start, goal, walls, maze


def get_neighbors(coordinates: tuple[int, int], walls: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Get the neighbors of a node
    """
    neighbors = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        neighbor = (coordinates[0] + dx, coordinates[1] + dy)
        if neighbor not in walls:
            neighbors.append(neighbor)
    return neighbors

def evaluate_trace_response(response: list[str], prompt: list[str], maze_size: tuple[int, int]=(10, 10), ground_truth_plan: list = None, check_for_optimality: bool = False) -> tuple[bool, list[str]]:
    """
    Evaluate the trace response by simulating A* search.
    Returns (is_valid, list_of_errors)
    """
    # print(f"Evaluating trace response...{len(prompt)}, {len(response)}")
    start, goal, walls, maze = parse_prompt_maze(prompt, DEBUG, maze_size)
    operations, llm_plan, string_response, operation_syntax_error, plan_syntax_error = parse_response_maze(response, DEBUG)
    # if len(llm_plan) == 0:
    #     print(f"Empty plan: {response}")
    trace_is_valid = False  
    llm_plan_is_valid = False
    if DEBUG:
        print(operations)
    errors = []
    llm_plan_errors = []
    if DEBUG:
        if plan_syntax_error and not operation_syntax_error:
            print(f"==========================Plan syntax error but not operation syntax error")
    if operation_syntax_error:
        errors.append("[TRACE ERROR: operation_syntax_error] Syntax error in operation format")
        trace_is_valid = False
    if plan_syntax_error:
        llm_plan_errors.append("[PLAN ERROR: plan_syntax_error] Syntax error in plan format")
        llm_plan_is_valid = False
    # Initialize A* data structures
    open_list = []  # list of (x,y,cost,heuristic)
    closed_list = []
    parent_dict_for_plan = {}
    last_closed = None
    g_score = {}
    # Simulate A* operations
    for op in operations:
        coords = op['coordinates']
        x, y = coords
        if DEBUG:
            print(f"Processing operation: {op}")
            print(f"Open list: {open_list}")
        # Check if coordinates are valid (within 0-9 bounds)
        if not (0 <= x <= maze_size[0] - 1 and 0 <= y <= maze_size[1] - 1):
            errors.append(f"[TRACE ERROR: invalid_coordinates] Invalid coordinates: ({x}, {y})")
            continue
            
        # Check if coordinates are walls
        if coords in walls:
            errors.append(f"[TRACE ERROR: create_close_wall] Attempted to create/close wall at ({x}, {y})")
            continue
            
        if op['action'] == 'create':
            if last_closed: 
                # Get neighbors of last closed node
                neighbors = get_neighbors(last_closed['coordinates'], walls)
                if coords not in neighbors:
                    errors.append(f"[TRACE ERROR: create_non_neighbor] Attempted to create node ({x}, {y}) that is not a neighbor of the last closed node ({last_closed['coordinates'][0]}, {last_closed['coordinates'][1]})")
                    continue
            # Check if node is already in closed list
            if coords in closed_list:
                errors.append(f"[TRACE ERROR: create_already_closed] Attempted to create node ({x}, {y}) that is already closed")
                continue
                
            # Add to open list
            open_list.append((x, y, op['cost'], op['heuristic']))
            if last_closed is not None:
                # If this is a new node or we found a better path
                if coords not in g_score or op['cost'] < g_score[coords]:
                    g_score[coords] = op['cost']
                    parent_dict_for_plan[coords] = last_closed['coordinates']
            else:
                # For the initial node (no parent)
                g_score[coords] = op['cost']
            
            
        elif op['action'] == 'close':
            # Check if node is in open list
            if not any((coords[0], coords[1], op['cost'], op['heuristic']) == (x,y,c,h) for x,y,c,h in open_list):
                errors.append(f"[TRACE ERROR: close_not_in_open_list] Attempted to close node ({x}, {y}) that is not in open list")
                continue
            # Check if node is the lowest f value in open list
            lowest_f_value = min([(c+h, (x,y,c,h)) for x,y,c,h in open_list], key=lambda x: x[0])
            node_f_value = op['cost'] + op['heuristic']
            if node_f_value != lowest_f_value[0]:
                errors.append(f"[TRACE ERROR: close_not_lowest_f_value] Attempted to close node ({x}, {y}) that is not the lowest f value in open list, f value is {node_f_value} and lowest f value is {lowest_f_value[0]} for node {lowest_f_value[1]}")
                continue
            
            # Move from open to closed list
            # remove the first occurence of the op x,y,c,h from open list
            index_to_remove = open_list.index((coords[0], coords[1], op['cost'], op['heuristic']))
            open_list.pop(index_to_remove)
            closed_list.append(coords)
            last_closed = deepcopy(op)
            # Get successors of the node 
    extracted_plan = []
    # Check if goal was reached
    if goal not in closed_list:
        errors.append("[TRACE ERROR: goal_not_reached] Goal was not reached")
    else:
        # Extract plan 
        
        current_node = goal
        if DEBUG:
            print(parent_dict_for_plan, closed_list, start, goal)
        while current_node != start:
            extracted_plan.append(current_node)
            try:
                current_node = parent_dict_for_plan[current_node]
            except KeyError:
                # print(f"KeyError: {current_node}, {goal}, {start}, {parent_dict_for_plan}")
                errors.append(f"[TRACE ERROR: plan_extraction_error] Plan extraction error at {current_node}")
                break
        extracted_plan.append(start)
        extracted_plan.reverse()
        if DEBUG:
            print("Plan: ", extracted_plan)
        

    if not plan_syntax_error:
        llm_plan_is_valid, llm_plan_errors_new  = evaluate_plan(start, goal, walls, llm_plan, maze_size, ground_truth_plan, check_for_optimality)
        llm_plan_errors.extend(llm_plan_errors_new)
        # Check if the extracted plan is same as the llm plan
        if len(extracted_plan) != len(llm_plan):
            errors.append("[TRACE ERROR: plan_length_mismatch] Plan length mismatch")
        else:
            # for i, op in enumerate(llm_plan):
            #     if op['coordinates'] != extracted_plan[i]:
                    
            #         #Pretty print maze with the path
            #         maze_with_path = deepcopy(maze)
            #         table = PrettyTable()
            #         table.field_names = ["X"] + [i for i in range(maze_size[0])]
            #         for i, m in enumerate(maze_with_path):
            #             table.add_row([i] + m)
            #         print(table)
            #         print([j['coordinates'] for j in llm_plan], extracted_plan, parent_dict_for_plan)
            #         errors.append(f"[TRACE ERROR: plan_order_mismatch] Plan order mismatch at step {i}: expected {extracted_plan[i]}, got {op['coordinates']}")
            ## Execute both plans in the maze and check if they reach the goal
            extracted_plan = [{"coordinates": (x, y), "action": "plan"} for x, y in extracted_plan]
            extracted_plan_is_valid, extracted_plan_errors = evaluate_plan(start, goal, walls, extracted_plan, maze_size, ground_truth_plan, check_for_optimality)
            if not extracted_plan_is_valid:
                errors.append("[TRACE ERROR: extracted_plan_invalid] Extracted plan is invalid")
    if len(errors) == 0:
        trace_is_valid = True
    else:
        trace_is_valid = False
    if len(llm_plan_errors) == 0:
        llm_plan_is_valid = True
    else:
        llm_plan_is_valid = False
    # if plan_syntax_error and not operation_syntax_error:
    #     if not trace_is_valid and not llm_plan_is_valid:
    #         print(errors)
    return trace_is_valid, llm_plan_is_valid, errors, llm_plan_errors

def evaluate_plan(start: tuple[int, int], goal: tuple[int, int], walls: list[tuple[int, int]], plan: list[dict], maze_size: tuple[int, int]=(10, 10), ground_truth_plan: list = None, check_for_optimality: bool = False) -> tuple[bool, list[str]]:
    """
    Evaluate the plan by simulating it in the maze.
    Returns (is_valid, list_of_errors)
    """
    maze = [[0 for _ in range(maze_size[0])] for _ in range(maze_size[1])]
    errors = []
    for wall in walls:
        maze[wall[0]][wall[1]] = 1
    maze[start[0]][start[1]] = 2
    try:
        current_node_plan = plan[0]['coordinates']
    except IndexError:
        print(f"IndexError: {plan}")
        raise IndexError(f"IndexError: {plan}")
    if current_node_plan != start:
        errors.append("[PLAN ERROR: illegal_start] Plan does not start at start node")
        return False, errors
    
    for i, op in enumerate(plan[1:]):
        previous_node_plan = current_node_plan
        current_node_plan = op['coordinates']
        neighbors = get_neighbors(previous_node_plan, walls)
        if current_node_plan not in neighbors:
            errors.append(f"[PLAN ERROR: illegal_move] Illegal move at step {i}: expected one of {neighbors}, got {current_node_plan}")
            return False, errors
    if current_node_plan != goal:
        errors.append("[PLAN ERROR: goal_not_reached] Plan does not reach goal node")
        return False, errors
    if check_for_optimality:
        optimal_path_length = get_optimal_path_length(ground_truth_plan)
        if len(plan) != optimal_path_length:
            errors.append(f"[PLAN ERROR: suboptimal_path] Plan is not optimal, length is {len(plan)} and optimal length is {optimal_path_length}")
            return False, errors
    return True, []

def get_optimal_path_length(ground_truth_plan: list) -> int:
    return ground_truth_plan.count("plan")
    
    

def eval_file(file_path: str, maze_size: int, check_for_optimality: bool):
    with open(file_path, "r") as f:
        data = json.load(f)
    trace_is_valid_count = 0
    llm_plan_is_valid_count = 0
    correct_plan = {
        "trace_is_valid": 0,
        "trace_is_invalid": {
            'total': 0
        },
    }
    incorrect_plan = {
        "trace_is_valid": 0,
        "trace_is_invalid": {
            'total': 0,
        },
    }
    total_instances = len(data)
    total_rollouts = 0
    for instance in data:
        prompt = instance["prompt"]
        for rollout in instance["rollouts"]:
            response = rollout
            trace_is_valid, llm_plan_is_valid, errors, llm_plan_errors = evaluate_trace_response(response, prompt, (maze_size, maze_size), instance["plan"], check_for_optimality)
            if trace_is_valid:
                trace_is_valid_count += 1
            if llm_plan_is_valid:
                llm_plan_is_valid_count += 1
            first_trace_error = errors[0] if errors else None
            first_llm_plan_error = llm_plan_errors[0] if llm_plan_errors else None
            if first_trace_error:
                first_trace_error = first_trace_error.split("]")[0].split(":")[1].strip()
            if first_llm_plan_error:
                first_llm_plan_error = first_llm_plan_error.split("]")[0].split(":")[1].strip()
            if llm_plan_is_valid:
                if trace_is_valid:
                    correct_plan["trace_is_valid"] += 1
                else:
                    if first_trace_error in correct_plan["trace_is_invalid"]:
                        correct_plan["trace_is_invalid"][first_trace_error] += 1
                    else:
                        correct_plan["trace_is_invalid"][first_trace_error] = 1
                    correct_plan["trace_is_invalid"]["total"] += 1
                    
            else:
                if trace_is_valid:
                    incorrect_plan["trace_is_valid"] += 1
                else:
                    if first_trace_error in incorrect_plan["trace_is_invalid"]:
                        incorrect_plan["trace_is_invalid"][first_trace_error] += 1
                    else:
                        incorrect_plan["trace_is_invalid"][first_trace_error] = 1
                    incorrect_plan["trace_is_invalid"]["total"] += 1
                if first_llm_plan_error:
                    if first_llm_plan_error in incorrect_plan:
                        incorrect_plan[first_llm_plan_error] += 1
                    else:
                        incorrect_plan[first_llm_plan_error] = 1
    total_json = {
        "correct_plan": correct_plan,
        "incorrect_plan": incorrect_plan,
    }
    
    # Get confusion matrix 
    confusion_matrix = {
        "correct_plan": {
            "trace_is_valid": total_json["correct_plan"]["trace_is_valid"],
            "trace_is_invalid": total_json["correct_plan"]["trace_is_invalid"]["total"],
        },
        "incorrect_plan": {
            "trace_is_valid": total_json["incorrect_plan"]["trace_is_valid"],
            "trace_is_invalid": total_json["incorrect_plan"]["trace_is_invalid"]["total"],
        },
    }
    
    return confusion_matrix
    
    


def eval_A_star_single():
    prompt= ['start', '1', '0', 'goal', '7', '3', 'wall', '7', '0', 'wall', '0', '1', 'wall', '2', '1', 'wall', '3', '1', 'wall', '8', '1', 'wall', '6', '2', 'wall', '0', '3', 'wall', '3', '3', 'wall', '4', '3', 'wall', '6', '3', 'wall', '0', '4', 'wall', '1', '4', 'wall', '2', '4', 'wall', '5', '4', 'wall', '0', '5', 'wall', '5', '5', 'wall', '8', '5', 'wall', '5', '6', 'wall', '6', '6', 'wall', '1', '7', 'wall', '2', '7', 'wall', '5', '7', 'wall', '7', '7', 'wall', '8', '7', 'wall', '4', '8', 'wall', '6', '8', 'wall', '7', '8', 'wall', '8', '8', 'wall', '0', '9', 'wall', '2', '9', 'wall', '4', '9', 'wall', '5', '9']
    response=['create', '1', '0', 'c0', 'c9', 'close', '1', '0', 'c0', 'c9', 'create', '2', '0', 'c1', 'c8', 'create', '0', '0', 'c1', 'c10', 'create', '1', '1', 'c1', 'c8', 'close', '2', '0', 'c1', 'c8', 'create', '3', '0', 'c2', 'c7', 'close', '1', '1', 'c1', 'c8', 'create', '1', '2', 'c2', 'c7', 'close', '1', '2', 'c2', 'c7', 'create', '1', '3', 'c3', 'c6', 'create', '2', '2', 'c3', 'c6', 'create', '0', '2', 'c3', 'c8', 'close', '2', '2', 'c3', 'c6', 'create', '2', '3', 'c4', 'c5', 'create', '3', '2', 'c4', 'c5', 'close', '3', '2', 'c4', 'c5', 'create', '4', '2', 'c5', 'c4', 'close', '4', '2', 'c5', 'c4', 'create', '5', '2', 'c6', 'c3', 'create', '4', '1', 'c6', 'c5', 'close', '3', '0', 'c2', 'c7', 'create', '4', '0', 'c3', 'c6', 'close', '5', '2', 'c6', 'c3', 'create', '5', '3', 'c7', 'c2', 'create', '5', '1', 'c7', 'c4', 'close', '2', '3', 'c4', 'c5', 'close', '1', '3', 'c3', 'c6', 'close', '5', '3', 'c7', 'c2', 'close', '4', '0', 'c3', 'c6', 'create', '5', '0', 'c4', 'c5', 'create', '4', '1', 'c4', 'c5', 'close', '5', '0', 'c4', 'c5', 'create', '6', '0', 'c5', 'c4', 'create', '5', '1', 'c5', 'c4', 'close', '4', '1', 'c4', 'c5', 'create', '5', '1', 'c5', 'c4', 'close', '5', '1', 'c5', 'c4', 'create', '6', '1', 'c6', 'c3', 'close', '6', '0', 'c5', 'c4', 'close', '6', '1', 'c6', 'c3', 'create', '7', '1', 'c7', 'c2', 'close', '5', '1', 'c5', 'c4', 'close', '7', '1', 'c7', 'c2', 'create', '7', '2', 'c8', 'c1', 'close', '7', '2', 'c8', 'c1', 'create', '8', '2', 'c9', 'c2', 'create', '7', '3', 'c9', 'c0', 'close', '7', '3', 'c9', 'c0', 'plan', '1', '0', 'plan', '2', '0', 'plan', '3', '0', 'plan', '4', '0', 'plan', '4', '1', 'plan', '5', '1', 'plan', '6', '1', 'plan', '7', '1', 'plan', '7', '2', 'plan', '7', '3']
    trace_is_valid, llm_plan_is_valid, errors, llm_plan_errors = evaluate_trace_response(response, prompt, [], True)
    print(f"Trace is valid: {trace_is_valid}, LLM plan is valid: {llm_plan_is_valid}")
    print(f"Errors: {errors}")
    print(f"LLM plan errors: {llm_plan_errors}")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="maze-sweep-rep-nondet-large-0.json")
    parser.add_argument("--maze_size", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--check_for_optimality", action="store_false")
    args = parser.parse_args()
    DEBUG = args.debug
    if args.single or args.file is None:
        eval_A_star_single()
    else:
        confusion_matrix = eval_file(args.file, args.maze_size, args.check_for_optimality)
        print(confusion_matrix)
    # eval_file("rollouts_maze_10_by_10_swapped.json")
    # eval_file("deter_10x10.json")
    # eval_file("rollouts_maze-sweep-rep-nondet-large-33.json")
    
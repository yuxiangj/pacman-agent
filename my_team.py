# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions, Actions
from util import nearest_point
from util import Stack, PriorityQueue
from util import manhattan_distance

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

enemy_captured = False
def threshold_limit_search(initial_position, forbidden_position, walls, pellets, threshold=12):
        stack = Stack()
        stack.push(initial_position)
        visited = set([initial_position])
        moves = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

        while not stack.is_empty():
            current_position = stack.pop()

            for action in moves:
                next_position = nearest_point(Actions.get_successor(current_position, action))
                # Skip positions that are walls
                if walls[next_position[0]][next_position[1]]:
                    continue
                # Exclude the position from which the agent arrived
                if next_position == forbidden_position:
                    continue
                # If a pellet is found within the region, it is not considered a trap
                if pellets is not None and next_position in pellets:
                    return False
                
                if next_position not in visited:
                    visited.add(next_position)
                    stack.push(next_position)

            # If the number of visited cells exceeds the threshold, the region is not a trap
            if len(visited) > threshold:
                return False  

        return True

class OffensiveReflexAgent(CaptureAgent):
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.chased_time = 0 
        self.last_food_carried = 0

    def a_star_search(self, start_position, target_position, walls):
        frontier = PriorityQueue()
        frontier.push((start_position, [], 0), 0)  # (position, path, cost)
        explored = set()
        moves = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

        while not frontier.is_empty():
            current_position, path, g_cost = frontier.pop()

            # Goal condition
            if current_position == target_position:
                return path[0] if path else Directions.STOP

            if current_position in explored:
                continue
            explored.add(current_position)

            for action in moves:
                next_position = nearest_point(Actions.get_successor(current_position, action))

                # Skip positions that are walls
                if walls[next_position[0]][next_position[1]]:
                    continue
                # If trapped, chase ghost to return home
                if self.chased_time >= 30 and next_position == self.start:
                    return action

                if next_position not in explored:
                    new_path = path + [action]
                    new_g = g_cost + 1
                    h = manhattan_distance(next_position, target_position)
                    cost = new_g + h
                    frontier.push((next_position, new_path, new_g), cost)

        return Directions.STOP

    def choose_action(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        defenders = [e for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0]
        my_position = game_state.get_agent_state(self.index).get_position()
        walls = game_state.get_walls()
        legal_actions = game_state.get_legal_actions(self.index)
        if invaders:
            invader_distances = [(self.get_maze_distance(my_position, i.get_position()), i.get_position()) for i in invaders]
            closest_distance, pacman_position = min(invader_distances)

        # Side goal: Chase invaders if our agent is a ghost and the e Pacman is nearby
        if invaders and not game_state.get_agent_state(self.index).is_pacman:
            if game_state.get_agent_state(self.index).scared_timer == 0:
                invader_distances = [(self.get_maze_distance(my_position, i.get_position()), i.get_position()) for i in invaders]
                closest_distance, pacman_position = min(invader_distances)
                if closest_distance <= 3+game_state.get_score()*3 and not enemy_captured:
                    chase_action = self.a_star_search(my_position, pacman_position, walls)
                    if chase_action in legal_actions and chase_action != Directions.STOP:
                        return chase_action

        current_food_carried = game_state.get_agent_state(self.index).num_carrying

        # Check if Pacman is trapped
        if game_state.get_agent_state(self.index).is_pacman and self.last_food_carried == current_food_carried and defenders:
            self.chased_time += 1
            if self.chased_time >= 30:
                defender_distances = [(self.get_maze_distance(my_position, defender.get_position()), defender.get_position()) for defender in defenders]
                closest_distance, ghost_position = min(defender_distances)
                chase_action = self.a_star_search(my_position, ghost_position, walls)
                if chase_action in legal_actions and chase_action != Directions.STOP:
                    return chase_action
        else:
            self.chased_time = 0

        # Track previous food carried
        self.last_food_carried = current_food_carried

        # Main goal: Collect food and return safely
        action_values = [self.evaluate(game_state, action) for action in legal_actions]
        max_value = max(action_values)
        best_actions = [action for action, value in zip(legal_actions, action_values) if value == max_value]
        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != util.nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_position = successor.get_agent_state(self.index).get_position()
        food_positions = self.get_food(successor).as_list()
        capsule_positions = self.get_capsules(successor)

        # Treat capsules and food as equal targets, both scoring and survival are crucial
        target_positions = food_positions + capsule_positions
        features['successor_score'] = -len(target_positions)

        # Identify e agents and filter those currently defending 
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        defenders = [e for e in enemies if not e.is_pacman and e.get_position() is not None and e.scared_timer == 0]

        if target_positions:
            adjusted_distances = []
            for target in target_positions:
                target_distance = self.get_maze_distance(my_position, target)
                # If defenders are close to this food, penalize it heavily to avoid risky paths
                if defenders:
                    defender_distances = [self.get_maze_distance(target, d.get_position()) for d in defenders]
                    closest_defender_distance = min(defender_distances)
                    if closest_defender_distance <= 3:
                        target_distance += 100

                adjusted_distances.append(target_distance)

            # Choose the safest food target by taking the minimum adjusted distance
            features['food_distance'] = min(adjusted_distances)
        
        # Track how close the nearest ghost is
        if defenders:
            defender_distances = [self.get_maze_distance(my_position, d.get_position()) for d in defenders]
            closest_distance = min(defender_distances)
            features['ghost_distance'] = closest_distance
        else:
            features['ghost_distance'] = 0

        # Incentivize returning home when carrying food, while favoring paths that keep distance from ghosts
        food_carried = successor.get_agent_state(self.index).num_carrying
        home_distance = self.get_maze_distance(my_position, self.start)
        features['home_distance'] = max(0, home_distance * food_carried - features['ghost_distance'])

        # Penalize stopping or reversing direction (discourages indecision/backtracking)
        if action == Directions.STOP:
            features['stop'] = 1
        reverse_direction = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse_direction:
            features['reverse'] = 1
        
        # Trap detection
        forbidden_position = game_state.get_agent_state(self.index).get_position()
        walls = game_state.get_walls()
        is_trap = threshold_limit_search(my_position, forbidden_position, walls, capsule_positions, threshold=12)
        if 0<features["ghost_distance"] and features["ghost_distance"]<=3 and is_trap:
            features["trap"] = 1
        else:
            features["trap"] = 0

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,
            'food_distance': -2,
            'ghost_distance': 1,
            'home_distance': -1,
            'stop': -100,
            'reverse': -2,
            'trap': -100
        }

class DefensiveReflexAgent(CaptureAgent):
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.last_invader_position = None
        # Determine the middle of the map to know which side we defend
        width = game_state.get_walls().width
        mid = width // 2
        self.boundary_x = mid - 1 if self.red else mid
        # Track the food we are responsible for defending at the start
        self.last_defend_food = self.get_food_you_are_defending(game_state).as_list()
        # Keep memory of the last missing food, it helps detect invaders
        self.last_missing_food = None

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        enemy_captured = (best_actions == Directions.STOP)
        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != util.nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights
    
    def openness(self, food, walls):
        legal_neighbors = Actions.get_legal_neighbors(food, walls)
        return len(legal_neighbors)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_position = successor.get_agent_state(self.index).get_position()
        walls = game_state.get_walls()

        # Mark whether we are defending (Ghost) or invading (Pacman)
        features['on_defense'] = int(not successor.get_agent_state(self.index).is_pacman)

        # Identify e agents and filter those currently invading 
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        features['invader_count'] = len(invaders)
        
        invader_position = None
        if invaders:
            # Track distance to the closest invader
            invader_distances = [(self.get_maze_distance(my_position, i.get_position()), i.get_position()) for i in invaders]
            closest_distance, invader_position = min(invader_distances)
            features['invader_distance'] = closest_distance
            
        if action == Directions.STOP:
            features['stop'] = 1
            # Trap the enemy pacman, set the enemy_captured flag as true for OffensiveReflexAgent
            if invader_position is not None:
                features['stop'] = -100*threshold_limit_search(invader_position, my_position, walls, None, threshold=12)
              
        reverse_direction = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse_direction:
            features['reverse'] = 1

        # Choose a defensive food target near the boundary, preferring positions with more open escape routes
        defend_food_positions = self.get_food_you_are_defending(successor).as_list()
        if defend_food_positions:
            closest_distance = min(defend_food_positions, key=lambda pos: (abs(pos[0] - self.boundary_x), -self.openness(pos, walls)))
            features['boundary_distance'] = self.get_maze_distance(my_position, closest_distance)
        else:
            features['boundary_distance'] = 0

        # Detect missing food and remember its location
        missing_food_positions = [f for f in self.last_defend_food if f not in defend_food_positions]
        self.last_defend_food = defend_food_positions
        if missing_food_positions:
            missing_food_distances = [(self.get_maze_distance(my_position, food), food) for food in missing_food_positions]
            _, nearest_food = min(missing_food_distances)
            self.last_missing_food = nearest_food

        # Use the location of missing food as a clue to chase the e Pacman
        if self.last_missing_food:
            features['missing_food_distance'] = self.get_maze_distance(my_position, self.last_missing_food)
        else:
            features['missing_food_distance'] = 0

        return features

    def get_weights(self, game_state, action):
        return {
            'invader_count': -1000,
            'on_defense': 10,
            'invader_distance': -100,
            'missing_food_distance': -10,
            'boundary_distance': -1,
            'stop': -100,
            'reverse': -2,
        }

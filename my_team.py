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
from game import Directions
from util import nearest_point
from util import PriorityQueue
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

class OffensiveReflexAgent(CaptureAgent):
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
    
    def a_star_search(self, game_state, target):
        start = game_state.get_agent_position(self.index)
        frontier = PriorityQueue()
        frontier.push((start, []), 0)
        explored = set()

        while not frontier.is_empty():
            pos, path = frontier.pop()
            # If we've reached the target, return the first move that got us here
            if pos == target:
                return path[0] if path else Directions.STOP
            if pos in explored:
                continue
            explored.add(pos)

            # Expand possible moves from the current position
            for action in game_state.get_legal_actions(self.index):
                successor = game_state.generate_successor(self.index, action)
                new_pos = successor.get_agent_position(self.index)
                if new_pos not in explored:
                    new_path = path + [action]
                    # Cost combines real path length and Manhattan distance
                    cost = len(new_path) + manhattan_distance(new_pos, target)
                    frontier.push((new_pos, new_path), cost)

        # If no path exists, just stop
        return Directions.STOP

    def choose_action(self, game_state):
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        my_pos = game_state.get_agent_state(self.index).get_position()

        # Side goal: Chase invaders if our agent is a ghost and the enemy Pacman is nearby
        if invaders and not game_state.get_agent_state(self.index).is_pacman:
            dists = [(self.get_maze_distance(my_pos, a.get_position()), a.get_position()) for a in invaders]
            min_dist, target = min(dists)
            if min_dist <= 4:  
                chase_action = self.a_star_search(game_state, target)
                if chase_action != Directions.STOP:   
                    return chase_action
        
        # Main goal: Collect food and return safely      
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

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
        my_pos = successor.get_agent_state(self.index).get_position()
        food_list = self.get_food(successor).as_list()
        capsules = self.get_capsules(successor)

        # Treat capsules and food as equal targets, both scoring and survival are crucial
        food_list = food_list + capsules
        features['successor_score'] = -len(food_list)

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None and a.scared_timer == 0]

        if food_list:
            adjusted_distances = []
            for food in food_list:
                dist_food = self.get_maze_distance(my_pos, food)

                # If defenders are close to this food, penalize it heavily to avoid risky paths
                if defenders:
                    ghost_dists = [self.get_maze_distance(food, d.get_position()) for d in defenders]
                    if min(ghost_dists) <= 2:
                        dist_food += 10
                adjusted_distances.append(dist_food)
            
            # Choose the safest food target by taking the minimum adjusted distance
            features['food_distance'] = min(adjusted_distances)

        # Track how close the nearest ghost is
        if defenders:
            min_distance = min([self.get_maze_distance(my_pos, d.get_position()) for d in defenders])
            features['ghost_distance'] = min_distance
        else:
            features['ghost_distance'] = 0

        # Incentivize returning home when carrying food, while favoring paths that keep distance from ghosts
        carrying = successor.get_agent_state(self.index).num_carrying
        features['home_distance'] = max(0, self.get_maze_distance(my_pos, self.start) * carrying - features['ghost_distance'])

        # Penalize stopping or reversing direction (discourages indecision/backtracking)
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,
            'food_distance': -2,
            'ghost_distance': 1,
            'home_distance': -2,
            'stop': -100,
            'reverse': -2
        }


class DefensiveReflexAgent(CaptureAgent):
    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)

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
    
    def openness(self, food, game_state):
            neighbors = [(food[0]+dx, food[1]+dy) for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]]
            return sum(1 for n in neighbors if not game_state.has_wall(n[0], n[1]))

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Mark whether we are defending (Ghost) or invading (Pacman)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Identify enemy agents and filter those currently invading 
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['invader_count'] = len(invaders)

        if invaders:
            # Track distance to the closest invader
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

            # If scared, flip the sign so the agent prefers to increase distance instead of chasing
            if my_state.scared_timer > 0:
                features['invader_distance'] *= -1

        # Penalize stopping or reversing direction (discourages indecision/backtracking)
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # Choose a defensive food target near the boundary, preferring positions with more open escape routes
        defend_food = self.get_food_you_are_defending(successor).as_list()
        if defend_food:
            closest = min(defend_food, key=lambda p: (abs(p[0] - self.boundary_x), -self.openness(p, game_state)))
            features['boundary_distance'] = self.get_maze_distance(my_pos, closest)
        else:
            features['boundary_distance'] = 0

        # Detect missing food and remember its location
        missing = [f for f in self.last_defend_food if f not in defend_food]
        self.last_defend_food = defend_food
    
        if missing:
            self.last_missing_food = missing[0]

        # Use the location of missing food as a clue to chase the enemy Pacman
        if self.last_missing_food:
            features['missing_food_distance'] = self.get_maze_distance(my_pos, self.last_missing_food)
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

from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
import random
import json
import os

class CustomPlayer(BasePokerPlayer):

    def __init__(self, is_training=False, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2, weights_file=None, use_json=True):
        self.is_training = is_training
        self.uuid = None
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        self.weights = {
            'hand_strength': 0.4,
            'pot_odds': 0.3,
            'position': 0.2,
            'aggression': 0.1,
        }
        
        self.weights_file = weights_file or 'player_weights.json'
        self.use_json = use_json
        
        self._load_weights()
        
        self.last_state = None
        self.last_action = None
        self.last_stack = 0
        
    def declare_action(self, valid_actions, hole_card, round_state):
        if self.is_training and random.random() < self.exploration_rate:
            random_choice = random.random()
            if random_choice < 0.6:
                action = 'call'
            elif random_choice < 0.8:
                action = 'raise'
            else:
                action = 'fold'
                
            valid_action_names = [a['action'] for a in valid_actions]
            if action not in valid_action_names:
                action = 'call' if 'call' in valid_action_names else 'fold'
            
            self._record_state(valid_actions, hole_card, round_state, action)
            return action
            
        win_rate = self._get_hand_strength(hole_card, round_state.get('community_card', []))
        
        pot_odds = self._calculate_pot_odds(valid_actions, round_state)
        
        position = self._get_position_value(round_state)
        
        aggression = self._get_aggression_value(round_state)
        
        decision_value = (
            self.weights['hand_strength'] * win_rate +
            self.weights['pot_odds'] * pot_odds +
            self.weights['position'] * position +
            self.weights['aggression'] * aggression
        )
        
        action = 'fold'
        valid_action_names = [a['action'] for a in valid_actions]
        
        if decision_value > 0.6:
            action = 'raise' if 'raise' in valid_action_names else 'call'
        elif decision_value > 0.3:
            action = 'call'
        else:
            action = 'fold' if 'fold' in valid_action_names else 'call'
        
        self._record_state(valid_actions, hole_card, round_state, action)
        
        return action
        
    def _record_state(self, valid_actions, hole_card, round_state, action):
        if self.is_training:
            current_state = (hole_card, valid_actions, round_state)
            
            current_stack = 0
            for player in round_state['seats']:
                if player['uuid'] == self.uuid:
                    current_stack = player['stack']
                    break
            
            if self.last_state is not None:
                reward = current_stack - self.last_stack
                
                win_rate = self._get_hand_strength(self.last_state[0], self.last_state[2].get('community_card', []))
                pot_odds = self._calculate_pot_odds(self.last_state[1], self.last_state[2])
                position = self._get_position_value(self.last_state[2])
                aggression = self._get_aggression_value(self.last_state[2])
                
                self._update_weights(reward, {
                    'hand_strength': win_rate,
                    'pot_odds': pot_odds,
                    'position': position,
                    'aggression': aggression
                })
            
            self.last_state = current_state
            self.last_action = action
            self.last_stack = current_stack
    
    def _update_weights(self, reward, features):
        effective_lr = self.learning_rate * 0.1
        
        capped_reward = max(min(reward, 100), -100)
        
        for key in self.weights:
            if key in features:
                feature_contribution = features[key] - 0.5
                
                update = effective_lr * capped_reward * feature_contribution * 0.001
                self.weights[key] += update
                
                self.weights[key] = max(0.05, min(0.7, self.weights[key]))
        
        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total
    
    def _get_hand_strength(self, hole_card, community_card):
        if not community_card:
            rank_counts = {}
            for card in hole_card:
                rank = card[1]
                rank_counts[rank] = rank_counts.get(rank, 0) + 1
            
            if len(rank_counts) == 1:
                return 0.8
            
            high_cards = ['A', 'K', 'Q', 'J', 'T']
            high_card_count = sum(1 for rank in rank_counts if rank in high_cards)
            
            suited = hole_card[0][0] == hole_card[1][0]
            
            if high_card_count == 2:
                return 0.7 if suited else 0.6
            elif high_card_count == 1:
                return 0.5 if suited else 0.4
            else:
                return 0.3 if suited else 0.2
        else:
            try:
                hole_card_obj = gen_cards(hole_card)
                community_card_obj = gen_cards(community_card)
                
                win_rate = estimate_hole_card_win_rate(
                    nb_simulation=100,
                    nb_player=2,
                    hole_card=hole_card_obj,
                    community_card=community_card_obj
                )
                return win_rate
            except Exception:
                return 0.5
    
    def _calculate_pot_odds(self, valid_actions, round_state):
        try:
            call_action = [a for a in valid_actions if a['action'] == 'call'][0]
            call_amount = call_action.get('amount', 0)
            
            pot_amount = round_state['pot']['main']['amount']
            
            if call_amount == 0:
                return 1.0
                
            pot_odds = pot_amount / (pot_amount + call_amount)
            return pot_odds
        except (KeyError, IndexError, ZeroDivisionError):
            return 0.5
    
    def _get_position_value(self, round_state):
        try:
            seats = round_state['seats']
            player_pos = -1
            
            for i, player in enumerate(seats):
                if player['uuid'] == self.uuid:
                    player_pos = i
                    break
            
            active_players = sum(1 for p in seats if p['state'] == 'participating')
            if active_players <= 1:
                return 0.5
                
            relative_pos = player_pos / active_players
            return relative_pos
        except (KeyError, ZeroDivisionError):
            return 0.5
    
    def _get_aggression_value(self, round_state):
        try:
            street_history = round_state.get('action_histories', {})
            
            if not street_history:
                return 0.5
                
            current_street = round_state['street']
            if current_street in street_history:
                actions = street_history[current_street]
                raises = sum(1 for a in actions if a.get('action') == 'RAISE')
                calls = sum(1 for a in actions if a.get('action') == 'CALL')
                folds = sum(1 for a in actions if a.get('action') == 'FOLD')
                
                if raises > 0 and folds > 0:
                    return 0.7
                elif raises > 0 and calls > raises:
                    return 0.3
            
            return 0.5
        except KeyError:
            return 0.5
    
    def receive_game_start_message(self, game_info):
        if 'uuid' in game_info:
            self.uuid = game_info['uuid']
        elif 'player_id' in game_info:
            self.uuid = game_info['player_id']
        else:
            for player in game_info.get('seats', []):
                if player.get('name') == "TrainingAgent":
                    self.uuid = player.get('uuid')
                    break

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        if self.is_training and self.last_state is not None:
            current_stack = 0
            for player in round_state['seats']:
                if player['uuid'] == self.uuid:
                    current_stack = player['stack']
                    break
            
            reward = current_stack - self.last_stack
            
            if self.last_state:
                win_rate = self._get_hand_strength(self.last_state[0], self.last_state[2].get('community_card', []))
                pot_odds = self._calculate_pot_odds(self.last_state[1], self.last_state[2])
                position = self._get_position_value(self.last_state[2])
                aggression = self._get_aggression_value(self.last_state[2])
                
                self._update_weights(reward, {
                    'hand_strength': win_rate,
                    'pot_odds': pot_odds,
                    'position': position,
                    'aggression': aggression
                })
            
            self.last_state = None
            self.last_action = None
            
            if random.random() < 0.1:
                self._save_weights()
    
    def _save_weights(self, filename=None):
        try:
            if filename is None:
                filename = self.weights_file
                
            if self.use_json or filename.endswith('.json'):
                with open(filename, 'w') as f:
                    json.dump(self.weights, f, indent=4)
                    
        except Exception as e:
            print(f"Error saving weights to {filename}: {e}")
    
    def _load_weights(self):
        try:
            filename = self.weights_file
            
            if os.path.exists(filename):
                if self.use_json or filename.endswith('.json'):
                    with open(filename, 'r') as f:
                        loaded_weights = json.load(f)
                        
                        if "_COMMENTS" in loaded_weights:
                            del loaded_weights["_COMMENTS"]
                            
                        self.weights = loaded_weights
                        
                total = sum(self.weights.values())
                if total > 0:
                    for key in self.weights:
                        self.weights[key] /= total
                        
        except Exception as e:
            print(f"Error loading weights from {self.weights_file}: {e}")
            print("Using default weights instead")

def setup_ai():
    return CustomPlayer(is_training=False)
##
## =======================================================
## Mark Erenberg 
## Reinforcement Learning TF Tutorial
## =======================================================
##

import sys
sys.path.append('C:/Users/marke/Downloads/,CS3243/Project/mypoker-master/')

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import random
import math
import pypokerengine
from pypokerengine.players import BasePokerPlayer
from pypokerengine.api.emulator import Emulator
from pypokerengine.utils.game_state_utils import restore_game_state, attach_hole_card_from_deck, attach_hole_card
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from collections import defaultdict
import pprint


class Model:
      def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()
    
      def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 50, activation=tf.nn.relu)
        fc2 = tf.layers.dense(fc1, 50, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc2, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

      def predict_one(self, state, sess):
        # state is a 4-dimension vector
        state = np.array(state)
        return sess.run(self._logits, feed_dict={self._states:
                                                     state.reshape(1, self.num_states)})

      def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

      def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

      @property
      def num_states(self):
         return self._num_states

      @property
      def num_actions(self):
         return self._num_actions

      @property
      def batch_size(self):
         return self._batch_size

      @property
      def var_init(self):
         return self._var_init
    
    
class Memory:
      def __init__(self, max_memory):
          self._max_memory = max_memory
          self._samples = []
    
      def add_sample(self, sample):
          self._samples.append(sample)
          if len(self._samples) > self._max_memory:
              self._samples.pop(0)
    
      def sample(self, no_samples):
          if no_samples > len(self._samples):
              return random.sample(self._samples, len(self._samples))
          else:
              return random.sample(self._samples, no_samples)
          
  
def emulate(hole_card, round_state):
      # 1. Set game settings in Emulator
      emulator = Emulator()
      sb_amount = round_state['small_blind_amount']
      # emulator(nb_player,max_rounds,sb_amount,ante)
      emulator.set_game_rule(2, 10, sb_amount, 1)
          
      # 2. Setup Gamestate object      

      game_state = restore_game_state(round_state)
      
      # Attach hole_cards for each player (at random for opponent)
      game_state = attach_hole_card(game_state,round_state['seats'][0]['uuid'], gen_cards(hole_card))
      game_state = attach_hole_card_from_deck(game_state,round_state['seats'][1]['uuid'])
      
      # 3. Run simulation and get updated GameState object
      # updated_state, events = emulator.apply_action(game_state, "call", 10)
        
      return game_state, emulator
    
    

class QPlayer(BasePokerPlayer):
  
  def __init__(self):
    self.curr_round = 0 # to store round info; avoid recalculating round info
    self.curr_cards = {} # store card info for current round
    self.suits = [0,0,0,0] # store suit infor for current round
    self.combo = 0 # store current card combination 
    self.opp_actions = [0,0] # store oppenent actions [number of calls, number of raises]
    self.epsilon = 1 # to allow greedy choosing for the first 100 rounds 
    self.stack = 10000 # store current stack per round; to trace back reward per round
    self.max_stack = 10000
    self.reward = 0 # store reward
    self.tot_reward = 0
    self.steps = 0
    self.reward_stores = []
    self.max_stack_stores = []
    # Global variables to be used for Tensorflow model
    self.num_states = 4                  # Dimension of observation space
    self.num_actions = 3
    self.MAX_EPSILON = 1                 # Max amount of exploration (randomness in action selection)
    self.MIN_EPSILON = 0.01              # Min amont of exploration
    self.LAMBDA = 0.1                    # Learning rate
    self.GAMMA = 0.9                     # Penalty of delayed reward (discounts future state rewards)
    self.BATCH_SIZE = 50    
    self.model = Model(self.num_states, self.num_actions, self.BATCH_SIZE)
    self.memory = Memory(100000)
    self.sess = tf.Session()
    #self.saver = tf.train.Saver({"states": self.model._states})
    

  def reset_round(self): # reset the round
    #self.curr_round = 0
    self.combo = 0
    self.curr_cards = {}
    self.suits = [0,0,0,0]
    self.opp_actions = [0,0]
    
  def card_to_score(self, card): # makes it easier to track score
      suit = card[0]
      num = card[1]
      d = {'D':0, 'C':1, 'H':2, 'S':3, 'T':10, 'J':11, 'Q':12, 'K': 13, 'A':14}
      try:
          num_score = int(num) 
      except:
          num_score = d[num]
      suit_score = d[suit]
      return [num_score, suit_score]

  def hand_combo(self, cards): # find card combinations in hole cards
      for card in cards:
          num = card[0]
          suit = card[1]
          if num >= 10 and self.combo == 0:
            self.combo = 1 # HC
          if num not in self.curr_cards:
            self.curr_cards[num] = []
          self.curr_cards[num].append(suit)
          if len(self.curr_cards[num]) == 2:
            self.combo = 2 # 1P
          self.suits[suit] += 1

  def update_combo(self, community, street): # update combinations accordingly
        '''
        Combination scores are as such:
        0=NA;1=Highcard;2=one-pair;3=two-pairs;4=three-of-a-kind;5=straight;
        6=flush;7=fullhouse;8=four-of-a-kind;9=straightflush;10=royalflush
        '''
        for rw_card in community:
          card = self.card_to_score(rw_card)
          num = card[0]
          suit = card[1]
          if num not in self.curr_cards:
            self.curr_cards[num] = []
          self.curr_cards[num].append(suit)
          
          if len(self.curr_cards[num]) == 2:
            if self.combo < 2:
              self.combo = 2 # one-pair
            elif self.combo == 2:
              self.combo = 3 # two-pairs
            elif self.combo == 4:
              self.combo = 7 # fullhouse
          
          elif len(self.curr_cards[num]) == 3:
            if self.combo == 2:
              self.combo = 4 # three-of-a-kind
            elif self.combo == 3:
              self.combo = 7 # fullhouse
    
          elif len(self.curr_cards[num]) == 4:
            self.combo = 8 # four-of-a-kind
    
          self.suits[suit] += 1
    
        sorted_cards = sorted(list(self.curr_cards.keys()))
        for suit in self.suits:
          if suit >= 5 and self.combo < 6:
            self.combo = 6 # flush
        if street == 1:
          if len(sorted_cards) == 5:
            self.combo_helper(sorted_cards) 
        if street == 2:
          if len(sorted_cards) == 5:
            self.combo_helper(sorted_cards)
          elif len(sorted_cards) == 6:
            self.combo_helper(sorted_cards[:5])
            self.combo_helper(sorted_cards[1:6])
        if street == 3:
          if len(sorted_cards) == 5:
            self.combo_helper(sorted_cards)
          elif len(sorted_cards) == 6:
            self.combo_helper(sorted_cards[:5])
            self.combo_helper(sorted_cards[1:6])
          elif len(sorted_cards) == 7:
            self.combo_helper(sorted_cards[:5])
            self.combo_helper(sorted_cards[1:6])
            self.combo_helper(sorted_cards[1:7])

  def combo_helper(self, cards):
        suits = [0,0,0,0]
        if cards[len(cards)-1]-cards[0] == 4:
          if self.combo < 5: self.combo = 5 # straight
          for card in cards:
            for suit in self.curr_cards[card]:
              suits[suit] += 1
              if suits[suit] == 5:
                self.combo = 9 # straightflush
                if cards[0] == 10:
                  self.combo = 10 # royalflush  
                   
  def update_opp_actions(self, action_histories): # update oppenent actions
    if len(action_histories) != 0:
      latest_history = action_histories[len(action_histories)-1]
      if latest_history['uuid'] != self.uuid:
        if latest_history['action'] == 'CALL':
          self.opp_actions[0] += 1
        elif latest_history['action'] == 'RAISE':
          self.opp_actions[1] += 1                 

    
  def find_street_index(self, street): # create search index for street
        if street == 'preflop': return 0
        elif street == 'flop': return 1
        elif street == 'turn': return 2
        else: return 3
    
  def find_pot_index(self, pot): # create search index for pot
          return int(pot/20)
    
  def find_opp_actions_index(self): # create search index for opponent's actions; if the opponent call more or less
        if self.opp_actions[0] >= self.opp_actions[1]:
          return 0
        else:
          return 1  
      
  def extract_state(self, round_state):
    # accessing state info and extracting relevant info
    street = round_state['street'] # street
    pot = round_state['pot']['main']['amount'] # pot amount 

    pot_index = self.find_pot_index(pot)
    street_index = self.find_street_index(street) 
    combo_index = self.combo
    opp_actions_index = self.find_opp_actions_index()
    
    return [pot_index, street_index, combo_index, opp_actions_index]   

#####################################################################################
                        #    New Declare Action function #
#####################################################################################

       
  def declare_action(self, valid_actions, hole_card, round_state):  
      print(valid_actions)
      print(hole_card)
      print(round_state)
      state = self.extract_state(round_state)   
      #env = restore_game_state(round_state)
      game_state, emulator = emulate(hole_card, round_state)
      round_count = round_state['round_count']
      small_blind_pos = round_state['small_blind_pos']
      community = round_state['community_card'] # community cards
      street = round_state['street'] # street
      action_histories = round_state['action_histories'] # track opponent's actions
      cards = list(map(lambda card: self.card_to_score(card), hole_card)) # hole cards
      
      #print(round_count)
      #print(self.curr_round)
      if self.curr_round == 0:
          self.sess.run(self.model.var_init)
          print('RAN VAR INIT')

      
      # IF A NEW ROUND HAS BEGUN:
      #print(round_count != self.curr_round)
      if round_count != self.curr_round:   
          if round_count > self.curr_round: # only update Qtable when new game round while running testperf 
            self.epsilon *= 0.75 # scale accordingly; to reduce greediness as agent learns
    
          # correct for bigblind and smallblind bets when starting new round
          for seat in round_state['seats']:
            if seat['uuid'] == self.uuid:
              if small_blind_pos == 0: curr_stack = seat['stack'] + 10
              else: curr_stack = seat['stack'] + 20
            
          # find reward for previous round and update stack
          self.reward = curr_stack - self.stack
          self.tot_reward += self.reward
          self.stack = curr_stack
    
          self.curr_round = round_count # update current round
          self.reset_round() # reset state info
          self.hand_combo(cards) # find combination of hand cards    
          
          
      # update combo hands accordingly to street 
      if street == 'flop':
          self.update_combo(community, 1) 
        
      elif street == 'turn':
          self.update_combo([community[len(community)-1]], 2)  
          self.update_opp_actions(action_histories['flop']) # update previous street info; if opp calls, declare action does not get called
        
      elif street == 'river':
          self.update_combo([community[len(community)-1]], 3)
          self.update_opp_actions(action_histories['turn']) # update previous street info; if opp calls, declare action does not get called

      self.update_opp_actions(action_histories[street]) # update previous action info
      
      # Select best action using TF
      action_index = self._choose_action(state)
      
    
      # Define step function (using Emulator)
      actions = emulator.generate_possible_actions(game_state)
      if (len(actions) < 3) & (action_index == 2):
          action_index = 1
      
      # bet_amount = actions[action_index]['amount'] if action_index != 2 else actions[1]['amount'] + 10
      updated_state, events = emulator.apply_action(game_state, actions[action_index]['action'], 10) 
      next_round_state = events[0]['round_state']
      next_state = self.extract_state(next_round_state)
    
      self.max_stack = next_round_state['seats'][0]['stack'] if next_round_state['seats'][0]['stack'] > self.stack else self.stack
    
      self.memory.add_sample((state, action_index, self.reward, next_state))
      
      # Replay the last action to train the model
      batch = self.memory.sample(self.model.batch_size)
      states = np.array([val[0] for val in batch])
      next_states = np.array([(np.zeros(self.model.num_states)
                               if val[3] is None else val[3]) for val in batch])
      # predict Q(s,a) given the batch of states
      q_s_a = self.model.predict_batch(states, self.sess)
      # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
      q_s_a_d = self.model.predict_batch(next_states, self.sess)
      # setup training arrays
      x = np.zeros((len(batch), self.model.num_states))
      y = np.zeros((len(batch), self.model.num_actions))
      for i, b in enumerate(batch):
          state, action, reward, next_state = b[0], b[1], b[2], b[3]
          # get the current q values for all actions in state
          current_q = q_s_a[i]
          # update the q value for action
          if next_state is None:
              # in this case, the game completed after action, so there is no max Q(s',a')
              # prediction possible
              current_q[action] = reward
          else:
              current_q[action] = reward + self.GAMMA * np.amax(q_s_a_d[i])
          x[i] = state
          y[i] = current_q
      self.model.train_batch(self.sess, x, y)
      
    
      # exponentially decay the eps value
      self.steps += 1
      self.epsilon = self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON) \
                                * math.exp(-self.LAMBDA * self.steps)
    
      # move the agent to the next state and accumulate the reward
      state = next_state

      # if the game is done, break the loop
      self.reward_stores.append(self.tot_reward)
      self.max_stack_stores.append(self.max_stack)
      
      
      if action_index == 0:
        action = 'fold'
      elif action_index == 1:
        action = 'call'
      else:
        action = 'raise'
      
      return action

  def _choose_action(self, state_vector):
        if random.random() < self.epsilon:
            r = random.randint(0, self.num_actions - 1)
            return r
        else:
            return np.argmax(self.model.predict_one(state_vector, self.sess))

  @property
  def reward_store(self):
      return self.reward_stores

  @property
  def max_stack_store(self):
      return self.max_stack_stores    
  
  def receive_game_start_message(self, game_info):
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass

def setup_ai():
  return QPlayer_R()
    

    
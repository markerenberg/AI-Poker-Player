from pypokerengine.players import BasePokerPlayer
import random as rand
import pprint

class RandomPlayer(BasePokerPlayer):

  def check_str8(self, sorted_lst):
    count = 0
    for i in range(len(sorted_lst)-1):
      if sorted_lst[i] - sorted_lst[i+1] == 1:
        count += 1
      else:
        break
    if count >= 4:
      return True
    else:
      return False

  def h(self, cards, community, pot):
    num_d = {}
    suits_d = {}
    pairs = False
    threes = False

    for card in cards:
      num = card[0]
      suit = card[1]
      if num not in num_d:
        num_d[num] = 1
      else:
        num_d[num] += 1
      if suit not in suits_d:
        suits_d[suit] = 1
      else:
        suits_d[suit] += 1
    for card in community:
      card = self.card_to_score (card)
      num = card[0]
      suit = card[1]
      if num not in num_d:
        num_d[num] = 1
      else:
        num_d[num] += 1
      if suit not in suits_d:
        suits_d[suit] = 1
      else:
        suits_d[suit] += 1

    score = 0
    for num_key in num_d:
      score += num_key * num_d[num_key] * 10
      if num_d[num_key] == 2:
        score += 1000 # pairs
        pairs = True
        print('Pairs',num_key)
        if threes:
          score += 4000 # full hse
          print('House')
      if num_d[num_key] == 3:
        score += 3000 # 3 kinds
        threes = True
        print('3Kinds', num_key)
        if pairs:
          score += 3000 # full hse
          print('House')
      if num_d[num_key] == 4: 
        score += 7000 # 4 kinds
        print('4Kinds', num_key)
    lst = list(num_d.keys())
    lst.sort(reverse = True)
    str8 = self.check_str8(lst)
    if  str8 == 1:
      score += 4000 # straight
      print('Straight')
    for suit in suits_d:
      score += suit
      if suits_d[suit] >= 5: # flush
        score += 5000
        print('Flush')
        if len(lst) >= 4:
          if lst[0] == 14 and lst[4] == 10:
            score += 1000
            print('Royal')
    #print("num_d:", num_d)
    #print("suits_d:", suits_d)
    #print("score:", score)
    if score < 140:
      return -10000
    return score

  def minimax_search(self, pot, depth, player, cards, community):
    if depth == 1:
      if player == 'MAX':
        action = max([(0,-pot), (1, self.h(cards, community, pot))], key = lambda x:x[1])
        #print(player, depth,action)
        return action
      else:
        action = min([(0,-pot), (1, self.h(cards, community, pot))], key = lambda x:x[1])
        #print(player, depth, action)
        return action
    else:
      if player == 'MAX':
        action = max([(0,-pot), (1, self.h(cards, community, pot)), (2, self.minimax_search(pot+10, depth-1, 'MIN', cards, community)[1])], key = lambda x:x[1])
        #print(player, depth, action)
        return action
      else:
        action = min([(0,-pot), (1, self.h(cards, community, pot)), (2, self.minimax_search(pot+10, depth-1, 'MAX', cards, community)[1])], key = lambda x:x[1])
        #print(player, depth, action)
        return action

  def card_to_score(self, card):
    suit = card[0]
    num = card[1]
    d = {'D':1, 'C':2, 'H':3, 'S':4, 'T':10, 'J':11, 'Q':12, 'K': 13, 'A':14}
    try:
      num_score = int(num)
    except:
      num_score = d[num]
    suit_score = d[suit]
    return [num_score, suit_score]

  def declare_action(self, valid_actions, hole_card, round_state):
    # valid_actions format => [raise_action_pp = pprint.PrettyPrinter(indent=2)
    #pp = pprint.PrettyPrinter(indent=2)
    #print("------------ROUND_STATE(RANDOM)--------")
    #pp.pprint(round_state)
    #print("------------HOLE_CARD----------")
    #pp.pprint(hole_card)
    #print("------------VALID_ACTIONS----------")
    #pp.pprint(valid_actions)
    #print("-------------------------------")
    cards = list(map(lambda card: self.card_to_score(card), hole_card))
    depth = 4
    player = 'MAX'
    pot = round_state['pot']['main']['amount']
    community = round_state['community_card']
    print(hole_card)
    #self.h(cards,community,pot)
    minimax_search_results = self.minimax_search(pot, depth, player, cards, community)
    call_action_info = valid_actions[minimax_search_results[0]]
    #call_action_info = valid_actions[1]
    action = call_action_info["action"]
    return action  # action returned here is sent to the poker engine

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
  return RandomPlayer()

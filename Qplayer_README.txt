So, these are the parameters of the state [pot, combo, street, opp_actions]

Pot:
	For pot, basically since this is limit texas, the pot can be easily discretized. I divided it by 2 because I realized it can only raise by 20 or 40
	Maximum pot is 1440 so the size of pot dimension is 72 if I'm not wrong

Combo:
	For combo is quite straightfoward, I added in Not Applicable (NA) and High Card as well. The size of combo dimension is 11

Street:
	Street is defined as such: flop = 1, turn = 2, river = 3. I initially added preflop = 0 but I realized that it results in a lot of folds
	It makes sense as more often than not, if the agent has a negative reward, it must have called or raised at the preflop level. Hence, it sorts
	of learns that it minimizes losses if it folds at the preflop level. But this is not taking into account the probablistic nature of poker, that
	it is possible to earn more money in the future if it calls or raises. So I decided to ignore the preflop street. The size of street dimension is 3.

Opp_actions:
	This is the hardest to discretize. So I decided to just learn if the opponent called more or raised more in the particular street. This is based on the fact that if the opponent has a good hand, it will raise more. If it has a passable hand, it will call more. If it folds, it is a terminal state and has no impact on our learning. The size of this dimension is 2.

Combining all the dimensions, the Qtable size is effectively 4752 states x 3 actions = 14,256 Q-values

To implement minimax, I was wondering if we can reduce the actions to just call and fold and for raise, we just pick the next state. But everything else remains pretty much the same. I'm not sure how reliable this method is, especially if the next state has not been seen before and the agent will pick a random amount? Not sure how that will work. 

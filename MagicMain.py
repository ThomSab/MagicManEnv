import random
import numpy as np 
import torch

from collections import deque

import MagicManDeck  as deck
from MagicManPlayer import AdversaryPlayer, TrainPlayer


"""
Currently all activations are passed in lists
so its
node.activation = [1] instead of
node.activation = 1

--> ToDo
"""

number_of_players = 4

class Game:

    def __init__(self,train_player,adversary_players):

        self.deck = deck
        if (n_players < 3) or (n_players > 6):
            print('INPUT ERROR: Invalid amount of players.')
        self.players = players 
        self.noorder_players = players #pls dont be a deep copy
        self.n_players = len(players)+1 #one learner
        self.max_rounds = int(60/n_players)
        self.current_round = 0
        self.bids = torch.zeros(self.n_players)
        self.trump = 0 #trump is red every time so the bots have a better time learning
        #score board
        random.shuffle(self.players)
        self.players = deque(self.players) #pick from a list of players
        self.turnorder_idx = 0
        self.bid_idx = 0
        self.turn_cards = []
        self.all_bid_completion = torch.zeros(self.n_players)
        
        #Observation Variables:
        self.obs = None
        self.info = None
        self.done = False
    
    def reset():
        self.current_round = 0
        self.bids = torch.zeros(self.n_players)
        random.shuffle(self.players)
        self.players = deque(self.players)
        self.turnorder_idx = 0
        self.turn_cards = []
        self.all_bid_completion = torch.zeros(self.n_players)
        
        self.obs = None
        self.info = None
        self.done = False
        
        raise NotImplementedError
        return observation
    
    def starting_player(self,starting_player):
        self.players.rotate(  -self.players.index(starting_player) )
    
    def init_turn_step(self):
        player_obs, r, done, info = self.turn_step(action = None)
        return player_obs, r, done, info
            
    def turn_step(self,action): #!!not round
        
        if action is not None:
            played_card = deck.deck[action]
            turn_cards.append(played_card)
            self.turnorder_idx +=1
        if action is None:
            assert self.turnorder_idx==0, f"The turn index is {self.turnorder_idx} | Should be 0."
        
        norm_bids = (self.bids-self.bids.mean())/(self.bids.std()+1e-5)
        
        while self.turnorder_idx <= (self.n_players-1):   
            #____________________________________________________
            #card playing
            
            player = self.players[self.turnorder_idx] #pls do not be a deep copy
            
            player_idx = torch.zeros(self.n_players)
            player_idx[self.turnorder_idx] = 1
            
            player_self_bid_completion = torch.tanh(player.round_score-player.current_bid)
            #this will be passed twice - once in an array of all players and once for self 
            #might be a problem because its biasing the decision
            #the agent will be told that there is another player with the exact same bid completion as him?
            #might also just not be a problem who knows
            
            n_cards = torch.zeros(self.max_rounds)
            n_cards[player.cards.sum()-1] = 1#how many cards there are in his hand
            
            played_cards = torch.zeros(60)
            for card in turn_cards:
                played_cards[deck.deck.index(card)] = 1

            current_suit = deck.legal(turn_cards,player.cards,self.trump)
                        
            player_obs = torch.cat((norm_bids,
                                    self.all_bid_completion,
                                    player_idx,
                                    player_self_bid_completion,
                                    n_cards,
                                    played_cards,
                                    owned_cards,
                                    current_suit),dim=0)
            
            if isinstance(player,AdversaryPlayer):
                #action is input not output!!!
                net_out = player.play(player_obs)
                action_idx = np.argmax(net_out)
                played_card = deck.deck[action_idx]
                turn_cards.append(played_card)
            else:
                return player_obs, r, self.done, self.info
            
            self.turnorder_idx +=1
            
        if self.turnorder_idx == (self.n_players):
            
            deck.turn_value(turn_cards,turn_cards,self.trump,current_suit) #turn value of the players cards    
            winner = self.players[[card.turn_value for card in turn_cards].index(max(card.turn_value for card in turn_cards))]        
            self.starting_player(winner)# --> rearanges the players of the player such that the winner is in the first position

            winner.round_score += 1 #winner of the suit    

            all_round_scores = torch.tensor([player.round_score for player in self.noorder_players])
            self.all_bid_completion = torch.tanh(all_round_scores-self.bids)
            self.turn_cards = []
            self.turnorder_idx = 0

    def init_bid_step(self):
                
        for player in self.noorder_players:
            player.round_score = 0
        
        random.shuffle(self.deck) 
        round_deck = self.deck.copy()
        
        for _ in range(self.current_round+1):
            for player in self.noorder_players:
                player.cards[deck.deck.index(round_deck.pop(-1))] = 1
                #pop not only removes the item at index but also returns it
                
        self.bids = torch.zeros(self.noorder_players.shape)#bids that are not yet given become 0 (bids are normalized so 0 is the expected bid)
        
        self.bid_idx = 0
        obs, r, d, info = bid_step(action=None)
        return obs, r, d, info


    def bid_step(self,action,lastround = False): #!!not turn

        
        if action is not None:
            self.bids.append(action)
        
        while self.bid_idx <= (self.n_players-1) #order is relevant

            player = self.players[self.bid_idx]
            
            n_cards = torch.zeros(self.max_rounds)
            n_cards[len(player.cards)] = 1#how many cards there are in his hand
            player_idx = torch.zeros(self.n_players)
            player_idx[self.players.index(player,0,n_players)] = 1#what place in the players the player has

            owned_cards = torch.zeros(60)
            for card in player.cards:
                owned_cards[deck.deck.index(card)] = 1 # cards in hand
            last_player_bool = torch.zeros(1)
            #if self.players.index(player) ==  3
            #    last_player_bool[0] = 1
            
            norm_bids = (self.bids-self.bids.mean())/(self.bids.std()+1e-5)

            player_obs = torch.cat((norm_bids,
                                    player_idx,
                                    n_cards,
                                    owned_cards,
                                    self.current_round),dim=0)
            
            if isinstance(player,AdversaryPlayer):
                self.bids.append(player.bid(player_obs))
            elif isinstance(player,TrainPlayer):
                return player_obs, r, d, info
            else:
                raise UserWarning (f"Player is {type(player)} not instance of either AdversaryPlayer or TrainPlayer")
                
        if self.bid_idx == (self.n_players):
            self.turn_idx = 0
            self.init_turn_step()
            
            if self.turn_idx == self.current_round
                for player in self.players:
                    if player.current_bid == player.round_score:
                        player.game_score += player.current_bid + 2 #ten point for every turn won and 20 for guessing right
                    else:
                        player.game_score -= abs(player.current_bid-player.round_score) #ten points for every falsly claimed suit

                for player in self.players:
                    player.clean_hand() #at this point all hands should be empty anyways
                    self.all_bid_completion = torch.zeros(self.n_players)
                    

    def step(self,action):
        observation,reward,done,info = None,None,None,None
        raise NotImplementedError
        return observation,reward,done,info

"""

The step function iterates through the rounds and turns

keep a self.counter for both and then return the done


"""
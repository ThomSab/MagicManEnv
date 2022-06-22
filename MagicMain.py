import random
import numpy as np 
import torch

from collections import deque

import MagicManDeck  as deck

"""
Currently all activations are passed in lists
so its
node.activation = [1] instead of
node.activation = 1

--> ToDo
"""

number_of_players = 4

class Game:
    """
    The Game Class
    No Documentation
    ______
        Recent Features:
        True Trump:
            In the original game, a new trump is chosen for each new round
            While it is fun, this has no real effect on the game
        No True Trump:
            The game keeps a single trump - in this case red, for no reason but red beeing trump "0"
            This way it should be a lot easier for the bots to learn
            Otherwise they would have to be a lot more complex bc the cards do different things each round
    
    """
    def __init__(self,players,deck):

        self.deck = deck
        if (n_players < 3) or (n_players > 6):
            print('INPUT ERROR: Invalid amount of players.')
        self.players = players 
        self.static_players = players #pls dont be a deep copy
        self.n_players = len(players)+1 #one learner
        self.max_rounds = int(60/n_players)
        self.current_round = 1
        self.bids = torch.zeros(self.n_players)
        self.trump = 0 #trump is red every time so the bots have a better time learning
        #score board
        random.shuffle(self.players)
        self.players = deque(self.players) #pick from a list of players
        
        #Observation Variables:
        self.
    
    def reset():
        self.current_round = 1
        self.bids = torch.zeros(self.n_players)
        random.shuffle(self.players)
        self.players = deque(self.players)

        observation = None
        raise NotImplementedError
        return observation
    
    def starting_player(self,starting_player):
        self.players.rotate(  -self.players.index(starting_player) )
    
    def turn(self): #!!not round
        turn_cards = []

        #____________________________________________________
        #PLAYING
        
        norm_bids = (self.bids-self.bids.mean())/(self.bids.std()+1e-5)
        
        all_round_scores = torch.tensor([player.round_score for player in self.static_players])
        all_bid_completion = torch.tanh(all_round_scores-self.bids)
        
        for player in self.players:
            
            #____________________________________________________
            #card playing
            
            player_idx = torch.zeros(self.n_players)
            player_idx[self.players.index(player,0,n_players)] = 1
            
            player_self_bid_completion = torch.tanh(player.round_score-player.current_bid)
            #this will be passed twice - once in an array of all players and once for self 
            #might be a problem because its biasing the decision
            #the agent will be told that there is another player with the exact same bid completion as him?
            #might also just not be a problem who knows
            
            n_cards = torch.zeros(self.max_rounds)
            n_cards[len(player.cards)-1] = 1#how many cards there are in his hand
            
            played_cards = torch.zeros(60)
            owned_cards = torch.zeros(60)
            for card in turn_cards:
                played_cards[deck.deck.index(card)] = 1
            for card in player.cards:
                owned_cards[deck.deck.index(card)] = 1


            current_suit = deck.legal(turn_cards,player.cards,self.trump)
            
            #action is input not output!!!
            turn_cards.append(player.play())
            
            
        deck.turn_value(turn_cards,turn_cards,self.trump,current_suit) #turn value of the players cards    
        winner = self.players[[card.turn_value for card in turn_cards].index(max(card.turn_value for card in turn_cards))]        
        self.starting_player(winner)# --> rearanges the players of the player such that the winner is in the first position

        winner.round_score += 1 #winner of the suit
        
  
    def round(self,round_idx,lastround = False): #!!not turn
        self.current_round = round_idx
        random.shuffle(self.deck) 
        round_deck = self.deck.copy()
        
        for _ in range(self.current_round):
            for player in self.players:
                player.cards.append(round_deck.pop(-1)) 
                #pop not only removes the item at index but also returns it
        
        for player in self.players:
            player.round_score = 0
        
        self.bids = []
        
        for player in self.players:

            #____________________________________________________
            #bid estimation BOTS
            player.bid_net.list_input(0,[0,0,0,0]) #s.t. bids that are not yet given become 0 (the average bid height)
            if self.bids:
                player.bid_net.list_input(0,[bid[0] - (len(player.cards)/4) for bid in self.bids])  #how high all the players bid minus the average expected amount of suits they win
            player.bid_net.list_input(4,[1 if _ == len(player.cards) else 0 for _ in range(self.max_rounds)])                 #how many cards there are in his hand
            player.bid_net.list_input(19,[1 if _ == self.players.index(player) else 0 for _ in range(number_of_players)])      #what place in the players the player has
            player.bid_net.list_input(23,[1 if _ == self.trump else 0 for _ in range(6)]) #which color is currently trump

            player.bid_net.list_input(29,[ 1 if card in player.cards else 0 for card in deck.deck  ])# cards in hand
            
            last_player_bool = (True if self.players.index(player) ==  3 else False)
            self.bids.append([player.bid(self.current_round,last_player = last_player_bool)])

            
        for turn in range(self.current_round):
            self.turn()
        
        for player in self.players:
            if player.current_bid == player.round_score:
                player.game_score += player.current_bid*10 + 20 #ten point for every turn won and 20 for guessing right

            else:
                player.game_score -= abs(player.current_bid-player.round_score)*10 #ten points for every falsly claimed suit

        for player in self.players:
            player.clean_hand()
                #at this point all hands should be empty anyways

    def step(self,action):
        observation,reward,done,info = None,None,None,None
        raise NotImplementedError
        return observation,reward,done,info

"""

The step function iterates through the rounds and turns

keep a self.counter for both and then return the done


"""
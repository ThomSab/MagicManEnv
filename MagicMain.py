import random
import numpy as np 
import torch

from collections import deque

import MagicNet as net
import MagicManDeck  as deck
from MagicManPlayer import AdversaryPlayer, TrainPlayer


class Game:

    def __init__(self,train_player,adversary_players,device=torch.device('cpu')):

        self.device = device
    
        self.round_deck = []
        self.players = []
        self.players = adversary_players
        self.train_player = train_player
        self.players.append(self.train_player)
        self.noorder_players = self.players #pls dont be a deep copy
        self.n_players = len(self.players)
        self.max_rounds = int(60/self.n_players)
        self.current_round = 0
        self.bids = torch.full(tuple([self.n_players]),float(round(self.current_round/self.n_players)))
        self.trump = 0 #trump is red every time so the bots have a better time learning
        random.shuffle(self.players)
        self.players = deque(self.players) #pick from a list of players
        self.turnorder_idx = 0
        self.bid_idx = 0
        self.turn_cards = []
        self.current_suit_idx = 5
        self.current_suit = torch.zeros(6)
        self.all_bid_completion = torch.zeros(self.n_players)
        
        #Observation Variables:
        self.bid_obs = None
        self.turn_obs = None
        self.r = 0
        self.info = None
        self.done = False
        
    
    def reset(self):
        self.round_deck = []
        self.bids = torch.full(tuple([self.n_players]),float(round(self.current_round/self.n_players)))
        random.shuffle(self.players)
        self.players = deque(self.players)
        self.turnorder_idx = 0
        self.turn_cards = []
        self.current_suit_idx = 5
        self.current_suit = torch.zeros(6)
        self.all_bid_completion = torch.zeros(self.n_players)
        
        self.bid_obs = None
        self.turn_obs = None
        self.r = 0
        self.info = None
        self.done = False

        for player in self.noorder_players:
            player.round_suits = 0
            player.game_score = 0
            player.cards_obj = []
            player.cards_tensor = torch.zeros(self.max_rounds)

        obs,r,done,info = self.init_bid_step()
        
        return obs, r, done, info
    
    def starting_player(self,starting_player):
        self.players.rotate(  -self.players.index(starting_player) )
    
    def init_turn_step(self):
        self.state = "TURN"
        self.turn_idx = 0
        self.current_suit_idx = 5
        self.current_suit = torch.zeros(6)
        self.turn_obs ,self.r, self.done, self.info = self.turn_step(action = None)
        return self.turn_obs, self.r, self.done, self.info
            
    def turn_step(self,action): #!!not round
        
        if action is not None:
            played_card = deck.deck[action]
            self.turn_cards.append(played_card)
            self.train_player.cards_obj.remove(played_card)
            self.r,self.info = 0,None
            self.turnorder_idx +=1
        if action is None:
            assert self.turnorder_idx==0, f"The turn index is {self.turnorder_idx} | Should be 0."
        
        norm_bids = (self.bids-self.bids.mean())/(self.bids.std()+1e-5)
        
        while True: #returns when necessary
            if not self.turnorder_idx == (self.n_players):
                #____________________________________________________
                #card playing
                
                player = self.players[self.turnorder_idx] #pls do not be a deep copy
                
                player_idx = torch.zeros(self.n_players)
                player_idx[self.turnorder_idx] = 1
                
                player_self_bid_completion = torch.tanh(torch.tensor(player.round_suits-player.current_bid))
                player_self_bid_completion = torch.unsqueeze(player_self_bid_completion,0)
                #this will be passed twice - once in an array of all players and once for self 
                #might be a problem because its biasing the decision
                #the agent will be told that there is another player with the exact same bid completion as him?
                #might also just not be a problem who knows
                        
                    
                n_cards = torch.zeros(self.max_rounds)
                n_cards[int(player.cards_tensor.sum()-1)] = 1 #how many cards there are in his hand

                self.current_suit_idx = deck.legal(self.turn_cards,player.cards_obj,self.trump)
                self.current_suit[self.current_suit_idx] = 1

                played_cards = torch.zeros(60)
                for card in self.turn_cards:
                    played_cards[deck.deck.index(card)] = 1
                player.cards_tensor = torch.zeros(60)
                for card in player.cards_obj:
                    if card.legal:
                        player.cards_tensor[deck.deck.index(card)] = 1

                player_obs = torch.cat((norm_bids,
                                        self.all_bid_completion,
                                        player_idx,
                                        player_self_bid_completion,
                                        n_cards,
                                        played_cards,
                                        player.cards_tensor,
                                        self.current_suit),dim=0).to(device=self.device)

                if isinstance(player,AdversaryPlayer):
                    #action is input not output!!!
                    net_out = player.play(player_obs)
                    action_idx = torch.argmax(net_out)
                    played_card = deck.deck[action_idx]
                    player.cards_obj.remove(played_card)
                    self.turn_cards.append(played_card)
                    self.turnorder_idx +=1
                elif isinstance(player,TrainPlayer):
                    
                    self.turn_obs = player_obs
                    return self.turn_obs, self.r, self.done, self.info
                else:
                    raise UserWarning (f"Player is {type(player)} not instance of either AdversaryPlayer or TrainPlayer")            
                
            if self.turnorder_idx == (self.n_players):
                
                deck.turn_value(self.turn_cards,self.trump,self.current_suit_idx) #turn value of the players cards    
                winner = self.players[[card.turn_value for card in self.turn_cards].index(max(card.turn_value for card in self.turn_cards))]        
                self.starting_player(winner)# --> rearanges the players of the player such that the winner is in the first position

                winner.round_suits += 1 #winner of the suit    

                all_round_scores = torch.tensor([player.round_suits for player in self.noorder_players])
                self.all_bid_completion = torch.tanh(all_round_scores-self.bids)
                self.turn_cards = []
                self.turnorder_idx = 0

                self.turn_idx += 1
            
                if self.turn_idx == self.current_round+1:
                    return self.conclude_step()

        raise UserWarning (f"Turn Step should have returned an Observation but has not")

    def init_bid_step(self):

        self.state = "BID"
        
        
        self.round_deck = deck.deck.copy()
        random.shuffle(self.round_deck) 
        
        for _ in range(self.current_round+1):
            for player in self.noorder_players:
                player.cards_obj.append(self.round_deck.pop(-1))
                #pop not only removes the item at index but also returns it
                
        self.bids = torch.full(tuple([self.n_players]),float(round(self.current_round/self.n_players)))#bids that are not yet given become 0 (bids are normalized so 0 is the expected bid)

        self.bid_idx = 0
        self.done = False
        self.r,self.info = 0,None
        
        self.bid_obs, self.r, self.done, self.info = self.bid_step(action=None)
        return self.bid_obs, self.r, self.done, self.info


    def bid_step(self,action,active_bid=False): # !!not turn
        
        if action is not None:
            self.bids[self.bid_idx] = action
            self.train_player.current_bid = action
            self.r,self.info = 0,None
            self.bid_idx += 1
        
        while self.bid_idx <= (self.n_players-1): # order is relevant

            player = self.players[self.bid_idx]
            
            n_cards = torch.zeros(self.max_rounds)
            n_cards[len(player.cards_obj)] = 1 # how many cards there are in his hand
            player_idx = torch.zeros(self.n_players)
            player_idx[self.players.index(player,0,self.n_players)] = 1 # what place in the players the player has

            player.cards_tensor = torch.zeros(60)
            for card in player.cards_obj:
                player.cards_tensor[deck.deck.index(card)] = 1 # cards in hand
            last_player_bool = torch.zeros(1)
            #if self.players.index(player) ==  3
            #    last_player_bool[0] = 1
            
            norm_bids = (self.bids-self.bids.mean())/(self.bids.std()+1e-5)
            
            player_obs = torch.cat((norm_bids,
                                    player_idx,
                                    n_cards,
                                    player.cards_tensor,
                                    torch.tensor([self.current_round])),dim=0).to(device=self.device)
            
            if isinstance(player,AdversaryPlayer):
                self.bids[self.bid_idx] = player.bid(player_obs)
                self.bid_idx += 1
            elif isinstance(player,TrainPlayer):
                if active_bid:
                    self.bid_obs = player_obs
                    return self.bid_obs, self.r, self.done, self.info
                else:
                    player.current_bid = round(self.current_round/self.n_players)
                    self.bids[self.bid_idx] = player.current_bid
                    self.bid_idx += 1
            else:
                raise UserWarning (f"Player is {type(player)} not instance of either AdversaryPlayer or TrainPlayer")
                
        if self.bid_idx == (self.n_players):
            
            return self.init_turn_step()


    def conclude_step(self):
        assert self.turn_idx == self.current_round+1, (f"Turn index is {self.turn_idx} and should be equal to Current Round [{self.current_round}]")       
        self.state="CONCLUDE"
        for player in self.players:

            if player.current_bid == player.round_suits:
                round_reward = player.current_bid + 2 #ten point for every turn won and 20 for guessing right
            else:
                round_reward =  -abs(player.current_bid-player.round_suits) #ten points for every falsly claimed suit
                
            player.game_score += round_reward
            if isinstance(player,TrainPlayer):
               self.r = round_reward
               self.done = True
                
        for player in self.players:
            player.clean_hand() #at this point all hands should be empty anyways
            self.all_bid_completion = torch.zeros(self.n_players)

        return self.turn_obs,self.r,self.done,self.info





if __name__ == "__main__":
    demo_train_player = TrainPlayer()

    adversary_players = [AdversaryPlayer(net.PlayNet(),net.BidNet()) for _ in range(3)]
    env = Game(demo_train_player, adversary_players)
    env.current_round = 8
    obs,r,done,info = env.reset()
    while not done:
        player_action = deck.deck.index(env.train_player.cards_obj[0])
        obs,r,done,info = env.turn_step(player_action)
        print(r,done)
    






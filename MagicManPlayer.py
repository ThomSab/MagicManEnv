import numpy as np
import os
import sys
import json
import MagicManDeck as deck
import torch


#______________________________________________________________________________
#Player Class constructor

class AdversaryPlayer:


    def __init__(self,play_network,bid_network):
        
        self.play_network = play_network
        self.bid_network = bid_network
        
        self.round_score = 0
        self.game_score  = 0
        self.cards_obj = []
        self.cards_tensor = torch.zeros(60) #one-hot encoded deck
       
       
    def play (self,obs):
        action_distribution = self.play_network(obs)
        card_activation = self.cards_tensor*action_distribution #sort out those that the player cant play

        return card_activation
    
    def bid (self,obs):
        activation = self.bid_network(obs)
        
        self.current_activation = activation[0]
            #multiply by the number of card in hand
            #then divide by the amount of players 
            #to have a better starting point for the bots
        self.current_bid = torch.round(self.current_activation)
        #might be an illegal move but ill ignore that for now
        
        return self.current_bid

    def clean_hand(self):
        self.cards = []  

class TrainPlayer:
    
    def __init__(self):
        self.round_score = 0
        self.game_score  = 0
        self.cards_obj = []
        self.cards_tensor = torch.zeros(60) #one-hot encoded deck
        
    def play(self):
        raise UserWarning("Train Player input is external not internal --> DO NOT CALL trainplayer.play()") 
    
    def bid(self):
        raise UserWarning("Train Player input is external not internal --> DO NOT CALL trainplayer.bid()") 
    
    def clean_hand(self):
        self.cards = []  
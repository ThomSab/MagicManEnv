import numpy as np
import os
import sys
import json
import magic_man_deck as deck
import torch


#______________________________________________________________________________
#Player Class constructor

class Player:


    def __init__(self,network):
        
        self.network = network
        
        self.round_score = 0
        self.game_score  = 0
        self.cards = torch.zeros(60) #one-hot encoded deck
       
       
    def play (self):
        try:
            activation = self.play_net.activation()
        except RecursionError:
            raise RecursionError
            
        
        hand_cards = np.array([ [1] if (card in self.cards and card.legal) else [0] for card in deck.deck ])
        card_activation = hand_cards*activation #sort out those that the player cant play
        bestcard_idx = np.where(card_activation == card_activation.max())[0][0]
        played_card = deck.deck[bestcard_idx]
        self.cards.remove(played_card)
        return played_card


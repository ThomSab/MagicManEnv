import stable_baselines.common.env_checker
import MagicMain
import MagicNet as net
import MagicManDeck  as deck
from MagicManPlayer import AdversaryPlayer, TrainPlayer


if __name__ == "__main__":

    demo_train_player = TrainPlayer()

    adversary_players = [AdversaryPlayer(net.PlayNet(),net.BidNet()) for _ in range(3)]
    env = Game(demo_train_player, adversary_players)
    env.current_round = 2
    
    env_checker(env)
    
    
    

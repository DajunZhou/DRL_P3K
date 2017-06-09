from ThreadTrainer import ThreadTrainer
from ThreadAgent import ThreadAgent
from configure import *
from replay_buffer import ReplayBuffer
import rospy

def main():
    rospy.init_node("GazeboDQN_Env", anonymous=True)
    replaybuffer = ReplayBuffer(configure.REPLAY_MEMORY)

    trainer = ThreadTrainer(replaybuffer)
    trainer.start()
    agents = []
    for id in xrange(0, configure.number_of_bots):
        agents.append(ThreadAgent(id, replaybuffer, trainer.trainerbrain))
        agents[-1].start()


    trainer.join()


if __name__ == '__main__':
    main()
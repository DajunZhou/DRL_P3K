from ThreadTrainer import ThreadTrainer
from ThreadAgent import ThreadAgent
from configure import *
from replay_buffer import ReplayBuffer
import rospy
from Queue import Queue
from N_Step_process import N_Step

def main():
    rospy.init_node("GazeboDQN_Env_NAF", anonymous=True)
    replaybuffer = ReplayBuffer(configure.REPLAY_MEMORY)
    Exp_queue = Queue()

    trainer = ThreadTrainer(replaybuffer, Exp_queue)
    trainer.start()

    # n_step_process = N_Step(replaybuffer, Exp_queue)
    # n_step_process.start()

    agents = []
    for id in xrange(0, configure.number_of_bots):
        agents.append(ThreadAgent(id, replaybuffer, trainer.trainerbrain, exp_queue=Exp_queue))
        agents[-1].start()


    trainer.join()


if __name__ == '__main__':
    main()
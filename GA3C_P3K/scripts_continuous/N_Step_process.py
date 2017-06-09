from threading import Thread

class N_Step(Thread):
    def __init__(self, rebuffer, exp_queue):
        super(N_Step, self).__init__()
        self.setDaemon(True)

        self.rebuffer = rebuffer
        self.exp_que = exp_queue

    def run(self):
        if not self.exp_que.empty():
            experiences = self.exp_que.get()

class Uniform(Agent):
    #行動集合から一様サンプリングを行う
    import random
    def policy(self, state):
        return random.choice(self.enviroment.actions)

class KFreqController:
    
    def __init__(self, freq, factor=1, cont=1):
        self.freq = freq
        self.factor = factor
        self.cont = cont
        self.cont_remained = cont
        self.ctr = 0
    
    
    def ok(self):
        return self.freq != -1 and self.cont_remained > 0
    

    def step(self):
        if self.freq == -1:
            return
        self.ctr += 1
        if self.cont_remained > 0:
            self.cont_remained -= 1
        if self.ctr == int(self.freq):
            self.ctr = 0
            self.cont_remained = self.cont
            self.freq *= self.factor

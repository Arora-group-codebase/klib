import torch
import torch.utils.data
import ffcv.loader

import time


class KDataLoader:
    total_load_time = 0
    
    def _enum(self, num):
        raise NotImplementedError()

    def enum(self, num=None):
        last = time.time()
        for ret in self._enum(num):
            KDataLoader.total_load_time += time.time() - last
            yield ret
            last = time.time()

    def __len__(self):
        return len(self.loader)


class KDataLoaderTorch(KDataLoader):
    loader: torch.utils.data.DataLoader
    
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
    
    def _enum(self, num):
        if num is None:
            for idx, pt in enumerate(self.loader):
                yield idx, tuple(p.to(self.device) for p in pt)
        else:
            it = iter(self.loader)
            for idx, pt in enumerate(self.loader):
                if idx >= num:
                    del it
                    break
                yield idx, tuple(p.to(self.device) for p in pt)


class KDataLoaderFFCV(KDataLoader):
    loader: ffcv.loader.Loader
    
    def __init__(self, loader):
        self.loader = loader
    
    def _enum(self, num):
        if num is None:
            for idx, pt in enumerate(self.loader):
                torch.cuda.synchronize()
                yield idx, pt
            torch.cuda.synchronize()
        else:
            it = iter(self.loader)
            for idx, pt in enumerate(it):
                if idx >= num:
                    it.close()
                    break
                torch.cuda.synchronize()
                yield idx, pt
            torch.cuda.synchronize()

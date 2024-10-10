class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, relative=False):
        self.relative = relative
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.relative:
            if not self.count:
                self.scale = 100 / abs(val)
            val *= self.scale
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def tup2list(tuple_list, tuple_idx):
    return list(zip(*tuple_list))[tuple_idx]
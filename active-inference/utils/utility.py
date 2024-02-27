import torch.autograd as autograd
import torch


def Variable(data, *args, **kwargs):
    if torch.cuda.is_available():
        data = data.cuda()
    return autograd.Variable(data, *args, **kwargs)


def unit_prefix(x, n=1):
    for i in range(n):
        x = x.unsqueeze(0)
    return x


def align(x, y, start_dim=0):
    xd, yd = x.dim(), y.dim()
    if xd > yd:
        y = unit_prefix(y, xd - yd)
    elif yd > xd:
        x = unit_prefix(x, yd - xd)

    xs, ys = list(x.size()), list(y.size())
    nd = len(ys)
    for i in range(start_dim, nd):
        td = nd-i-1
        if ys[td] == 1:
            ys[td] = xs[td]
        elif xs[td] == 1:
            xs[td] = ys[td]
    return x.expand(*xs), y.expand(*ys)


def matmul(X, Y):
    results = []
    for i in range(X.size(0)):
        result = torch.mm(X[i], Y[i])
        results.append(result.unsqueeze(0))
    return torch.cat(results)

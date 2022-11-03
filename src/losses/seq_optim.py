import torch
import torch.nn as nn


def _get_mov_avg(x, filt, pad_left, pad_right):
    """
    input x must be of size B x N !
    """
    x = x.permute(1, 0)  # N x B
    mov_avg = filt(x)
    mov_avg = mov_avg.permute(1, 0)  # B is shortened

    if pad_left > 0:
        left = mov_avg[:1].repeat(pad_left, 1)
        mov_avg = torch.cat((left, mov_avg), dim=0)
    if pad_right > 0:
        right = mov_avg[-1:].repeat(pad_right, 1)
        mov_avg = torch.cat((mov_avg, right), dim=0)
    return mov_avg  # B x S


class AvgSmoother(torch.nn.Module):
    def __init__(self, kernel_size=2, data_consistency_w=1.0):
        super(AvgSmoother, self).__init__()
        assert kernel_size > 1

        r = kernel_size % 2
        self.pad_left = kernel_size // 2
        self.pad_right = kernel_size // 2 + r - 1
        self.filter = nn.AvgPool1d(kernel_size=kernel_size, stride=1)
        self.kernel_size = kernel_size
        self.data_consistency_w = data_consistency_w

    def get_mov_avg(self, pred):
        return _get_mov_avg(pred, self.filter, self.pad_left, self.pad_right)

    def forward(self, pred, gt):
        pred = pred.flatten(start_dim=1)  # now it is B x N
        gt = gt.flatten(start_dim=1)  # now it is B x N
        mov_avg = self.get_mov_avg(pred)
        smooth = ((pred - mov_avg) ** 2).mean()

        data_cons = ((pred - gt) ** 2).mean()  # without it, the signal degrades
        return smooth + data_cons * self.data_consistency_w


if __name__ == "__main__":

    kernel_size = 3
    smooth_loss = AvgSmoother(kernel_size)

    batch_size = 5
    num_elem = 10
    x = torch.linspace(1, batch_size * num_elem, batch_size * num_elem)
    x = x.view(batch_size, num_elem)

    y = x + 3 * torch.randn(x.size())

    loss = smooth_loss(y, x)
    print(loss)

    ### smoother class also can extract moving average:
    z = torch.randn(100, 1)
    z_mov_avg = smooth_loss.get_mov_avg(z)

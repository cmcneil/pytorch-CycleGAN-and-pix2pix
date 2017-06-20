import torch


def laplacian_loss1d(input, epsilon=1e-12):
    # print input.size()
    norm_factor = torch.squeeze(torch.clamp(
                    torch.sum(torch.abs(input)), min=epsilon))
    # print norm_factor.size()
    norm = input / norm_factor.expand(input.size())
    return torch.sum(2.0*norm[1:-1] + norm[0:-2] + norm[2:])


def laplacian_loss(input, epsilon=1e-12):
    """
    ASSUMES a (input_dim x out_dim x 1 x 1) tensor. lol.
    """
    # print "called laplacian loss!"
    w = torch.squeeze(input)
    lap = torch.pow(laplacian_loss1d(w[0, :]), 2.0)
    for i in xrange(1, w.size()[0]):
        # print "called laplacian loss!"
        lap = lap + torch.pow(laplacian_loss1d(w[i, :]), 2.0)
    return lap

### TF version:
# def laplacian_loss(self, X, lambda_lap=1., axis=0):
#     X = tf.squeeze(X)
#     lap_feat = []
#     for i in xrange(X.get_shape()[-1].value):
#         lap_feat.append(tf.square(self.laplacian1d(X[:, i])))
#     reg = tf.add_n(lap_feat)
#     return tf.mul(tf.constant(lambda_lap), tf.reduce_mean(reg))

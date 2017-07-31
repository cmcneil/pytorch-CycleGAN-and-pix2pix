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


def orthoreg_loss(input, lambda_coeff=0.005, epsilon=1e-12, gpu_ids=None):
    """
    Orthoreg Loss from https://prlz77.github.io/iclr2017-paper/

    Args:
        input: pytorch Variable. Assumed to be of shape (nfilters, size_input)
        lambda_coeff: float. Controls the maximum angle between two feature vectors that will
            cause them to be regularized. 8.0 => about 90deg.
    """
    # print input
    w_unorm = torch.squeeze(input)
    w = w_unorm / torch.clamp(torch.sum(w_unorm**2, 1)
                              .expand(w_unorm.size()), min=epsilon)
    # print w
    wwt = torch.mm(w, torch.t(w))
    # print wwt
    # print torch.exp(lambda_coeff*(wwt - 1.0))
    all_dist_loss = torch.clamp(torch.log(1 + torch.exp(lambda_coeff*(wwt - 1.0))),
                                max=1e12)
    # print all_dist_loss
    selector = torch.autograd.Variable(1.0 - torch.eye(w.size()[0]))
    # print all_dist_loss
    # print all_dist_loss*selector.cuda()
    if gpu_ids is not None and len(gpu_ids) > 0:
        selector = selector.cuda()
    sum_dist_loss = torch.sum(all_dist_loss*selector)
    return sum_dist_loss / (2.0*w.size()[0])

    # for i in range(w.size()[0]):
    #     for j in range(w.size()[0]):
    #         if i == j:
    #             continue
    #         ang = torch.dot(w[i, :], w[j, :])
            # torch.log(torch.exp((ang - 1.0)*lambda_coeff) + 1.0)



### TF version:
# def laplacian_loss(self, X, lambda_lap=1., axis=0):
#     X = tf.squeeze(X)
#     lap_feat = []
#     for i in xrange(X.get_shape()[-1].value):
#         lap_feat.append(tf.square(self.laplacian1d(X[:, i])))
#     reg = tf.add_n(lap_feat)
#     return tf.mul(tf.constant(lambda_lap), tf.reduce_mean(reg))

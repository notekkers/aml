import torch
import math
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.functional import conv2d
from torch.autograd import Variable
from torch.nn import Sigmoid, Softplus
from torch.nn.modules.utils import _pair
from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal



class GNJConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):

        # Init torch module
        super(GNJConv2d, self).__init__()

        # Init conv params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Init filter latents
        self.weight_mu = Parameter(Tensor(out_channels, in_channels, *self.kernel_size))
        self.weight_logvar = Parameter(Tensor(out_channels, in_channels, *self.kernel_size))


        self.bias = bias
        self.bias_mu = Parameter(Tensor(out_channels)) if self.bias else None
        self.bias_logvar = Parameter(Tensor(out_channels)) if self.bias else None

        # Init prior latents
        self.z_mu = Parameter(Tensor(out_channels))
        self.z_logvar = Parameter(Tensor(out_channels))

        # Set initial parameters
        self._init_params()

        # for brevity to conv2d calls
        self.convargs = [self.stride, self.padding, self.dilation]

        # util activations
        self.sigmoid = Sigmoid()
        self.softplus = Softplus()


    # forward network pass
    def forward(self, x):

        # vanilla forward pass if testing
        if not self.training:
            post_weight_mu = self.weight_mu * self.z_mu[:, None, None, None]
            post_bias_mu = self.bias_mu * self.z_mu if (self.bias_mu is not None) else None
            return conv2d(x, post_weight_mu, post_bias_mu, *self.convargs)

        #batch_size = x.size()[0]

        # unpack mean/std
        mu = self.z_mu
        std = torch.exp(0.5 * self.z_logvar)

        # rsample: sample scale prior with reparam trick
        z = Normal(mu, std).rsample()[None, :, None, None]

        # weights and biases for variance estimation
        weight_v = self.weight_logvar.exp()
        bias_v = self.bias_logvar.exp() if self.bias else None

        # parameterise output distribution
        mu_out = conv2d(x, self.weight_mu, self.bias_mu, *self.convargs) * z
        var_out = conv2d(x**2, weight_v, bias_v, *self.convargs) * (z ** 2)

        # Init out, note multiplicative noise==variational dropout
        dist_out = Normal(mu_out, var_out.sqrt()).rsample()
        #dist_out = self.reparam(mu_out*z, (var_out * z.pow(2)).log())

        return dist_out

    def _init_params(self, weight=None, bias=None):

        n = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        thresh = 1/math.sqrt(n)

        # weights
        self.weight_logvar.data.normal_(-9, 1e-2)

        if weight is not None:
            self.weight_mu.data = weight
        else:
            self.weight_mu.data.uniform_(-thresh, thresh)


        if self.bias:
            # biases
            self.bias_logvar.data.normal_(-9, 1e-2)

            if bias is not None:
                self.bias_mu.data = bias
            else:
                self.bias_mu.data.fill_(0)

        # priors
        self.z_mu.data.normal_(1, 1e-2)
        self.z_logvar.data.normal_(-9, 1e-2)


    # shape,scale family reparameterization trick (rsample does this?)
    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)

        # check for cuda
        #tenv = torch.cuda if cuda else torch

        # draw from normal
        eps = torch.FloatTensor(std.size()).normal_()

        return mu + eps * std

    # KL div for GNJ w. Normal approx posterior
    def kl_divergence(self):

        # for brevity in kl_scale
        sg = self.sigmoid
        sp = self.softplus

        # Approximation parameters. Molchanov et al.
        k1, k2, k3 = 0.63576, 1.87320, 1.48695
        log_alpha = self._log_alpha()
        kl_scale = torch.sum(0.5 * sp(-log_alpha) + k1 - k1 * sg(k2  + k3 * log_alpha))
        kl_weight = self._conditional_kl_div(self.weight_mu, self.weight_logvar)
        kl_bias = self._conditional_kl_div(self.bias_mu, self.bias_logvar) if self.bias else 0

        return kl_scale + kl_weight + kl_bias

    @staticmethod
    def _conditional_kl_div(mu, logvar):
        # (8) Weight/bias divergence KL(q(w|z)||p(w|z))
        kl_div = -0.5 * logvar + 0.5 * (logvar.exp() + mu ** 2 - 1)
        return torch.sum(kl_div)

    # effective dropout rate
    def _log_alpha(self):
        epsilon = 1e-8
        log_a = self.z_logvar  - torch.log(self.z_mu ** 2 + epsilon)
        return log_a


class GHConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):

        # Init torch module
        super(GHConv2d, self).__init__()

        # Init conv params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # init constants according to section 5
        self.t0 = 1e-5

        # Init globals
        self.sa_mu = Parameter(Tensor(1))
        self.sa_logvar = Parameter(Tensor(1))
        self.sb_mu = Parameter(Tensor(1))
        self.sb_logvar = Parameter(Tensor(1))

        # Filter locals
        self.alpha_mu = Parameter(Tensor(out_channels))
        self.alpha_logvar = Parameter(Tensor(out_channels))
        self.beta_mu = Parameter(Tensor(out_channels))
        self.beta_logvar = Parameter(Tensor(out_channels))

        # Weight local
        self.weight_mu = Parameter(Tensor(out_channels, in_channels, *self.kernel_size))
        self.weight_logvar = Parameter(Tensor(out_channels, in_channels, *self.kernel_size))

        # Bias local if required
        self.bias = bias
        self.bias_mu = Parameter(Tensor(out_channels)) if self.bias else None
        self.bias_logvar = Parameter(Tensor(out_channels)) if self.bias else None

        # Set initial parameters
        self._init_params()

        # for brevity to conv2d calls
        self.convargs = [self.stride, self.padding, self.dilation]
    def _s_mu(self):
        return 0.5 * (self.sa_mu + self.sb_mu)

    def _s_var(self):
        return 0.25 * (self.sa_logvar.exp() + self.sb_logvar.exp())

    def _z_var(self):
        return 0.25 * (self.alpha_logvar.exp() + self.beta_logvar.exp())

    def _z_mu(self):
        return 0.5 * (self.alpha_mu + self.beta_mu)

    def forward(self, x):

        # vanilla forward pass if testing
        if not self.training:
            expect_z = torch.exp(0.5 * (self._z_var() + self._s_var()) + self._z_mu() + self._s_mu())
            post_weight_mu = self.weight_mu * expect_z[:, None, None, None]
            post_bias_mu = self.bias_mu * expect_z if (self.bias_mu is not None) else None
            return conv2d(x, post_weight_mu, post_bias_mu, *self.convargs)

        # compute global shrinkage
        s_mu = 0.5 * (self.sa_mu + self.sb_mu)
        s_sig = torch.sqrt(self._s_var())
        s = LogNormal(s_mu, s_sig).rsample()

        # compute filter scales
        z_mu = self._z_mu()
        z_var = self._z_var()
        z = s * LogNormal(z_mu, z_var.sqrt()).rsample()[None, :, None, None]


        # lognormal out params, local reparameterization trick
        bvar = self.bias_logvar.exp() if self.bias else None
        mu_out = conv2d(x, self.weight_mu, self.bias_mu, *self.convargs) * z
        scale_out = conv2d(x**2, self.weight_logvar.exp(), bvar, *self.convargs) * (z ** 2)

        # compute output weight distribution, again reparameterised
        dist_out = Normal(mu_out, scale_out.sqrt()).rsample()

        # return fully reparameterised forward pass
        return dist_out


    def _init_params(self, weight=None, bias=None):

        # initialisation params - note mean of lognormal is exp(mu + 0.5 *var)
        init_mu_logvar, init_mu, init_var = -9, 0., 1e-2

        # compute xavier initialisation on weights
        n = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        thresh = 1/math.sqrt(n)

        if weight is not None:
            self.weight_mu.data = weight
        else:
            self.weight_mu.data.uniform_(-thresh, thresh)

        # init variance according to appendix A
        self.weight_logvar.data.normal_(init_mu_logvar, init_var)

        if self.bias:
            if bias is not None:
                self.bias_mu.data = bias
            else:
                self.bias_mu.data.fill_(0)

            # biases
            self.bias_logvar.data.normal_(init_mu_logvar, init_var)

        # Decomposed prior means => E[z_init] init ~ 1
        self.alpha_mu.data.normal_(init_mu, init_var)
        self.beta_mu.data.normal_(init_mu, init_var)
        self.sa_mu.data.normal_(init_mu, init_var)
        self.sb_mu.data.normal_(init_mu, init_var)

        # Decomposed prior variances
        self.alpha_logvar.data.normal_(init_mu_logvar, init_var)
        self.beta_logvar.data.normal_(init_mu_logvar, init_var)
        self.sa_logvar.data.normal_(init_mu_logvar, init_var)
        self.sb_logvar.data.normal_(init_mu_logvar, init_var)


    # KL div for GNH with lognormal scale, normal weight variational posterior
    def kl_divergence(self):
        # negative kls, eqns (34-37)
        neg_kl_s = self._global_negative_kl()
        neg_kl_ab = self._filter_local_negative_kl()

        # weight/bias local
        kl_w = self._conditional_kl_div(self.weight_mu, self.weight_logvar)

        if self.bias:
            kl_b = self._conditional_kl_div(self.bias_mu, self.bias_logvar)
        else:
            kl_b = 0

        return kl_w + kl_b - (neg_kl_s + neg_kl_ab)


    def _global_negative_kl(self):

        # hyperparams
        t0 = self.t0

        # const added in every kl div
        c = 1 + math.log(2)

        # shape/scale of global scale parameters
        sa_mu, sb_mu = self.sa_mu, self.sb_mu
        sa_var, sb_var = self.sa_logvar.exp(), self.sb_logvar.exp()

        # Eqns (34)(35)
        kl_sa = math.log(t0) - torch.exp(sa_mu + 0.5 * sa_var)/t0 + 0.5 * (sa_mu + self.sa_logvar + c)
        kl_sb = 0.5 * (self.sb_logvar - sb_mu + c ) - torch.exp(0.5 * sb_var - sb_mu)

        return kl_sa + kl_sb


    def _filter_local_negative_kl(self):

        # const added in every kl div
        c = 1 + math.log(2)

        # hyperparams
        t0 = self.t0

        # filter level shape/scale parameters
        alpha_mu, beta_mu = self.alpha_mu, self.beta_mu
        alpha_logvar, beta_logvar = self.alpha_logvar, self.beta_logvar

        # Eqns (36)(37)
        kl_alpha = torch.sum(0.5 * (alpha_mu + alpha_logvar + c) - torch.exp(alpha_mu + 0.5 * alpha_logvar.exp()))
        kl_beta = torch.sum(0.5 * (beta_logvar - beta_mu + c) - torch.exp(0.5 * beta_logvar.exp() - beta_mu))

        return kl_alpha + kl_beta


    @staticmethod
    def _conditional_kl_div(mu, logvar):
        # eqn (8)
        kl_div = -0.5 * logvar + 0.5 * (logvar.exp() + mu ** 2 - 1)
        return torch.sum(kl_div)

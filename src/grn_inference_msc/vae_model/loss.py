import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NegBinLikelihoodLoss(nn.Module):
    """A class for the likelihood loss when the likelihood probability follows a
    negative binomial distribution
    """

    def __init__(self):
        super().__init__()


    def forward(self, x_true, r, p):
        """The function that calculates the loss

        Params:
            x_true (tensor): The true values
            r (tensor): A tensor containing the r values of the negbin distribution
            p (tensor): A tensor containing teh p values of the negbin distribution

        Returns:
            (tensor): The loss
        """
        loss = torch.sum(torch.lgamma(x_true + r) - torch.lgamma(r) + x_true * torch.log(1 - p) + r * torch.log(p) - torch.lgamma(x_true + 1), dim=1)
        return -torch.mean(loss)



class GaussianLikelihoodLoss(nn.Module):
    """
    Class for the likelihood loss when the likelihood distribution is Gaussian.
    Note that the covariance matrix of the distribution is assumed to be the Identity matrix
    so the input to the loss function is only the mean parameter of the distribution
    """
    def __init__(self):
        super().__init__()


    def forward(self, x_true, mu):
        """The function that calculates the loss

        Params:
            x_true (tensor): The true values
            mu (tensor): The mean parameters of the gaussian

        Returns:
            (tensor): The loss
        """
        loglikelihood = torch.sum(-0.5 * (math.log(2 * math.pi) + torch.pow(x_true - mu, 2)), dim=1)
        return -torch.mean(loglikelihood)



class KLDivergenceLoss(nn.Module):
    """A class for the kl divergence loss, assuming that the prior on the latent space is a standard gaussian"""
    def __init__(self):
        super().__init__()


    def forward(self, mu, logvar):
        """The function that calculates the loss

        Params:
            mu (tensor): A tensor containing the mu values of the latent space distribution
            logvar (tensor): A tensor containing the logvar values of the latent space distribution

        Returns:
            (tensor): The loss
        """
        loss = torch.sum(0.5 * (mu**2 + torch.exp(logvar)  - logvar - 1), dim=1)
        return torch.mean(loss)



class CovarianceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def forward(self, logvar):
        loss = -torch.sum(torch.log(torch.exp(logvar)), dim=1)
        return torch.mean(loss)



class L1RegularizerLoss(nn.Module):
    """A class to use for L1 regularizer loss"""
    def __init__(self):
        super().__init__()


    def forward(self, input):
        """The function that calculates the loss

        Params:
            input (tensor): A tensor of weights that are to be regularized

        Returns:
            (tensor): The loss, the loss is just the sum of the absolute weights provided
        """
        return torch.sum(torch.abs(input))



class VAEGaussianLikelihoodLoss(nn.Module):
    """A class that contains the full loss of the VAE with a
    Gaussian log likelihood. That is the reconstruction loss,
    the KL divergence loss and a L1 regularizer on the GRN matrix weights
    """
    def __init__(self, alpha, beta):
        """The constructor for the VAEGaussianLikelihoodLoss

        Params:
            beta (float): The coefficient of the KLLoss term
            alpha (float): The coefficient of the L1 regularizer term
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.kl_div_loss_fn = KLDivergenceLoss()
        self.ll_loss_fn = GaussianLikelihoodLoss()
        self.l1_reg_loss_fn = L1RegularizerLoss()



    def forward(self, x_true, mu, mu_latent, logvar_latent, A):
        """The function that calculates the loss

        Params:
            x_true (tensor): The true values
            mu (tensor): The mean parameters of the gaussian
            mu_latent (tensor): A tensor containing the mu values of the latent space distribution
            logvar_latent (tensor): A tensor containing the logvar values of the latent space distribution
            A (tensor): A tensor of weights of the GRN layer

        Returns:
            (tensor): The loss
        """
        return self.ll_loss_fn(x_true, mu) + self.beta * self.kl_div_loss_fn(mu_latent, logvar_latent) + self.alpha * self.l1_reg_loss_fn(A)



class BetaVAELoss(nn.Module):
    """A class that contains the full loss for a beta VAE without the GRN layer
    Note that this loss can be used on the model that does not use the GRN layer
    """
    def __init__(self, beta):
        """Constructor for the BetaVAELoss

        Params:
            beta (float): The coefficient to be used in front of the KL loss term
        """
        super().__init__()
        self.beta = beta
        self.recon_loss_fn = GaussianLikelihoodLoss()
        self.kl_loss_fn = KLDivergenceLoss()


    def forward(self, x_true, mu, mu_latent, logvar_latent, split=False):
        if split:
            return self.recon_loss_fn(x_true, mu), self.beta * self.kl_loss_fn(mu_latent, logvar_latent)
        return self.recon_loss_fn(x_true, mu) + self.beta * self.kl_loss_fn(mu_latent, logvar_latent)



class VAENegbinLikelihoodLoss(nn.Module):
    """A class that contains the full loss of the VAE with a
    Negative binomial likelihood. That is the reconstruction loss,
    the KL divergence loss and a L1 regularizer on the GRN matrix weights
    """
    def __init__(self, alpha, beta):
        """The constructor for VAENegbinLikelihoodLoss

        Params:
            beta (float): The coefficient of the KLLoss term
            alpha (float): The coefficient of the L1 regularizer term
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.kl_div_loss_fn = KLDivergenceLoss()
        self.ll_loss_fn = NegBinLikelihoodLoss()
        self.l1_reg_loss_fn = L1RegularizerLoss()


    def forward(self, x_true, r, p, mu_latent, logvar_latent, A):
        """The function that calculates the loss

        Params:
            x_true (tensor): The true values
            r (tensor): The r parameters of the negbin distribution
            p (tensor): The p parameters of the negbin distribution
            mu_latent (tensor): The mu parameters of the latent distribution
            logvar_latent (tensor): The logvar parameters of the latent distribution
            A (tensor): A tensor of weights of the GRN layer
        """
        return self.ll_loss_fn(x_true, r, p) + self.beta * self.kl_div_loss_fn(mu_latent, logvar_latent) + self.alpha * self.l1_reg_loss_fn(A)


class RankLoss(nn.Module):
    """A class that contains the RankLoss which is used as
    a substitution for the AUROC. This loss is differentiable and
    for every pair of negative and positive edges it computes the

    """
    def __init__(self):
        super().__init__()


    def forward(self, A_est, A_true: torch.tensor):
        positive_weights = A_est[A_true > 0]
        negative_weights = A_est[A_true == 0]

        loss = torch.log(F.sigmoid(positive_weights[None, :] - negative_weights[:, None]))

        return -torch.mean(loss)


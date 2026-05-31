import torch
import torch.nn as nn
from grn_inference_msc.vae_model.constants import ProbabilityDistributions
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    dim_in: int
    dim_latent: int
    n_genes: int
    grn_layer_type: str = 'before'
    likelihood_distribution: ProbabilityDistributions = ProbabilityDistributions.NEGBIN
    encoder_hidden_dimensions: list = field(default_factory = lambda: [16, 16])
    decoder_hidden_dimensions: list = field(default_factory = lambda: [16, 16])
    device: torch.device = torch.device('cuda')

    def __post_init__(self):
        assert self.likelihood_distribution in ProbabilityDistributions, 'likelihood distribution is invalid'
        assert self.grn_layer_type in ['none', 'before', 'after', 'both'], 'grn_layer_type is invalid'



class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dim_in = config.dim_in
        self.dim_latent = config.dim_latent
        self.likelihood_distribution = config.likelihood_distribution
        self.device = config.device
        self.n_genes = config.n_genes
        self.grn_layer_type = config.grn_layer_type
        self.A = nn.Parameter(torch.zeros(self.n_genes, self.n_genes)) # The parameters of the GRN layer

        self.encoder, last_layer_dim = self._initialize_linear_mlp(config.encoder_hidden_dimensions, self.dim_in)
        self.encoder_head_mu = nn.Linear(last_layer_dim, self.dim_latent)
        self.encoder_head_logvar = nn.Linear(last_layer_dim, self.dim_latent)

        self.decoder, last_layer_dim = self._initialize_linear_mlp(config.decoder_hidden_dimensions, self.dim_latent)

        if self.likelihood_distribution == ProbabilityDistributions.NEGBIN:
            self.decoder_head_r = nn.Linear(last_layer_dim, self.dim_in)
            self.decoder_head_p = nn.Linear(last_layer_dim, self.dim_in)

            self.decoder_head_r_activation = nn.Softplus()
            self.decoder_head_p_activation = nn.Sigmoid()

        elif self.likelihood_distribution == ProbabilityDistributions.GAUSSIAN:
            self.decoder_head_mu = nn.Linear(last_layer_dim, self.dim_in)
            self.decoder_head_mu_activation = nn.Identity()

        else:
            raise Exception('Invalid value for the likelihood distribution')

        self.to(self.device)


    def _initialize_linear_mlp(self, layers, starting_dimension):
        mlp_layers = []
        current_input_dimension = starting_dimension
        for out_dim in layers:
            mlp_layers.append(nn.Linear(current_input_dimension, out_dim))
            mlp_layers.append(nn.ReLU())
            current_input_dimension = out_dim
        return nn.Sequential(*mlp_layers), current_input_dimension


    def get_A(self):
        return self.A


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.normal(0, 1, size=mu.shape).to(self.device)

        return mu + std * noise


    def forward(self, x):
        if not self.grn_layer_type == 'none':
            assert x.shape[0] == self.n_genes, 'Note that the batch size needs to be the same size as the number of genes in the system'

        x_out_encoder = self.encoder(x)
        mu_latent = self.encoder_head_mu(x_out_encoder)
        logvar_latent = self.encoder_head_logvar(x_out_encoder)

        if self.grn_layer_type == 'before' or self.grn_layer_type == 'both':
            mu_latent = torch.matmul(torch.eye(self.n_genes, device=self.device) - self.A, mu_latent) # Mix the genes in the GRN layer

        z = self.reparameterize(mu_latent, logvar_latent)

        if self.grn_layer_type == 'after' or self.grn_layer_type == 'both':
            z = torch.matmul(torch.linalg.inv(torch.eye(self.n_genes, device=self.device) - self.A), z) # Mix the genes

        x_out_decoder = self.decoder(z)

        if self.likelihood_distribution == ProbabilityDistributions.NEGBIN:
            r = self.decoder_head_r_activation(self.decoder_head_r(x_out_decoder)).clamp(min=1e-4) # Clamp for numerical stability
            p = self.decoder_head_p_activation(self.decoder_head_p(x_out_decoder)).clamp(1e-6, 1 - 1e-6)
            return r, p, mu_latent, logvar_latent

        if self.likelihood_distribution == ProbabilityDistributions.GAUSSIAN:
            mu = self.decoder_head_mu_activation(self.decoder_head_mu(x_out_decoder))
            return mu, mu_latent, logvar_latent

        return None



@dataclass
class ConditionalModelConfig:
    dim_in: int
    dim_latent: int
    dim_condition: int
    n_genes: int
    grn_layer_type: str = 'before'
    likelihood_distribution: ProbabilityDistributions = ProbabilityDistributions.NEGBIN
    encoder_hidden_dimensions: list = field(default_factory = lambda: [16, 16])
    decoder_hidden_dimensions: list = field(default_factory = lambda: [16, 16])
    device: torch.device = torch.device('cuda')


    def __post_init__(self):
        assert self.likelihood_distribution in ProbabilityDistributions, 'likelihood distribution is invalid'
        assert self.grn_layer_type in ['none', 'before', 'after', 'both'], 'grn_layer_type is invalid'



class FiLMLayer(nn.Module):
    """A conditioning layer that uses Feature wIse Linear Modulation layers to
    modulate the activations of a layer.
    """
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.layer1 = nn.Linear(dim_in, dim_hidden)
        self.layer2_gamma = nn.Linear(dim_hidden, dim_out)
        self.layer2_beta = nn.Linear(dim_hidden, dim_out)
        self.activation = nn.ReLU()


    def forward(self, A, C):
        """The forward function of the FiLM conditioning layer

        Params:
            a (tensor): The activations to be FiLM-ed (transformed) (shape N x K), N is number of samples
            c (tensor): The external information to condition on
        Returns:
            (tensor): The output of the FiLM-ed layer
        """
        assert C.shape[0] == A.shape[0], 'C and A matrices must be have the same number of samples'
        assert A.shape[1] == self.dim_out, 'Features of A must match dim_out'

        h = self.layer1(C)
        h = self.activation(h)
        gamma = self.layer2_gamma(h)
        beta = self.layer2_beta(h)

        return gamma * A + beta



class ConditionalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim_in = config.dim_in
        self.dim_latent = config.dim_latent
        self.dim_condition = config.dim_condition
        self.likelihood_distribution = config.likelihood_distribution
        self.device = config.device
        self.n_genes = config.n_genes
        self.grn_layer_type = config.grn_layer_type
        self.A = nn.Parameter(torch.zeros(self.n_genes, self.n_genes)) # The parameters of the GRN layer

        self.encoder_first_film = FiLMLayer(self.dim_condition, 8, self.dim_in)
        self.encoder, self.encoder_FiLM, last_layer_dim = self._initialize_linear_mlp(config.encoder_hidden_dimensions, self.dim_in)
        self.encoder_head_mu = nn.Linear(last_layer_dim, self.dim_latent)
        self.encoder_head_logvar = nn.Linear(last_layer_dim, self.dim_latent)

        self.decoder_first_FiLM = FiLMLayer(self.dim_condition, 8, self.dim_latent)
        self.decoder, self.decoder_FiLM, last_layer_dim = self._initialize_linear_mlp(config.decoder_hidden_dimensions, self.dim_latent)

        if self.likelihood_distribution == ProbabilityDistributions.NEGBIN:
            self.decoder_head_r = nn.Linear(last_layer_dim, self.dim_in)
            self.decoder_head_p = nn.Linear(last_layer_dim, self.dim_in)

            self.decoder_head_r_activation = nn.Softplus()
            self.decoder_head_p_activation = nn.Sigmoid()

        elif self.likelihood_distribution == ProbabilityDistributions.GAUSSIAN:
            self.decoder_head_mu = nn.Linear(last_layer_dim, self.dim_in)
            self.decoder_head_mu_activation = nn.Identity()

        else:
            raise Exception('Invalid value for the likelihood distribution')

        self.to(self.device)


    def _initialize_linear_mlp(self, layers, starting_dimension):
        mlp_layers = []
        FiLM_layers = []
        current_input_dimension = starting_dimension
        for out_dim in layers:
            mlp_layers.append(nn.Linear(current_input_dimension, out_dim))
            mlp_layers.append(nn.ReLU())
            FiLM_layers.append(FiLMLayer(self.dim_condition, 8, out_dim))
            current_input_dimension = out_dim
        return nn.ModuleList(mlp_layers), nn.ModuleList(FiLM_layers), current_input_dimension


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.normal(0, 1, size=mu.shape).to(self.device)

        return mu + std * noise


    def get_A(self):
        return self.A


    def forward(self, x, c):
        if not self.grn_layer_type == 'none':
            assert x.shape[0] == self.n_genes, 'Note that the batch size needs to be the same size as the number of genes in the system'

        # Modulate the data right in the beginning using the contextual information
        x = self.encoder_first_film(x, c)

        # Pass through input through the encoder
        for i in range(len(self.encoder_FiLM)):
            x = self.encoder[2*i](x) # Linear layer
            x = self.encoder[2*i + 1](x) # Activation
            x = self.encoder_FiLM[i](x, c)

        mu_latent = self.encoder_head_mu(x)
        logvar_latent = self.encoder_head_logvar(x)

        if self.grn_layer_type == 'before' or self.grn_layer_type == 'both':
            mu_latent = torch.matmul(torch.eye(self.n_genes, device=self.device) - self.A, mu_latent) # Mix the genes in the GRN layer

        z = self.reparameterize(mu_latent, logvar_latent)

        if self.grn_layer_type == 'after' or self.grn_layer_type == 'both':
            z = torch.matmul(torch.linalg.inv(torch.eye(self.n_genes, device=self.device) - self.A), z) # Mix the genes

        z = self.decoder_first_FiLM(z, c)

        for i in range(len(self.decoder_FiLM)):
            z = self.decoder[2*i](z) # Linear
            z = self.decoder[2*i + 1](z) # Activation
            z = self.decoder_FiLM[i](z, c) # Condition

        if self.likelihood_distribution == ProbabilityDistributions.NEGBIN:
            r = self.decoder_head_r_activation(self.decoder_head_r(z)).clamp(min=1e-4) # Clamp for numerical stability
            p = self.decoder_head_p_activation(self.decoder_head_p(z)).clamp(1e-6, 1 - 1e-6)
            return r, p, mu_latent, logvar_latent

        if self.likelihood_distribution == ProbabilityDistributions.GAUSSIAN:
            mu = self.decoder_head_mu_activation(self.decoder_head_mu(z))
            return mu, mu_latent, logvar_latent

        return None


    def generate(self, c):
        z = torch.randn((1, self.dim_latent)).to(self.device)

        z = self.decoder_first_FiLM(z, c)

        for i in range(len(self.decoder_FiLM)):
            z = self.decoder[2*i](z) # Linear
            z = self.decoder[2*i + 1](z) # Activation
            z = self.decoder_FiLM[i](z, c) # Condition

        if self.likelihood_distribution == ProbabilityDistributions.NEGBIN:
            r = self.decoder_head_r_activation(self.decoder_head_r(z)).clamp(min=1e-4) # Clamp for numerical stability
            p = self.decoder_head_p_activation(self.decoder_head_p(z)).clamp(1e-6, 1 - 1e-6)
            return r, p

        if self.likelihood_distribution == ProbabilityDistributions.GAUSSIAN:
            mu = self.decoder_head_mu_activation(self.decoder_head_mu(z))
            return mu
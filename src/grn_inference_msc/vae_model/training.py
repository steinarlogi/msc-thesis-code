import torch
from tqdm import trange
from grn_inference_msc.vae_model.vae_model import Model, ModelConfig

def get_trained_model(
        dataset,
        dim_in,
        dim_hidden_encoder,
        dim_hidden_decoder,
        dim_latent,
        n_genes,
        lr,
        n_epochs,
        loss_fn,
        likelihood_distribution,
        grn_layer_type,

):
    """
    A function to get trained models. The function first
    Checks if a trained model with the provided parameters already exists
    and loads that one instead of training again if it exists
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_config = ModelConfig(
        dim_in=dim_in,
        dim_latent=dim_latent,
        encoder_hidden_dimensions=dim_hidden_encoder,
        decoder_hidden_dimensions=dim_hidden_decoder,
        n_genes=n_genes,
        grn_layer_type=grn_layer_type,
        likelihood_distribution=likelihood_distribution,
        device=device
    )

    model = Model(
        config=model_config
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = train_model(model, dataset, loss_fn, optimizer, n_epochs, device)

    return losses, model



def train_model(
    model,
    dataset,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    n_epochs=100,
    device = torch.device('cuda')
):
    """This is a helper function to train a  model

    Params:
        model (vae_model.Model): The model to train
        dataset (tensor): The gene expression data used for the training shape: N x K where N is the number of genes and K is the number of experiments
        loss_fn: The loss function. Should take a variable amount of tensors as arguments and A and X as keyword arguments and return a tensor,
        optimizer (torch.optim.Optimizer): The optimizer to use for the training
        n_epochs (int): The number of epochs to train the model for

    Returns:
        (list): A list of the epoch losses collected during training
    """

    EPOCHS = trange(n_epochs, desc='Training model ')
    losses = []

    for e in EPOCHS:
        optimizer.zero_grad()

        model_output = model(dataset.to(device))

        loss = loss_fn(*model_output, A=model.get_A(), X=dataset.to(device))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        EPOCHS.set_postfix(loss=loss.item())

    return losses

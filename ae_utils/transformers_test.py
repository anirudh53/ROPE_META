


class VAE(nn.Module):
    """Variational Autoencoder that builds itself based on hyperparameters from a config file"""

    def __init__(self, config: dict, device: torch.device):
        """initializes VAE object

        :param config: config for the VAE hyperparameters
        :type config: dict
        :param device: device you are running on (e.g. torch.device("cuda:0"))
        :type device: torch.device
        """
        super(VAE, self).__init__()
        # save necessary attributes from function inputs
        self._device = device
        self._bottleneck_size = config.get("bottleneck_size", 10)
        self._alpha = config.get("alpha_loss", 0.0)
        self._beta = config.get("beta_loss", 1.0)
        self._lambda = config.get("lambda_loss", 1.0)
        self._build_VAE = config.get("build_VAE", False)
        self._use_MMD_loss = config.get("use_MMD_loss", True)
        self._orthogonal = config.get("orthogonal", True)
        print(f"build_VAE: {self._build_VAE}")
        print(f"use_MMD_loss: {self._use_MMD_loss}")
        print(f"orthogonal: {self._orthogonal}")
        self._check_conflicts()
        # build encoder
        self.encoder = Encoder(
            bottleneck_size=self._bottleneck_size,
            build_VAE=self._build_VAE,
            num_encoder_downsamples=config.get("num_encoder_downsamples", 4),
            num_layers_E_0=config.get("num_layers_E_0", 1),
            num_layers_E_1=config.get("num_layers_E_1", 1),
            num_layers_E_2=config.get("num_layers_E_2", 1),
            num_layers_E_3=config.get("num_layers_E_3", 1),
            num_filters_E_0=config.get("num_filters_E_0", 16),
            num_filters_E_1=config.get("num_filters_E_1", 32),
            num_filters_E_2=config.get("num_filters_E_2", 64),
            num_filters_E_3=config.get("num_filters_E_3", 128),
            activation_E_0=config.get("activation_E_0", "relu"),
            activation_E_1=config.get("activation_E_1", "relu"),
            activation_E_2=config.get("activation_E_2", "relu"),
            activation_E_3=config.get("activation_E_3", "relu"),
        )
        # build decoder
        self.decoder = Decoder(
            bottleneck_size=self._bottleneck_size,
            num_decoder_upsamples=config.get("num_decoder_upsamples", 4),
            num_layers_D_0=config.get("num_layers_D_0", 1),
            num_layers_D_1=config.get("num_layers_D_1", 1),
            num_layers_D_2=config.get("num_layers_D_2", 1),
            num_layers_D_3=config.get("num_layers_D_3", 1),
            num_filters_D_0=config.get("num_filters_D_0", 128),
            num_filters_D_1=config.get("num_filters_D_1", 64),
            num_filters_D_2=config.get("num_filters_D_2", 32),
            num_filters_D_3=config.get("num_filters_D_3", 16),
            activation_D_0=config.get("activation_D_0", "relu"),
            activation_D_1=config.get("activation_D_1", "relu"),
            activation_D_2=config.get("activation_D_2", "relu"),
            activation_D_3=config.get("activation_D_3", "relu"),
        )

    def _load_config(self, filename: str) -> Dict[str, Union[str, int, float]]:
        """loads config file and returns dictionary

        :param filename: path to config file
        :type filename: str
        :return: configuration dictionary
        :rtype: Dict[str, Union[str, int, float]]
        """
        with open(filename, "r") as f:
            config = yaml.safe_load(f)
        return config.get("model")

    def _check_conflicts(self) -> None:
        """checks for conflicts in parameters `build_VAE`, `use_MMD_loss`; when `build_VAE` is False,
        there is no latent distribution and the MMD loss cannot be computed

        :raises ValueError: if there is a conflict between configurable parameters
        """
        if (not self._build_VAE) and self._use_MMD_loss:
            raise ValueError(
                f"Conflict with configurable parameters. You cannot use MMD loss if you are not building a VAE."
            )

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """uses the encoder to predict the latent distribution

        :param x: model / encoder input
        :type x: torch.Tensor
        :return: mean and log-variance of the latent space
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        return self.encoder(x)

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """sample the latent distribution

        :param mu: latent space mean
        :type mu: torch.Tensor
        :param logvar: log-variance of the latent variables
        :type logvar: torch.Tensor
        :return: sampled latent representation
        :rtype: torch.Tensor
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        """uses the decoder to predict the full-state from a reparameterized latent distribution

        :param x: decoder input
        :type x: torch.Tensor
        :return: prediction of the full-space
        :rtype: torch.Tensor
        """
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """runs an input through the model

        :param x: input
        :type x: torch.Tensor
        :return: prediction, latent mean, and latent log-variance
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        """
        z = self._encode(x)
        if self._build_VAE:
            mu, logvar = z
            z = self._reparameterize(mu, logvar)
            return self._decode(z), mu, logvar
        return self._decode(z), z

    def _compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """positive-definite kernel function for computing MMD loss

        :param x: first variable
        :type x: torch.Tensor
        :param y: second variable
        :type y: torch.Tensor
        :return: output of the kernel function
        :rtype: torch.Tensor
        """
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (
            (tiled_x - tiled_y).pow(2).mean(2)
        )  # / float(dim) # this is what the code example
        return torch.exp(-kernel_input)  # (x_size, y_size)

    def _compute_mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.FloatTensor:
        """computes the maximum mean discrepency between the 'true samples` and the reparameterized
        latent representation

        :param x: the 'true' or 'real' samples
        :type x: torch.Tensor
        :param y: generated latent code
        :type y: torch.Tensor
        :return: MMD loss
        :rtype: torch.FloatTensor
        """
        xx_kernel = self._compute_kernel(x, x)
        yy_kernel = self._compute_kernel(y, y)
        xy_kernel = self._compute_kernel(x, y)
        return torch.mean(xx_kernel) + torch.mean(yy_kernel) - 2 * torch.mean(xy_kernel)

    def loss(
        self,
        x: torch.Tensor,
        xhat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor = None,
    ) -> Tuple[torch.FloatTensor]:
        """loss function for InfoVAE; leverages mean square error (MSE), Kullback-Leibler divergence
        (KLD), and maximum mean discrepency (MMD)

        :param x: model input / desired output
        :type x: torch.Tensor
        :param xhat: model prediction
        :type xhat: torch.Tensor
        :param mu: mean of latent space
        :type mu: torch.Tensor
        :param logvar: log-variance of latent space, defaults to None
        :type logvar: torch.Tensor, optional
        :return: overall loss, MSE loss, KLD loss, MMD loss, and orthogonal loss for the batch
        :rtype: Tuple[torch.FloatTensor]
        """
        # compute basic MSE loss
        mse_loss = ((x - xhat).pow(2)).mean()
        # compute KLD loss from latent distribution
        kld_loss = torch.Tensor([0.0]).to(self._device)
        if self._build_VAE:
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # compute MMD loss
        mmd_loss = torch.Tensor([0.0]).to(self._device)
        if self._use_MMD_loss:
            true_samples = torch.autograd.Variable(
                torch.randn(200, self._bottleneck_size), requires_grad=False
            ).to(self._device)
            z = self._reparameterize(mu, logvar)
            mmd_loss = self._compute_mmd(true_samples, z) * x.size(0) * self._bottleneck_size
        # compute orthogonal loss
        orthogonal_loss = torch.Tensor([0.0]).to(self._device)
        if self._orthogonal:
            orthogonal_loss = (
                (
                    torch.matmul(torch.transpose(mu, 0, 1), mu)
                    - torch.eye(self._bottleneck_size).to(self._device)
                ).pow(2)
            ).mean()
        return (
            mse_loss
            + (1 - self._alpha) * kld_loss
            + (self._lambda + self._alpha - 1) * mmd_loss
            + self._beta * orthogonal_loss,
            mse_loss,
            kld_loss,
            mmd_loss,
            orthogonal_loss,
        )

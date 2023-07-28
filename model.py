import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv



class GDN(nn.Module):
    def __init__(
        self,
        num_steps: int,
        min_beta: float,
        max_beta: float,
        num_nodes: int,
        feature_size: int,
    ):
        super(GDN, self).__init__()
        self.num_steps = num_steps
        self.betas = torch.linspace(min_beta, max_beta, num_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))])
        
        self._GCN = GCNConv(feature_size, feature_size)  # By default this adds self loops and normalises
        
        self._PIUNet = PIUNet(num_nodes, num_steps)
        
        self._DDPM = DDPM(self._PIUNet, num_steps, self.betas, self.alphas, self.alpha_bars)
        
        self._DDRM = DDRM(self._DDPM, num_nodes, feature_size)
        
        self._linear = nn.Linear(feature_size, 1)
        
        
    def forward(
        self,
        x: torch.Tensor,  # [200, 16]
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: torch.Tensor = None,
        return_sample: bool = False,
    ) -> torch.Tensor:

        h = self._GCN(x,  edge_index)  # [200, 16]
        
        if return_sample==False:        
            h = self._linear(h)  # [200, 1]
            h = torch.squeeze(h)  # [200]
        
        return h



class PIUNet(nn.Module):
    def __init__(
        self,
        h: int,  # Height of input matrix, for us h=200
        num_steps: int,
        time_emb_dim: int = 100
    ):
        super(PIUNet, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(num_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(num_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # Downsampling regime
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            PICNN((1, h, 16), 1, 10),
            PICNN((10, h, 16), 10, 10),
            PICNN((10, h, 16), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, (1,4), (1,2), (0,1))

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            PICNN((10, h, 8), 10, 20),
            PICNN((20, h, 8), 20, 20),
            PICNN((20, h, 8), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, (1,4), (1,2), (0,1))

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            PICNN((20, h, 4), 20, 40),
            PICNN((40, h, 4), 40, 40),
            PICNN((40, h, 4), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, (1,2), 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, (1,3), 1, (0,1))
        )

        # Bottleneck point
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            PICNN((40, h, 3), 40, 20),
            PICNN((20, h, 3), 20, 20),
            PICNN((20, h, 3), 20, 40)
        )

        # Up sampling regime
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, (1,3), 1, (0,1)),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, (1,2), 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            PICNN((80, h, 4), 80, 40),
            PICNN((40, h, 4), 40, 20),
            PICNN((20, h, 4), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, (1,4), (1,2), (0,1))
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            PICNN((40, h, 8), 40, 20),
            PICNN((20, h, 8), 20, 10),
            PICNN((10, h, 8), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, (1,4), (1,2), (0,1))
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            PICNN((20, h, 16), 20, 10),
            PICNN((10, h, 16), 10, 10),
            PICNN((10, h, 16), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, (1,3), 1, (0,1))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
    
        # x is dim [N, 1, 200, 16], where N = batch size
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # [N, 10, 200, 16]
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # [N, 20, 200, 8]
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # [N, 40, 200, 4]
        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # [N, 40, 200, 3]
        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # [N, 80, 200, 4]        
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # [N, 20, 200, 4]
        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # [N, 40, 200, 8]
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # [N, 10, 200, 8]
        out = torch.cat((out1, self.up3(out5)), dim=1)  # [N, 20, 200, 16]        
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # [N, 10, 200, 16]
        out = self.conv_out(out)  # [N, 1, 200, 16]

        return out
    
    # The single hidden layer MLP which is used to map positional embeddings
    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.SiLU(),
        nn.Linear(dim_out, dim_out)
      )

# The "PICNN" called on is the below 1-D CNN class:
class PICNN(nn.Module):
    def __init__(
        self,
        shape: tuple,
        in_c: int,
        out_c: int,
        kernel_size: tuple = (1,3),
        stride: int = 1,
        padding: tuple = (0,1),
        normalize: bool = True,
    ):
        super(PICNN, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU()
        self.normalize = normalize

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
    
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        
        return out

# The "sinusoidal_embedding" called on is the below function, this
# returns the standard positional embedding
def sinusoidal_embedding(n, d):
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding



class DDPM(nn.Module):
    def __init__(
        self,
        network,
        num_steps,
        betas,
        alphas,
        alpha_bars,
    ):
        super(DDPM, self).__init__()
        self.num_steps = num_steps
        self.network = network  # Will pass in PIU-Net as network
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def forward(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        epsilon: torch.Tensor = None,
    ) -> torch.Tensor:
    
        n, c, h, w = x0.shape  # n = batch size, c = num channels, h = height, w = width
        a_bar = self.alpha_bars[t]

        if epsilon is None:
            epsilon = torch.randn(n, c, h, w)
        
        noisy = a_bar.sqrt().reshape(n, 1, 1, 1)*x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1)*epsilon
        return noisy

    def backward(self, x, t):
        # Runs each image in the batch through the network, each for their
        # random time step t, the network returns its estimation of
        # the noise that was added in the last step
        return self.network(x, t)



class DDRM(nn.Module):
    def __init__(
        self,
        ddpm,
        h: int,
        w: int,
    ):
        super(DDRM, self).__init__()
        self.ddpm = ddpm
        self.h = h
        self.w = w
        
    def forward(
        self,
        y: torch.Tensor,
        sigma_y: float,
    ) -> torch.Tensor:
        
        ddpm = self.ddpm
        h = self.h
        w = self.w
        
        for idx, t in enumerate(list(range(ddpm.num_steps))[::-1]):
            # Using \sigma_t^2 = \beta_t, Ho et al. found the model to be robust to the \sigma_t choice
            beta_t = ddpm.betas[t]
            sigma_t = beta_t.sqrt()

            # To satisfy Theorem 1
            eta = 1
            eta_b = (2*sigma_t**2)/(sigma_t**2 + sigma_y**2)
            
            z_t = torch.randn(h, w) 

            if idx==0:
                assert sigma_y<=sigma_t, f'Error: sigma_y = {sigma_y} > {sigma_t} = sigma_T, decrease sigma_y value.'
                x_t = y + (sigma_t**2 - sigma_y**2).sqrt() * z_t


            else:           
                time_tensor = (torch.ones(1, 1) * (t+1)).long()
                tilde_epsilon_t = ddpm.backward(x_t[None, None, :], time_tensor)[0][0]  # here x_t corresponds to t+1

                alpha_t = ddpm.alphas[t]
                alpha_t_bar = ddpm.alpha_bars[t]
                
                tilde_x_t = (1/alpha_t.sqrt()) * (x_t - (beta_t/((1 - alpha_t_bar).sqrt())) * tilde_epsilon_t)
                
                if sigma_t<sigma_y:
                    x_t = tilde_x_t + sigma_t * z_t
                else:
                    x_t = (1-eta_b)*tilde_x_t + eta_b*y + (sigma_t**2 - (sigma_y**2)*(eta_b**2)).sqrt() * z_t

        return x_t

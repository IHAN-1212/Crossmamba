import torch
import torch.nn as nn
from einops import repeat,rearrange

from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm

class Crossmamba(nn.Module):
    def __init__(self, data_dim, in_len, out_len, t_cycle, d_model=512, d_ff=1024, d_state=8, dropout=0.0, device=torch.device('cuda:0')):
        super(Crossmamba, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.t_cycle = t_cycle
        self.device = device

        # Embedding
        self.enc_value_embedding = TCE_Embedding(t_cycle, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.in_len // t_cycle), d_model))
        self.pre_norm = RMSNorm(d_model)

        # Encoder
        self.encoder = Encoder(d_model, d_state, d_ff, dropout)

        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.out_len // t_cycle), d_model))
        self.decoder = Decoder(t_cycle, d_model, d_state, d_ff, dropout)

    def forward(self, x_seq):
        batch_size = x_seq.shape[0]

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)

        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=batch_size)
        predict_y = self.decoder(dec_in, enc_out)

        return predict_y[:, :self.out_len, :]

class TCE_Embedding(nn.Module):
    def __init__(self, t_cycle, d_model):
        super(TCE_Embedding, self).__init__()
        self.t_cycle = t_cycle

        self.linear = nn.Linear(t_cycle, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(x, 'b (cycle_num t_cycle) d -> (b d cycle_num) t_cycle', t_cycle=self.t_cycle)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d cycle_num) d_model -> b d cycle_num d_model', b=batch, d=ts_dim)

        return x_embed

class Encoder(nn.Module):
    def __init__(self, d_model, d_state, d_ff, dropout):
        super(Encoder, self).__init__()
        self.noise_std_low = 0.2
        self.noise_std_medium = 0.01
        self.noise_std_high = 0.001
        self.encode_block_1 = FDSE(d_model, d_state, d_ff, dropout)
        self.encode_block_2 = FDSE(d_model, d_state, d_ff, dropout)
        self.encode_block_3 = FDSE(d_model, d_state, d_ff, dropout)
        self.encode_block_4 = FDSE(d_model, d_state, d_ff, dropout)

    def forward(self, x):
        encode_x = []
        encode_x.append(x)

        x = self.encode_block_1(x)
        encode_x.append(x)

        x2 = self.encode_block_2(self.add_low_frequency_noise(x))
        encode_x.append(x2)

        x3 = self.encode_block_3(self.add_medium_frequency_noise(x))
        encode_x.append(x3)

        x4 = self.encode_block_4(self.add_high_frequency_noise(x))
        encode_x.append(x4)

        return encode_x

    def add_low_frequency_noise(self, x):
        sqrt_alpha_t, sqrt_one_minus_alpha_t = self.noise_alphas(1000, self.noise_std_low)
        return sqrt_alpha_t * x + sqrt_one_minus_alpha_t * torch.randn_like(x)

    def add_medium_frequency_noise(self, x):
        sqrt_alpha_t, sqrt_one_minus_alpha_t = self.noise_alphas(1000, self.noise_std_medium)
        return sqrt_alpha_t * x + sqrt_one_minus_alpha_t * torch.randn_like(x)

    def add_high_frequency_noise(self, x):
        sqrt_alpha_t, sqrt_one_minus_alpha_t = self.noise_alphas(1000, self.noise_std_high)
        return sqrt_alpha_t * x + sqrt_one_minus_alpha_t * torch.randn_like(x)

    def noise_alphas(self, timesteps, std):
        betas = torch.linspace(0, std, timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        return sqrt_alphas_cumprod[timesteps - 1], sqrt_one_minus_alphas_cumprod[timesteps - 1]

class DecoderLayer(nn.Module):
    def __init__(self, t_cycle, d_model, d_state, d_ff=None, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_fdse = FDSE(d_model, d_state, d_ff, dropout)

        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.MLP = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        self.linear_pred = nn.Linear(d_model, t_cycle)

    def forward(self, E_dec, Z_enc):
        batch = E_dec.shape[0]
        Z_dec = self.self_fdse(E_dec)
        Z_dec = rearrange(Z_dec, 'b ts_d out_cycle_num d_model -> (b ts_d) out_cycle_num d_model')
        cross = rearrange(Z_enc, 'b ts_d in_cycle_num d_model -> (b ts_d) in_cycle_num d_model')

        Z_cat = torch.cat((cross, Z_dec), dim=1)
        Z_mam = self.mamba(Z_cat)

        x_len = Z_dec.shape[1]
        Z_split = Z_mam[:, -x_len:, :]

        Z_tmp = self.norm1(Z_dec + self.dropout(Z_split))
        dec_output = self.norm2(Z_tmp + self.MLP(Z_tmp))

        dec_output = rearrange(dec_output, '(b ts_d) cycle_dec_num d_model -> b ts_d cycle_dec_num d_model', b=batch)

        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d cycle_num t_cycle -> b (out_d cycle_num) t_cycle')

        return dec_output, layer_predict

class Decoder(nn.Module):
    def __init__(self, t_cycle, d_model, d_state, d_ff, dropout):
        super(Decoder, self).__init__()

        self.decode_layers_0 = DecoderLayer(t_cycle, d_model, d_state, d_ff, dropout)
        self.decode_layers_1 = DecoderLayer(t_cycle, d_model, d_state, d_ff, dropout)
        self.decode_layers_2 = DecoderLayer(t_cycle, d_model, d_state, d_ff, dropout)
        self.decode_layers_3 = DecoderLayer(t_cycle, d_model, d_state, d_ff, dropout)
        self.decode_layers_4 = DecoderLayer(t_cycle, d_model, d_state, d_ff, dropout)

    def forward(self, x, cross):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        cross_enc = cross[0]
        x, layer_predict = self.decode_layers_0(x, cross_enc)
        final_predict = layer_predict

        cross_enc = cross[1]
        x, layer_predict = self.decode_layers_1(x, cross_enc)
        final_predict = final_predict + layer_predict

        cross_enc = cross[2]
        x2, layer_predict = self.decode_layers_2(x, cross_enc)
        final_predict = final_predict + layer_predict

        cross_enc = cross[3]
        x3, layer_predict = self.decode_layers_3(x, cross_enc)
        final_predict = final_predict + layer_predict

        cross_enc = cross[4]
        x4, layer_predict = self.decode_layers_4(x, cross_enc)
        final_predict = final_predict + layer_predict

        final_predict = rearrange(final_predict, 'b (out_d cycle_num) t_cycle -> b (cycle_num t_cycle) out_d', out_d=ts_d)

        return final_predict

class MamMLP(nn.Module):
    def __init__(self, d_model, d_state, d_ff, dropout=0.1):
        super(MamMLP, self).__init__()

        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)

        self.MLP = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, Z):
        Z_mam = self.norm1(self.dropout(self.mamba(Z)) + Z)
        Z_mammlp = self.norm1(self.dropout(self.MLP(Z_mam)) + Z_mam)
        return Z_mammlp

class FDSE(nn.Module):
    def __init__(self, d_model, d_state, d_ff, dropout=0.1):
        super(FDSE, self).__init__()

        self.time_MamMLP = MamMLP(d_model, d_state, d_ff, dropout=0.1)

        self.dim_MamMLP = MamMLP(d_model, d_state, d_ff, dropout=0.1)

    def forward(self, x):
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d cycle_num d_model -> (b ts_d) cycle_num d_model')

        dim_in = self.time_MamMLP(time_in)

        dim_in = rearrange(dim_in, '(b ts_d) cycle_num d_model -> (b cycle_num) ts_d d_model', b=batch)

        dim_enc = self.dim_MamMLP(dim_in)

        final_out = rearrange(dim_enc, '(b cycle_num) ts_d d_model -> b ts_d cycle_num d_model', b=batch)

        return final_out

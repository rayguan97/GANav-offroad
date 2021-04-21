
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math

class MHA(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):


        super().__init__()    
        
        self.d_model = d_model
        self.d_k = d_model // nhead
        self.h = nhead
                                            
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k ,v):
        r"""
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - Outputs:
          - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
                          E is the embedding dimension.
          - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
                                  L is the target sequence length, S is the source sequence length.
        """
        # from IPython import embed
        # embed()
        L = len(q)
        S = len(k)

        k = self.k_linear(k).view(S, -1, self.h, self.d_k)
        q = self.q_linear(q).view(L, -1, self.h, self.d_k)
        v = self.v_linear(v).view(S, -1, self.h, self.d_k).permute(1, 2, 0, 3)
        
        if L == 1:
            scores = (q * k).sum(3).permute(1,2,0).view(-1, self.h, L, S) / math.sqrt(self.d_k)
        else:
            k = k.permute(1, 2, 0, 3)
            q = q.permute(1, 2, 0, 3)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
         
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        concat = output.permute(2, 0, 1, 3).contiguous().view(L, -1, self.d_model)
        output = self.out(concat)

        return output


class CTransformer(nn.Module):
    def __init__(self, input_channel=32, out_dim = -1, stride=1, partition=[3, 3], nhead=4, num_encoder_layers=1,
                 num_decoder_layers=0, dim_feedforward=64, dropout=0.1,
                 activation="relu", normalize_before=False, temperature=10000,
                 return_intermediate_dec=False, normalize_pos=False, scale=None, linformer=False, padding=-1, encoder_norm=False, encoder_dropout=False):
        super().__init__()
        if out_dim == -1: 
            out_dim = input_channel
        self._trans = Transformer(input_channel, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, out_dim, dropout,
                 activation, normalize_before, return_intermediate_dec, linformer=linformer, encoder_dropout=encoder_dropout, encoder_norm=encoder_norm)
        self.partition = partition
        self.normalize = normalize_pos
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.pos_embed = None
        self.temperature = temperature
        if padding == -1:
            self.padding = [x//2 for x in self.partition]
        else:
            self.padding = padding
        self.unfold = torch.nn.Unfold(self.partition, padding=self.padding, stride=stride)
        self.fold = None
        self.stride = stride
        self.out_dim = out_dim

    def forward(self, src):
        b, c, h, w = src.size()
        # out = torch.zeros_like(src)
        if self.pos_embed is None or self.pos_embed.size() != src.size():
            # from IPython import embed 
            # embed()
            mask = torch.ones(b, self.partition[0], self.partition[1], device=src.device)
            y_embed = mask.cumsum(1, dtype=torch.float32)
            x_embed = mask.cumsum(2, dtype=torch.float32)
            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
            
            dim_t = torch.arange(c/2, dtype=torch.float32, device=src.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / (c/2))
            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t
            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
            self.pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            self.pos_embed = self.pos_embed.flatten(2).permute(2, 0, 1)
            # print(self.pos_embed.size())

        # print(src.size())
        blocks = self.unfold(src)
        # print(blocks.size())

        blocks = blocks.view(b, c, self.partition[0] * self.partition[1], -1).permute(0, 3, 1, 2).flatten(0, 1)
        # print(blocks.size())
        # exit(0)

        blocks = self._trans(blocks, pos_embed=self.pos_embed)
        # blocks = self._trans(blocks, pos_embed=self.pos_embed.repeat(1, blocks.shape[0]//b,1))
        # if self.fold == None or self.fold.output_size != src.size()[-2:]:
        if self.fold == None:
            self.fold = torch.nn.Fold(src.size()[-2:], kernel_size=[1, 1], padding=0, stride=self.stride)
        # print(blocks.size())
        blocks = blocks.view(b, -1, self.out_dim, 1).permute(0, 2, 3, 1).flatten(1, 2)
        # print(blocks.size())


        out = self.fold(blocks)
        # assert(out.size() == src.size())
        # print("hi")
        # exit(0)

        return out

    # the size should remain the same
    # def forward(self, src):
    #     b, c, h, w = src.size()
    #     h_grid = h // self.partition[0]
    #     w_grid = w // self.partition[1]
    #     out = torch.zeros_like(src)
    #     if self.pos_embed == None or self.pos_embed.size() != src.size():
    #         mask = torch.ones(b, h_grid, w_grid, device=src.device)
    #         y_embed = mask.cumsum(1, dtype=torch.float32)
    #         x_embed = mask.cumsum(2, dtype=torch.float32)
    #         if self.normalize:
    #             eps = 1e-6
    #             y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
    #             x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
            
    #         dim_t = torch.arange(c/2, dtype=torch.float32, device=src.device)
    #         dim_t = self.temperature ** (2 * (dim_t // 2) / (c/2))
    #         pos_x = x_embed[:, :, :, None] / dim_t
    #         pos_y = y_embed[:, :, :, None] / dim_t
    #         pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    #         pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    #         self.pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    #         self.pos_embed = self.pos_embed.flatten(2).permute(2, 0, 1)

    #     for i in range(self.partition[0]):
    #         for j in range(self.partition[1]):
    #             # print(i, j, "++")
    #             h_in = i * h_grid
    #             w_in = j * w_grid
    #             out[:,:,h_in:h_in+h_grid,w_in:w_in+w_grid] = self._trans(src[:,:,h_in:h_in+h_grid,w_in:w_in+w_grid], pos_embed=self.pos_embed)

    #     return out

class Transformer(nn.Module):

    def __init__(self, d_model=64, nhead=4, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=2048, out_dim=-1, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, linformer=False, encoder_dropout=False, encoder_norm=False, temperature=10000):
        super().__init__()
        if out_dim == -1: 
            out_dim = d_model 
        # encoder_layer = TransformerEncoderLayerSimple(d_model, nhead, dim_feedforward, out_dim,
        #                                         dropout, activation, normalize_before, linformer, encoder_dropout, encoder_norm)
        
          
        encoder_layer = TransformerEncoderLayer3D(d_model, nhead, dim_feedforward, out_dim,
                                                dropout, activation, normalize_before, linformer, True)

        encoder_norm = None
        # encoder_norm = nn.LayerNorm(out_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                   return_intermediate=return_intermediate_dec)
        
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask=None, query_embed=None, pos_embed=None):
        # flatten NxCxHxW to HWxNxC
        # print(src.size())
        bs, c, h, w = src.shape
        src = src.view(bs, c, -1)
        src = src.permute(2, 0, 1)
        # query_size = h * w
        # src -> [h * w, bs, c]
        # pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # mask = mask.flatten(1)
        # tgt = torch.zeros((query_size, bs, c)).to(src.device)
        # print(src.size())
        # pri nt(memory.size())
        # print(pos_embed.size())
        # print(pos_embed[None,:, :].repeat(src.size()[0], 1, 1).size())
        # exit(0)
        memory = self.encoder(src, pos=pos_embed)
        # hs = self.decoder(tgt, memory, pos=pos_embed)
        # print(hs.permute(1, 2, 0).view(bs, c, h, w).size())
        # exit(0)

        return memory.permute(1, 2, 0).view(bs, c, h, w)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayerSimple(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, out_dim=-1, dropout=0.1,
                 activation="relu", normalize_before=True, linformer=False, encoder_dropout=False, encoder_norm=False):
        super().__init__()

        self.linformer = False
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MHA(d_model, nhead, dropout=dropout)
    
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        if out_dim == -1: 
            out_dim = d_model
        # self.linear2 = nn.Linear(dim_feedforward, out_dim)
        self.linear = nn.Linear(d_model, out_dim)
        if encoder_norm:
            self.norm1 = nn.LayerNorm(d_model)
        self.encoder_norm = encoder_norm
        # self.norm2 = nn.LayerNorm(d_model)
        if encoder_dropout:
            self.dropout1 = nn.Dropout(dropout)
        self.encoder_dropout = encoder_dropout
        # self.dropout2 = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # print("hi")
        # exit(0)
        return tensor if pos is None else tensor + pos.repeat(1, tensor.shape[1]//pos.shape[1], 1)

    # def forward_post(self,
    #                  src,
    #                  src_mask: Optional[Tensor] = None,
    #                  src_key_padding_mask: Optional[Tensor] = None,
    #                  pos: Optional[Tensor] = None):


    #     src2 = self.self_attn(src[4:5], src, value=src, attn_mask=src_mask,
    #                           key_padding_mask=src_key_padding_mask)[0]
        
    #     src = src + self.dropout1(src2.repeat(src.size(0), 1, 1))
    #     # src = src[4:5] + self.dropout1(src2)
    #     src = self.norm1(src)
    #     src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    #     # src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))

    #     src = src + self.dropout2(src2)
    #     src = self.norm2(src)

    #     return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # from IPython import embed 
        # embed()
        if self.encoder_norm:
            src2 = self.norm1(src)
        else:
            src2 = src
        q = k = self.with_pos_embed(src2, pos)
        q = k = src2
        l = len(q) // 2
        if self.linformer:
            src2 = src2.permute(1, 0, 2)
            src2 = self.self_attn(src2)
            src2 = src2.permute(1, 0, 2)[l:l+1]
        else:           
            q = q[l:l+1]
            # from IPython import embed
            # embed()
            # src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
            #              key_padding_mask=src_key_padding_mask)[0]
            src2 = self.self_attn(q, k, src2)
        if self.encoder_dropout:
            src = src[l:l+1] + self.dropout1(src2)
        else:
            src = src2
        # src2 = self.norm2(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = self.linear(src2)
        # # src = src + self.dropout2(src2)
        # src = src2 + self.dropout2(src2)

        # exit(0)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # if self.normalize_before:
        return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        # return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, out_dim=-1, dropout=0.1,
                 activation="relu", normalize_before=True, linformer=False):
        super().__init__()

        self.linformer = False
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MHA(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        if out_dim == -1: 
            out_dim = d_model
        self.linear2 = nn.Linear(dim_feedforward, out_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # print("hi")
        # exit(0)
        return tensor if pos is None else tensor + pos.repeat(1, tensor.shape[1]//pos.shape[1], 1)

    # def forward_post(self,
    #                  src,
    #                  src_mask: Optional[Tensor] = None,
    #                  src_key_padding_mask: Optional[Tensor] = None,
    #                  pos: Optional[Tensor] = None):


    #     src2 = self.self_attn(src[4:5], src, value=src, attn_mask=src_mask,
    #                           key_padding_mask=src_key_padding_mask)[0]
        
    #     src = src + self.dropout1(src2.repeat(src.size(0), 1, 1))
    #     # src = src[4:5] + self.dropout1(src2)
    #     src = self.norm1(src)
    #     src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    #     # src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))

    #     src = src + self.dropout2(src2)
    #     src = self.norm2(src)

    #     return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # from IPython import embed 
        # embed()
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        q = k = src2
        l = len(q) // 2
        if self.linformer:
            src2 = src2.permute(1, 0, 2)
            src2 = self.self_attn(src2)
            src2 = src2.permute(1, 0, 2)[l:l+1]
        else:           
            q = q[l:l+1]
            # from IPython import embed
            # embed()
            # src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
            #              key_padding_mask=src_key_padding_mask)[0]
            src2 = self.self_attn(q, k, src2)
        src = src[l:l+1] + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # src = src + self.dropout2(src2)
        src = src2 + self.dropout2(src2)

        # exit(0)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # if self.normalize_before:
        return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        # return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, queryc_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

        
class TransformerEncoderLayer3D(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, out_dim=-1, dropout=0.1,
                 activation="relu", normalize_before=True, linformer=False, skip=False):
        super().__init__()
        if linformer:
            self.linformer = True
            self.self_attn = MHAttention(9, d_model, d_model, 2, nhead, dropout)
        else:
            self.linformer = False
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        if out_dim == -1: 
            out_dim = d_model
        self.linear2 = nn.Linear(dim_feedforward, out_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.skip = skip

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # print("hi")
        # exit(0)
        return tensor if pos is None else tensor + pos.repeat(1, tensor.shape[1]//pos.shape[1], 1)

    # def forward_post(self,
    #                  src,
    #                  src_mask: Optional[Tensor] = None,
    #                  src_key_padding_mask: Optional[Tensor] = None,
    #                  pos: Optional[Tensor] = None):


    #     src2 = self.self_attn(src[4:5], src, value=src, attn_mask=src_mask,
    #                           key_padding_mask=src_key_padding_mask)[0]
        
    #     src = src + self.dropout1(src2.repeat(src.size(0), 1, 1))
    #     # src = src[4:5] + self.dropout1(src2)
    #     src = self.norm1(src)
    #     src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    #     # src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))

    #     src = src + self.dropout2(src2)
    #     src = self.norm2(src)

    #     return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # from IPython import embed 
        # embed()
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        q = k = src2
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        # from IPython import embed
        # embed()
        # src2 = self.norm2(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # if self.skip:
        #     src = src + self.dropout2(src2)
        # else:
        #     src = src2 + self.dropout2(src2)


        # exit(0)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # if self.normalize_before:
        return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        # return self.forward_post(src, src_mask, src_key_padding_mask, pos)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

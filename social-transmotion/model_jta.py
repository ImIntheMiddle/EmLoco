import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import argparse
from datetime import datetime

class AuxilliaryEncoderCMT(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderCMT, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        attn_matrices = []

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class AuxilliaryEncoderST(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderST, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        attn_matrices = []

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class LearnedIDEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=21, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.person_encoding = nn.Embedding(1000, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:

        seq_len = 21
        import pdb; pdb.set_trace()
        x = x.to(self.device) + self.person_encoding(torch.arange(num_people).repeat_interleave(seq_len, dim=0)).to(self.device).unsqueeze(1)
        return self.dropout(x)


class LearnedTrajandIDEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=21, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model//2, max_norm=True) # maybe requires .to(device)
        self.person_encoding = nn.Embedding(1000, d_model//2, max_norm=True) # maybe requires .to(device)

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:
        seq_len = 21
        half = x.size(3)//2 ## 124
        # import pdb; pdb.set_trace()
        x[:,:,:,0:half*2:2] = x[:,:,:,0:half*2:2] + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)
        x[:,:,:,1:half*2:2] = x[:,:,:,1:half*2:2] + self.person_encoding(torch.arange(num_people).unsqueeze(0).repeat_interleave(seq_len, dim=0).to(self.device)).unsqueeze(0)

        return self.dropout(x)

class Learnedbb3dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=9, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = 9
        x = x + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)
        return self.dropout(x)

class Learnedbb2dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=9, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = 9
        x = x + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)
        return self.dropout(x)

class Learnedpose3dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=216, device='cuda:0'): # default seq_len: 198
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)
        x = x + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)
        return self.dropout(x)

class Learnedpose2dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=198, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)
        x = x + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)

class TransMotionJTA(nn.Module):
    def __init__(self, tok_dim=21, nhid=256, nhead=4, dim_feedfwd=1024, nlayers_local=2, nlayers_global=4, nmode=5, dropout=0.1, activation='relu', output_scale=1, obs_and_pred=21,  num_tokens=47, device='cuda:0', multi_modal=False):

        super(TransMotionJTA, self).__init__()
        self.seq_len = tok_dim
        self.nhid = nhid
        self.output_scale = output_scale
        self.token_num = num_tokens
        self.joints_3dpose = 24
        self.joints_2dpose = 22
        self.obs_and_pred = 21
        self.device = device
        self.multi_modal = multi_modal
        self.nmode = nmode

        self.fc_in_traj = nn.Linear(2,nhid)

        # add heads for multi-modal prediction
        if self.multi_modal:
            print("Multi-modal prediction")
            print(f"Number of modes: {self.nmode}")
            self.predict_head = []
            for i in range(self.nmode):
                self.predict_head.append(nn.Linear(nhid, 2))
                # self.add_module(f'fc_out_traj_{i}', fc_out_traj)
            self.predict_head = nn.ModuleList(self.predict_head)
        else:
            print("Deterministic prediction")
            self.fc_out_traj = nn.Linear(nhid, 2)
        self.double_id_encoder = LearnedTrajandIDEncoding(nhid, dropout, seq_len=21, device=device)
        self.id_encoder = LearnedIDEncoding(nhid, dropout, seq_len=21, device=device)

        self.scale = torch.sqrt(torch.FloatTensor([nhid])).to(device)

        self.fc_in_3dbb = nn.Linear(4,nhid)
        self.bb3d_encoder = Learnedbb3dEncoding(nhid, dropout, device=device)

        self.fc_in_2dbb = nn.Linear(4,nhid)
        self.bb2d_encoder = Learnedbb2dEncoding(nhid, dropout, device=device)

        self.fc_in_3dpose = nn.Linear(3, nhid)
        self.pose3d_encoder = Learnedpose3dEncoding(nhid, dropout, device=device)

        self.fc_in_2dpose = nn.Linear(2, nhid)
        self.pose2d_encoder = Learnedpose2dEncoding(nhid, dropout, device=device)


        encoder_layer_local = nn.TransformerEncoderLayer(d_model=nhid, nhead=nhead, dim_feedforward=dim_feedfwd, dropout=dropout, activation=activation)
        self.local_former = AuxilliaryEncoderCMT(encoder_layer_local, num_layers=nlayers_local)

        encoder_layer_global = nn.TransformerEncoderLayer(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.global_former = AuxilliaryEncoderST(encoder_layer_global, num_layers=nlayers_global)

    def forward(self, tgt, padding_mask, random_masking=False, limit_obs=0, frame_masking=False):

        B, in_F, NJ, K = tgt.shape

        F = self.obs_and_pred
        J = self.token_num

        out_F = F - in_F
        N = NJ // J # N: number of people

        ## keep padding
        pad_idx = np.repeat([in_F - 1], out_F)
        i_idx = np.append(np.arange(0, in_F), pad_idx)
        tgt = tgt[:,i_idx]
        tgt = tgt.reshape(B,F,N,J,K)

        ## add mask
        # 0.1 for training
        mask_ratio_traj = 0.2 if random_masking else 0
        # 0.1 for training
        mask_ratio_joints = 0.2 if random_masking else 0
        # 0.3 for training
        mask_ratio_modality = 0.3 if random_masking else 0
        # 0.2 for training
        mask_ratio_frame = 0.2 if frame_masking else 0

        # import pdb; pdb.set_trace()
        tgt_traj = tgt[:,:,:,0,:2].to(self.device)
        traj_mask = torch.rand((B,F,N)).float().to(self.device)
        traj_mask = traj_mask > mask_ratio_traj
        traj_mask = traj_mask.unsqueeze(3).repeat_interleave(2,dim=-1)
        tgt_traj = tgt_traj*traj_mask

        #### frame-mask
        # choose the frames to mask
        frame_mask = torch.rand((B,in_F)).float().to(self.device) > mask_ratio_frame
        frame_mask = frame_mask.unsqueeze(2).repeat_interleave(N,dim=2)
        frame_mask = frame_mask.unsqueeze(3).repeat_interleave(2,dim=3)
        tgt_traj[:,:in_F] *= frame_mask

        # if frame_masking: # new frame masking
        #     # choose the number of available frames for each person randomly
        #     # randomly select the number of frames to keep for each person
        #     num_frames = torch.randint(1, in_F, (B,N)).to(self.device)
        #     # create a mask for the frames to keep
        #     frame_mask = torch.zeros((B,in_F,N)).to(self.device)
        #     for i in range(B):
        #         for j in range(N):
        #             frame_mask[i,:num_frames[i,j],j] = 1
        #     # import pdb; pdb.set_trace()
        #     frame_mask = frame_mask.unsqueeze(3).repeat_interleave(J,dim=3).unsqueeze(4).repeat_interleave(4,dim=4)
        #     tgt[:,:in_F] *= frame_mask

        #### mask for specific modality for whole observation horizon
        modality_selection_3dbb = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,1,4)
        modality_selection_2dbb = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,1,4)
        modality_selection_3dpose = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,self.joints_3dpose,4)
        modality_selection_2dpose = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,self.joints_2dpose,4)
        modality_selection = torch.cat((modality_selection_3dbb, modality_selection_2dbb, modality_selection_3dpose, modality_selection_2dpose),3)

        # print(modality_selection.shape)
        # print(tgt.shape)
        #### target
        tgt_vis = tgt[:,:,:,1:]*modality_selection
        # import pdb; pdb.set_trace()
        tgt_3dbb = tgt_vis[:,:,:,0,:4].to(self.device)
        tgt_2dbb = tgt_vis[:,:,:,1,:4].to(self.device)
        tgt_3dpose = tgt_vis[:,:,:,2:26,:3].to(self.device) # 2:24 default
        tgt_2dpose = tgt_vis[:,:,:,26:,:2].to(self.device) # 24: default

        #### joints-mask
        joints_3d_mask = torch.rand((B,F,N,self.joints_3dpose)).float().to(self.device) > mask_ratio_joints
        joints_3d_mask = joints_3d_mask.unsqueeze(4).repeat_interleave(3,dim=-1)
        tgt_3dpose = tgt_3dpose*joints_3d_mask

        joints_2d_mask = torch.rand((B,F,N,self.joints_2dpose)).float().to(self.device) > mask_ratio_joints
        joints_2d_mask = joints_2d_mask.unsqueeze(4).repeat_interleave(2,dim=-1)
        tgt_2dpose = tgt_2dpose*joints_2d_mask

        if limit_obs!=0: # masking for the first 8 frames
            # import pdb; pdb.set_trace()
            limit_mask = torch.ones((B,F,N)).float().to(self.device)
            limit_mask[:, :(9-limit_obs)] = 0
            tgt_traj = tgt_traj*limit_mask.unsqueeze(3).repeat_interleave(2,dim=-1) # mask tgt_traj

            tgt_3dbb = tgt_3dbb*limit_mask.unsqueeze(3).repeat_interleave(4,dim=-1) # mask tgt_3dbb
            tgt_2dbb = tgt_2dbb*limit_mask.unsqueeze(3).repeat_interleave(4,dim=-1) # mask tgt_2dbb
            tgt_3dpose = tgt_3dpose*limit_mask.unsqueeze(3).repeat_interleave(self.joints_3dpose,dim=-1).unsqueeze(4).repeat_interleave(3,dim=-1) # mask tgt_3dpose
            tgt_2dpose = tgt_2dpose*limit_mask.unsqueeze(3).repeat_interleave(self.joints_2dpose,dim=-1).unsqueeze(4).repeat_interleave(2,dim=-1) # mask tgt_2dpose

        ############
        # Transformer
        ###########

        tgt_traj = self.fc_in_traj(tgt_traj)
        tgt_traj = self.double_id_encoder(tgt_traj, num_people=N)


        tgt_3dbb = self.fc_in_3dbb(tgt_3dbb[:,:9])
        tgt_3dbb = self.bb3d_encoder(tgt_3dbb)

        tgt_2dbb = self.fc_in_2dbb(tgt_2dbb[:,:9])
        tgt_2dbb = self.bb2d_encoder(tgt_2dbb)

        tgt_3dpose = tgt_3dpose[:,:9].transpose(2,3).reshape(B,-1,N,3)
        tgt_3dpose = self.fc_in_3dpose(tgt_3dpose)
        tgt_3dpose = self.pose3d_encoder(tgt_3dpose)

        tgt_2dpose = tgt_2dpose[:,:9].transpose(2,3).reshape(B,-1,N,2)
        tgt_2dpose = self.fc_in_2dpose(tgt_2dpose)
        tgt_2dpose = self.pose2d_encoder(tgt_2dpose)

        tgt_padding_mask_global = padding_mask.repeat_interleave(F, dim=1)
        tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(self.seq_len,dim=1)

        tgt_traj = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid)

        tgt_3dbb = torch.transpose(tgt_3dbb,0,1).reshape(in_F,-1,self.nhid)
        tgt_2dbb = torch.transpose(tgt_2dbb,0,1).reshape(in_F,-1,self.nhid)
        tgt_3dpose = torch.transpose(tgt_3dpose, 0,1).reshape(in_F*self.joints_3dpose, -1, self.nhid)
        tgt_2dpose = torch.transpose(tgt_2dpose, 0,1).reshape(in_F*self.joints_2dpose, -1, self.nhid)

        tgt = torch.cat((tgt_traj,tgt_3dbb,tgt_2dbb,tgt_3dpose,tgt_2dpose),0)

        out_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local)

        ##### local residual ######
        out_local = out_local * self.output_scale + tgt

        out_local = out_local[:21].reshape(21,B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid)
        out_global = self.global_former(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global)

        ##### global residual ######
        out_global = out_global * self.output_scale + out_local
        out_primary = out_global.reshape(N,F,out_global.size(1),self.nhid)[0] # primary agent

        if self.multi_modal:
            all_out = []
            for i in range(self.nmode):
                all_out.append(self.predict_head[i](out_primary))
            # import pdb; pdb.set_trace()
            all_out = torch.stack(all_out, dim=2)
            out = all_out.transpose(0,1) # multi-modal predictions (n) for primary agent
            out_primary = self.predict_head[0](out_primary)
            # out_det = out_primary.transpose(0, 1).reshape(B, F, 1, 2)
            # assert torch.allclose(out_det[:,:,0], out[:,:,0])
        else:
            out_primary = self.fc_out_traj(out_primary)
            out = out_primary.transpose(0, 1).reshape(B, F, 1, 2)
        return out

# class TransMotionJTA(nn.Module):
    # def __init__(self, tok_dim=21, nhid=256, nhead=4, dim_feedfwd=1024, nlayers_local=2, nlayers_global=4, nmode=5, dropout=0.1, activation='relu', output_scale=1, obs_and_pred=21,  num_tokens=47, device='cuda:0', multi_modal=False):

    #     super(TransMotionJTA, self).__init__()
    #     self.seq_len = tok_dim
    #     self.nhid = nhid
    #     self.output_scale = output_scale
    #     self.token_num = num_tokens
    #     self.joints_3dpose = num_tokens-1
    #     # self.joints_3dpose = 19
    #     self.joints_2dpose = 22
    #     self.obs_and_pred = 21
    #     self.device = device
    #     self.multi_modal = multi_modal
    #     self.nmode = nmode

    #     self.fc_in_traj = nn.Linear(2,nhid)

    #     # add heads for multi-modal prediction
    #     if self.multi_modal:
    #         print("Multi-modal prediction")
    #         print(f"Number of modes: {self.nmode}")
    #         self.predict_head = []
    #         for i in range(self.nmode):
    #             self.predict_head.append(nn.Linear(nhid, 2))
    #             # self.add_module(f'fc_out_traj_{i}', fc_out_traj)
    #         self.predict_head = nn.ModuleList(self.predict_head)
    #     else:
    #         print("Deterministic prediction")
    #         self.fc_out_traj = nn.Linear(nhid, 2)
    #     self.double_id_encoder = LearnedTrajandIDEncoding(nhid, dropout, seq_len=21, device=device)
    #     self.id_encoder = LearnedIDEncoding(nhid, dropout, seq_len=21, device=device)

    #     self.scale = torch.sqrt(torch.FloatTensor([nhid])).to(device)

    #     # self.fc_in_3dbb = nn.Linear(4,nhid)
    #     # self.bb3d_encoder = Learnedbb3dEncoding(nhid, dropout, device=device)

    #     # self.fc_in_2dbb = nn.Linear(4,nhid)
    #     # self.bb2d_encoder = Learnedbb2dEncoding(nhid, dropout, device=device)

    #     self.fc_in_3dpose = nn.Linear(3, nhid)
    #     self.pose3d_encoder = Learnedpose3dEncoding(nhid, dropout, device=device)

    #     # self.fc_in_2dpose = nn.Linear(2, nhid)
    #     # self.pose2d_encoder = Learnedpose2dEncoding(nhid, dropout, device=device)


    #     encoder_layer_local = nn.TransformerEncoderLayer(d_model=nhid, nhead=nhead, dim_feedforward=dim_feedfwd, dropout=dropout, activation=activation)
    #     self.local_former = AuxilliaryEncoderCMT(encoder_layer_local, num_layers=nlayers_local)

    #     encoder_layer_global = nn.TransformerEncoderLayer(d_model=nhid,
    #                                                nhead=nhead,
    #                                                dim_feedforward=dim_feedfwd,
    #                                                dropout=dropout,
    #                                                activation=activation)
    #     self.global_former = AuxilliaryEncoderST(encoder_layer_global, num_layers=nlayers_global)

    # def forward(self, tgt, padding_mask, random_masking=False, limit_obs=0, frame_masking=False):

    #     B, in_F, NJ, K = tgt.shape

    #     F = self.obs_and_pred
    #     J = self.token_num

    #     out_F = F - in_F
    #     N = NJ // J # N: number of people

    #     ## keep padding
    #     pad_idx = np.repeat([in_F - 1], out_F)
    #     i_idx = np.append(np.arange(0, in_F), pad_idx)
    #     tgt = tgt[:,i_idx]
    #     tgt = tgt.reshape(B,F,N,J,K)

    #     ## add mask
    #     # 0.1 for training
    #     mask_ratio_traj = 0.2 if random_masking else 0
    #     # 0.1 for training
    #     mask_ratio_joints = 0.2 if random_masking else 0
    #     # 0.3 for training
    #     mask_ratio_modality = 0.3 if random_masking else 0
    #     # 0.2 for training
    #     mask_ratio_frame = 0.2 if frame_masking else 0

    #     # import pdb; pdb.set_trace()
    #     tgt_traj = tgt[:,:,:,0,:2].to(self.device)
    #     traj_mask = torch.rand((B,F,N)).float().to(self.device)
    #     traj_mask = traj_mask > mask_ratio_traj
    #     traj_mask = traj_mask.unsqueeze(3).repeat_interleave(2,dim=-1)
    #     tgt_traj = tgt_traj*traj_mask

    #     #### frame-mask
    #     # choose the frames to mask
    #     frame_mask = torch.rand((B,in_F)).float().to(self.device) > mask_ratio_frame
    #     frame_mask = frame_mask.unsqueeze(2).repeat_interleave(N,dim=2)
    #     frame_mask = frame_mask.unsqueeze(3).repeat_interleave(2,dim=3)
    #     tgt_traj[:,:in_F] *= frame_mask

    #     # if frame_masking: # new frame masking
    #     #     # choose the number of available frames for each person randomly
    #     #     # randomly select the number of frames to keep for each person
    #     #     num_frames = torch.randint(1, in_F, (B,N)).to(self.device)
    #     #     # create a mask for the frames to keep
    #     #     frame_mask = torch.zeros((B,in_F,N)).to(self.device)
    #     #     for i in range(B):
    #     #         for j in range(N):
    #     #             frame_mask[i,:num_frames[i,j],j] = 1
    #     #     # import pdb; pdb.set_trace()
    #     #     frame_mask = frame_mask.unsqueeze(3).repeat_interleave(J,dim=3).unsqueeze(4).repeat_interleave(4,dim=4)
    #     #     tgt[:,:in_F] *= frame_mask

    #     #### mask for specific modality for whole observation horizon
    #     # modality_selection_3dbb = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,1,4)
    #     # modality_selection_2dbb = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,1,4)
    #     modality_selection_3dpose = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,self.joints_3dpose,4)
    #     # modality_selection_2dpose = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,self.joints_2dpose,4)
    #     # modality_selection = torch.cat((modality_selection_3dbb, modality_selection_2dbb, modality_selection_3dpose, modality_selection_2dpose),3)
    #     modality_selection = modality_selection_3dpose

    #     #### target
    #     tgt_vis = tgt[:,:,:,1:]*modality_selection
    #     # import pdb; pdb.set_trace()
    #     # tgt_3dbb = tgt_vis[:,:,:,0,:4].to(self.device)
    #     # tgt_2dbb = tgt_vis[:,:,:,1,:4].to(self.device)
    #     # tgt_3dpose = tgt_vis[:,:,:,2:26,:3].to(self.device) # 2:24 default
    #     tgt_3dpose = tgt_vis[:,:,:,:,:3].to(self.device) # 2:24 default
    #     # tgt_2dpose = tgt_vis[:,:,:,26:,:2].to(self.device) # 24: default

    #     #### joints-mask
    #     joints_3d_mask = torch.rand((B,F,N,self.joints_3dpose)).float().to(self.device) > mask_ratio_joints
    #     joints_3d_mask = joints_3d_mask.unsqueeze(4).repeat_interleave(3,dim=-1)
    #     tgt_3dpose = tgt_3dpose*joints_3d_mask

    #     # joints_2d_mask = torch.rand((B,F,N,self.joints_2dpose)).float().to(self.device) > mask_ratio_joints
    #     # joints_2d_mask = joints_2d_mask.unsqueeze(4).repeat_interleave(2,dim=-1)
    #     # tgt_2dpose = tgt_2dpose*joints_2d_mask

    #     if limit_obs!=0: # masking for the first 8 frames
    #         # import pdb; pdb.set_trace()
    #         limit_mask = torch.ones((B,F,N)).float().to(self.device)
    #         limit_mask[:, :(9-limit_obs)] = 0
    #         tgt_traj = tgt_traj*limit_mask.unsqueeze(3).repeat_interleave(2,dim=-1) # mask tgt_traj
    #         tgt_3dpose = tgt_3dpose*limit_mask.unsqueeze(3).repeat_interleave(self.joints_3dpose,dim=-1).unsqueeze(4).repeat_interleave(3,dim=-1) # mask tgt_3dpose

    #         tgt_3dbb = tgt_3dbb*limit_mask.unsqueeze(3).repeat_interleave(4,dim=-1) # mask tgt_3dbb
    #         tgt_2dbb = tgt_2dbb*limit_mask.unsqueeze(3).repeat_interleave(4,dim=-1) # mask tgt_2dbb
    #         tgt_2dpose = tgt_2dpose*limit_mask.unsqueeze(3).repeat_interleave(self.joints_2dpose,dim=-1).unsqueeze(4).repeat_interleave(2,dim=-1) # mask tgt_2dpose

    #     ############
    #     # Transformer
    #     ###########

    #     tgt_traj = self.fc_in_traj(tgt_traj)
    #     tgt_traj = self.double_id_encoder(tgt_traj, num_people=N)


    #     # tgt_3dbb = self.fc_in_3dbb(tgt_3dbb[:,:9])
    #     # tgt_3dbb = self.bb3d_encoder(tgt_3dbb)

    #     # tgt_2dbb = self.fc_in_2dbb(tgt_2dbb[:,:9])
    #     # tgt_2dbb = self.bb2d_encoder(tgt_2dbb)

    #     tgt_3dpose = tgt_3dpose[:,:9].transpose(2,3).reshape(B,-1,N,3)
    #     tgt_3dpose = self.fc_in_3dpose(tgt_3dpose)
    #     tgt_3dpose = self.pose3d_encoder(tgt_3dpose)

    #     # tgt_2dpose = tgt_2dpose[:,:9].transpose(2,3).reshape(B,-1,N,2)
    #     # tgt_2dpose = self.fc_in_2dpose(tgt_2dpose)
    #     # tgt_2dpose = self.pose2d_encoder(tgt_2dpose)

    #     tgt_padding_mask_global = padding_mask.repeat_interleave(F, dim=1)
    #     tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(self.seq_len,dim=1)

    #     tgt_traj = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid)

    #     # tgt_3dbb = torch.transpose(tgt_3dbb,0,1).reshape(in_F,-1,self.nhid)
    #     # tgt_2dbb = torch.transpose(tgt_2dbb,0,1).reshape(in_F,-1,self.nhid)
    #     tgt_3dpose = torch.transpose(tgt_3dpose, 0,1).reshape(in_F*self.joints_3dpose, -1, self.nhid)
    #     # tgt_2dpose = torch.transpose(tgt_2dpose, 0,1).reshape(in_F*self.joints_2dpose, -1, self.nhid)

    #     # tgt = torch.cat((tgt_traj,tgt_3dbb,tgt_2dbb,tgt_3dpose,tgt_2dpose),0)
    #     tgt = torch.cat((tgt_traj,tgt_3dpose),0)

    #     out_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local)

    #     ##### local residual ######
    #     out_local = out_local * self.output_scale + tgt

    #     out_local = out_local[:21].reshape(21,B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid)
    #     out_global = self.global_former(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global)

    #     ##### global residual ######
    #     out_global = out_global * self.output_scale + out_local
    #     out_primary = out_global.reshape(N,F,out_global.size(1),self.nhid)[0] # primary agent

    #     if self.multi_modal:
    #         all_out = []
    #         for i in range(self.nmode):
    #             all_out.append(self.predict_head[i](out_primary))
    #         # import pdb; pdb.set_trace()
    #         all_out = torch.stack(all_out, dim=2)
    #         out = all_out.transpose(0,1) # multi-modal predictions (n) for primary agent
    #         out_primary = self.predict_head[0](out_primary)
    #         # out_det = out_primary.transpose(0, 1).reshape(B, F, 1, 2)
    #         # assert torch.allclose(out_det[:,:,0], out[:,:,0])
    #     else:
    #         out_primary = self.fc_out_traj(out_primary)
    #         out = out_primary.transpose(0, 1).reshape(B, F, 1, 2)
    #     return out

def create_model(config, logger):
    seq_len = config["MODEL"]["seq_len"]
    token_num = config["MODEL"]["token_num"]
    nhid=config["MODEL"]["dim_hidden"]
    nhead=config["MODEL"]["num_heads"]
    nlayers_local=config["MODEL"]["num_layers_local"]
    nlayers_global=config["MODEL"]["num_layers_global"]
    dim_feedforward=config["MODEL"]["dim_feedforward"]
    nmode = config["MODEL"].get("num_modes", 1)
    multi_modal=config["MULTI_MODAL"]

    if config["MODEL"]["type"] == "transmotion":
        logger.info("Creating bert model.")
        model = TransMotionJTA(tok_dim=seq_len,
            nhid=nhid,
            nhead=nhead,
            nmode=nmode,
            dim_feedfwd=dim_feedforward,
            nlayers_local=nlayers_local,
            nlayers_global=nlayers_global,
            output_scale=config["MODEL"]["output_scale"],
            obs_and_pred=config["TRAIN"]["input_track_size"] + config["TRAIN"]["output_track_size"],
            num_tokens=token_num,
            device=config["DEVICE"],
            multi_modal=multi_modal,
        ).float().to(config["DEVICE"])
    else:
        raise ValueError(f"Model type '{config['MODEL']['type']}' not found")

    return model
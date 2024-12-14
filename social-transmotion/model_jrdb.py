import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import argparse
from datetime import datetime

from model_jta import AuxilliaryEncoderCMT, AuxilliaryEncoderST, LearnedIDEncoding, LearnedTrajandIDEncoding, Learnedbb3dEncoding, Learnedbb2dEncoding, Learnedpose3dEncoding, Learnedpose2dEncoding, TransMotionJTA


class TransMotionJRDB(TransMotionJTA):
    def __init__(self, tok_dim=21, nhid=256, nhead=4, dim_feedfwd=1024, nlayers_local=2, nlayers_global=4, nmode=5, dropout=0.1, activation='relu', output_scale=1, obs_and_pred=21,  num_tokens=47, device='cuda:0', multi_modal=False):
        super(TransMotionJRDB, self).__init__(tok_dim, nhid, nhead, dim_feedfwd, nlayers_local, nlayers_global, nmode, dropout, activation, output_scale, obs_and_pred, num_tokens, device, multi_modal)

    def forward(self, tgt, padding_mask, random_masking=False, limit_obs=0, frame_masking=False, noisy_traj=False):

        B, in_F, NJ, K = tgt.shape

        F = self.obs_and_pred
        J = self.token_num

        out_F = F - in_F
        N = NJ // J

        ## keep padding
        pad_idx = np.repeat([in_F - 1], out_F)
        i_idx = np.append(np.arange(0, in_F), pad_idx)
        tgt = tgt[:,i_idx]
        tgt = tgt.reshape(B,F,N,J,K)

        ## add mask
        # 0.1 default
        mask_ratio_traj = 0.2 if random_masking else 0
        # 0.1 default
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
        # import pdb; pdb.set_trace()s
        frame_mask = torch.rand((B,in_F)).float().to(self.device) > mask_ratio_frame
        frame_mask = frame_mask.unsqueeze(2).repeat_interleave(N,dim=2)
        frame_mask = frame_mask.unsqueeze(3).repeat_interleave(2,dim=3)
        tgt_traj[:,:in_F] *= frame_mask

        # if frame_masking:
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
        modality_selection_2dbb = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,1,4)
        modality_selection_3dpose = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,self.joints_3dpose,4)
        modality_selection = torch.cat((modality_selection_2dbb, modality_selection_3dpose),3)

        #### target
        # import pdb; pdb.set_trace()
        tgt_vis = tgt[:,:,:,1:]*modality_selection
        tgt_2dbb = tgt_vis[:,:,:,0,:4].to(self.device)
        tgt_3dpose = tgt_vis[:,:,:,1:,:3].to(self.device)

        #### joints-mask
        joints_3d_mask = torch.rand((B,F,N,self.joints_3dpose)).float().to(self.device) > mask_ratio_joints
        joints_3d_mask = joints_3d_mask.unsqueeze(4).repeat_interleave(3,dim=-1)
        tgt_3dpose = tgt_3dpose*joints_3d_mask

        if limit_obs!=0: # masking for the first 8 frames
            # import pdb; pdb.set_trace()
            limit_mask = torch.ones((B,F,N)).float().to(self.device)
            limit_mask[:, :(9-limit_obs)] = 0
            tgt_traj = tgt_traj*limit_mask.unsqueeze(3).repeat_interleave(2,dim=-1) # mask tgt_traj

            tgt_2dbb = tgt_2dbb*limit_mask.unsqueeze(3).repeat_interleave(4,dim=-1) # mask tgt_2dbb
            tgt_3dpose = tgt_3dpose*limit_mask.unsqueeze(3).repeat_interleave(self.joints_3dpose,dim=-1).unsqueeze(4).repeat_interleave(3,dim=-1) # mask tgt_3dpose

        ############
        # Transformer
        ###########

        tgt_traj = self.fc_in_traj(tgt_traj)
        tgt_traj = self.double_id_encoder(tgt_traj, num_people=N)

        tgt_2dbb = self.fc_in_2dbb(tgt_2dbb[:,:9])
        tgt_2dbb = self.bb2d_encoder(tgt_2dbb)

        tgt_3dpose = tgt_3dpose[:,:9].transpose(2,3).reshape(B,-1,N,3)
        tgt_3dpose = self.fc_in_3dpose(tgt_3dpose)
        tgt_3dpose = self.pose3d_encoder(tgt_3dpose)

        tgt_padding_mask_global = padding_mask.repeat_interleave(F, dim=1)
        tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(self.seq_len,dim=1)

        tgt_traj = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid)
        tgt_2dbb = torch.transpose(tgt_2dbb,0,1).reshape(in_F,-1,self.nhid)
        tgt_3dpose = torch.transpose(tgt_3dpose, 0,1).reshape(in_F*self.joints_3dpose, -1, self.nhid)

        tgt = torch.cat((tgt_traj,tgt_2dbb,tgt_3dpose),0)

        out_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local)

        ##### local residual ######
        out_local = out_local * self.output_scale + tgt

        out_local = out_local[:21].reshape(21,B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid)
        out_global = self.global_former(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global)

        ##### global residual ######
        out_global = out_global * self.output_scale + out_local
        out_primary = out_global.reshape(N,F,out_global.size(1),self.nhid)[0]

        if self.multi_modal:
            all_out = []
            for i in range(self.nmode):
                all_out.append(self.predict_head[i](out_primary))
            all_out = torch.stack(all_out, dim=2)
            out = all_out.transpose(0,1) # multi-modal predictions (n) for primary agent
            out_primary = self.predict_head[0](out_primary)
            out_det = out_primary.transpose(0, 1).reshape(B, F, 1, 2)
            assert torch.allclose(out_det[:,:,0], out[:,:,0])
        else:
            out_primary = self.fc_out_traj(out_primary)
            out = out_primary.transpose(0, 1).reshape(B, F, 1, 2)

        return out

# class TransMotionJRDB(TransMotionJTA):
#     def __init__(self, tok_dim=21, nhid=256, nhead=4, dim_feedfwd=1024, nlayers_local=2, nlayers_global=4, nmode=5, dropout=0.1, activation='relu', output_scale=1, obs_and_pred=21,  num_tokens=47, device='cuda:0', multi_modal=False):
#         super(TransMotionJRDB, self).__init__(tok_dim, nhid, nhead, dim_feedfwd, nlayers_local, nlayers_global, nmode, dropout, activation, output_scale, obs_and_pred, num_tokens, device, multi_modal)

#     def forward(self, tgt, padding_mask, random_masking=False, limit_obs=0, frame_masking=False, noisy_traj=False):

#         B, in_F, NJ, K = tgt.shape

#         F = self.obs_and_pred
#         J = self.token_num

#         out_F = F - in_F
#         N = NJ // J

#         ## keep padding
#         pad_idx = np.repeat([in_F - 1], out_F)
#         i_idx = np.append(np.arange(0, in_F), pad_idx)
#         tgt = tgt[:,i_idx]
#         tgt = tgt.reshape(B,F,N,J,K)

#         ## add mask
#         # 0.1 default
#         mask_ratio_traj = 0.2 if random_masking else 0
#         # 0.1 default
#         mask_ratio_joints = 0.2 if random_masking else 0
#         # 0.3 for training
#         mask_ratio_modality = 0.3 if random_masking else 0
#         # 0.2 for training
#         mask_ratio_frame = 0.2 if frame_masking else 0

#         # import pdb; pdb.set_trace()
#         tgt_traj = tgt[:,:,:,0,:2].to(self.device)
#         traj_mask = torch.rand((B,F,N)).float().to(self.device)
#         traj_mask = traj_mask > mask_ratio_traj
#         traj_mask = traj_mask.unsqueeze(3).repeat_interleave(2,dim=-1)
#         tgt_traj = tgt_traj*traj_mask

#         #### frame-mask
#         # choose the frames to mask
#         # import pdb; pdb.set_trace()s
#         frame_mask = torch.rand((B,in_F)).float().to(self.device) > mask_ratio_frame
#         frame_mask = frame_mask.unsqueeze(2).repeat_interleave(N,dim=2)
#         frame_mask = frame_mask.unsqueeze(3).repeat_interleave(2,dim=3)
#         tgt_traj[:,:in_F] *= frame_mask

#         # if frame_masking:
#         #     # choose the number of available frames for each person randomly
#         #     # randomly select the number of frames to keep for each person
#         #     num_frames = torch.randint(1, in_F, (B,N)).to(self.device)
#         #     # create a mask for the frames to keep
#         #     frame_mask = torch.zeros((B,in_F,N)).to(self.device)
#         #     for i in range(B):
#         #         for j in range(N):
#         #             frame_mask[i,:num_frames[i,j],j] = 1
#         #     # import pdb; pdb.set_trace()
#         #     frame_mask = frame_mask.unsqueeze(3).repeat_interleave(J,dim=3).unsqueeze(4).repeat_interleave(4,dim=4)
#         #     tgt[:,:in_F] *= frame_mask


#         #### mask for specific modality for whole observation horizon
#         # modality_selection_2dbb = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,1,4)
#         modality_selection_3dpose = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,self.joints_3dpose,4)
#         # modality_selection = torch.cat((modality_selection_2dbb, modality_selection_3dpose),3)
#         modality_selection = modality_selection_3dpose

#         #### target
#         # import pdb; pdb.set_trace()
#         tgt_vis = tgt[:,:,:,1:]*modality_selection
#         # tgt_2dbb = tgt_vis[:,:,:,0,:4].to(self.device)
#         tgt_3dpose = tgt_vis[:,:,:,:,:3].to(self.device)

#         #### joints-mask
#         joints_3d_mask = torch.rand((B,F,N,self.joints_3dpose)).float().to(self.device) > mask_ratio_joints
#         joints_3d_mask = joints_3d_mask.unsqueeze(4).repeat_interleave(3,dim=-1)
#         tgt_3dpose = tgt_3dpose*joints_3d_mask

#         if limit_obs!=0: # masking for the first 8 frames
#             # import pdb; pdb.set_trace()
#             limit_mask = torch.ones((B,F,N)).float().to(self.device)
#             limit_mask[:, :(9-limit_obs)] = 0
#             tgt_traj = tgt_traj*limit_mask.unsqueeze(3).repeat_interleave(2,dim=-1) # mask tgt_traj
#             tgt_3dpose = tgt_3dpose*limit_mask.unsqueeze(3).repeat_interleave(self.joints_3dpose,dim=-1).unsqueeze(4).repeat_interleave(3,dim=-1) # mask tgt_3dpose

#             # tgt_2dbb = tgt_2dbb*limit_mask.unsqueeze(3).repeat_interleave(4,dim=-1) # mask tgt_2dbb

#         ############
#         # Transformer
#         ###########

#         tgt_traj = self.fc_in_traj(tgt_traj)
#         tgt_traj = self.double_id_encoder(tgt_traj, num_people=N)

#         # tgt_2dbb = self.fc_in_2dbb(tgt_2dbb[:,:9])
#         # tgt_2dbb = self.bb2d_encoder(tgt_2dbb)

#         tgt_3dpose = tgt_3dpose[:,:9].transpose(2,3).reshape(B,-1,N,3)
#         tgt_3dpose = self.fc_in_3dpose(tgt_3dpose)
#         tgt_3dpose = self.pose3d_encoder(tgt_3dpose)

#         tgt_padding_mask_global = padding_mask.repeat_interleave(F, dim=1)
#         tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(self.seq_len,dim=1)

#         tgt_traj = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid)
#         # tgt_2dbb = torch.transpose(tgt_2dbb,0,1).reshape(in_F,-1,self.nhid)
#         tgt_3dpose = torch.transpose(tgt_3dpose, 0,1).reshape(in_F*self.joints_3dpose, -1, self.nhid)

#         # tgt = torch.cat((tgt_traj,tgt_2dbb,tgt_3dpose),0)
#         tgt = torch.cat((tgt_traj,tgt_3dpose),0)

#         out_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local)

#         ##### local residual ######
#         out_local = out_local * self.output_scale + tgt

#         out_local = out_local[:21].reshape(21,B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid)
#         out_global = self.global_former(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global)

#         ##### global residual ######
#         out_global = out_global * self.output_scale + out_local
#         out_primary = out_global.reshape(N,F,out_global.size(1),self.nhid)[0]

#         if self.multi_modal:
#             all_out = []
#             for i in range(self.nmode):
#                 all_out.append(self.predict_head[i](out_primary))
#             all_out = torch.stack(all_out, dim=2)
#             out = all_out.transpose(0,1) # multi-modal predictions (n) for primary agent
#             out_primary = self.predict_head[0](out_primary)
#             out_det = out_primary.transpose(0, 1).reshape(B, F, 1, 2)
#             assert torch.allclose(out_det[:,:,0], out[:,:,0])
#         else:
#             out_primary = self.fc_out_traj(out_primary)
#             out = out_primary.transpose(0, 1).reshape(B, F, 1, 2)

#         return out

def create_model(config, logger):
    seq_len = config["MODEL"]["seq_len"]
    token_num = config["MODEL"]["token_num"]
    nhid=config["MODEL"]["dim_hidden"]
    nhead=config["MODEL"]["num_heads"]
    nmode=config["MODEL"].get("num_modes", 1)
    nlayers_local=config["MODEL"]["num_layers_local"]
    nlayers_global=config["MODEL"]["num_layers_global"]
    dim_feedforward=config["MODEL"]["dim_feedforward"]
    multi_modal = config["MULTI_MODAL"]

    if config["MODEL"]["type"] == "transmotion":
        logger.info("Creating bert model.")
        model = TransMotionJRDB(tok_dim=seq_len,
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
            multi_modal=multi_modal).float().to(config["DEVICE"])
    else:
        raise ValueError(f"Model type '{config['MODEL']['type']}' not found")

    return model
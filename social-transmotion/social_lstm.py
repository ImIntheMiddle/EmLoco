import itertools
import copy

import numpy as np
import torch

NAN = float('nan')

@staticmethod
def paths_to_xy(paths):
    """Convert paths to numpy array with nan as blanks."""
    frames = set(r.frame for r in paths[0])
    pedestrians = set(row.pedestrian
                      for path in paths
                      for row in path if row.frame in frames)
    paths = [path for path in paths if path[0].pedestrian in pedestrians]
    frames = sorted(frames)
    pedestrians = list(pedestrians)

    frame_to_index = {frame: i for i, frame in enumerate(frames)}
    xy = np.full((len(frames), len(pedestrians), 3), np.nan)

    for ped_index, path in enumerate(paths):
        for row in path:
            if row.frame not in frame_to_index:
                continue
            entry = xy[frame_to_index[row.frame]][ped_index]
            entry[0] = row.x
            entry[1] = row.y
            entry[2] = row.v

    return xy

class InputEmbedding(torch.nn.Module):
    """Linear embedding, ReLU non-linearity, input scaling.

    Input scaling is important for ReLU activation combined with the initial
    bias distribution. Otherwise some units will never be active.
    """
    def __init__(self, input_dim, embedding_dim, scale, use_tags=True):
        super(InputEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.use_tags = use_tags

        linear_embedding_dim = self.embedding_dim
        if use_tags:
            linear_embedding_dim -= 3
        self.input_embeddings = torch.nn.Sequential(
            torch.nn.Linear(input_dim, linear_embedding_dim),
            torch.nn.ReLU(),
        )

    def forward(self, vel):
        if self.use_tags:
            return torch.cat([
                self.input_embeddings(vel * self.scale),
                torch.zeros(vel.size(0),3, device=vel.device),
            ], dim=1)
        return self.input_embeddings(vel * self.scale)


class Hidden2Normal(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(Hidden2Normal, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, 3)

    def forward(self, hidden_state):
        normal = self.linear(hidden_state)
        normal[:,2] = torch.sigmoid(normal[:,2])

        return normal

def keep_valid_neigh(xy):
    xy_n_t = np.transpose(xy, (1, 0, 2))
    mask = np.ones(xy_n_t.shape[0], dtype=bool)
    for n in range(xy_n_t.shape[0]):
        if xy_n_t[n,7,-1]==0 or xy_n_t[n,8,-1]==0:
            ## need at least 2 last frames
            mask[n] = False
            
    return np.transpose(xy_n_t[mask], (1, 0, 2))


def drop_distant_far(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy[:, :, 0:2] - xy[:, 0:1, 0:2]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2

    return xy[:, mask], mask


class LSTM(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None, pool_to_input=True, goal_dim=None, goal_flag=False):
        """ Initialize the LSTM forecasting model

        Attributes
        ----------
        embedding_dim : Embedding dimension of location coordinates
        hidden_dim : Dimension of hidden state of LSTM
        pool : interaction module
        pool_to_input : Bool
            if True, the interaction vector is concatenated to the input embedding of LSTM [preferred]
            if False, the interaction vector is added to the LSTM hidden-state
        goal_dim : Embedding dimension of the unit vector pointing towards the goal
        goal_flag: Bool
            if True, the embedded goal vector is concatenated to the input embedding of LSTM 
        """

        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pool = pool
        self.pool_to_input = pool_to_input

        ## Location
        scale = 4.0
        self.input_embedding = InputEmbedding(3, self.embedding_dim, scale)

        ## Goal
        self.goal_flag = goal_flag
        self.goal_dim = goal_dim or embedding_dim
        self.goal_embedding = InputEmbedding(2, self.goal_dim, scale)
        goal_rep_dim = self.goal_dim if self.goal_flag else 0

        ## Pooling
        pooling_dim = 0
        if pool is not None and self.pool_to_input:
            pooling_dim = self.pool.out_dim 
        
        ## LSTMs
        self.encoder = torch.nn.LSTMCell(self.embedding_dim + goal_rep_dim + pooling_dim, self.hidden_dim)
        self.decoder = torch.nn.LSTMCell(self.embedding_dim + goal_rep_dim + pooling_dim, self.hidden_dim)

        self.hidden2normal = Hidden2Normal(self.hidden_dim)

    def step(self, lstm, hidden_cell_state, obs1, obs2, batch_split):
        """Do one step of prediction: two inputs to one normal prediction.
        
        Parameters
        ----------
        lstm: torch nn module [Encoder / Decoder]
            The module responsible for prediction
        hidden_cell_state : tuple (hidden_state, cell_state)
            Current hidden_cell_state of the pedestrians
        obs1 : Tensor [num_tracks, 2]
            Previous x-y positions of the pedestrians
        obs2 : Tensor [num_tracks, 2]
            Current x-y positions of the pedestrians
        goals : Tensor [num_tracks, 2]
            Goal coordinates of the pedestrians
        
        Returns
        -------
        hidden_cell_state : tuple (hidden_state, cell_state)
            Updated hidden_cell_state of the pedestrians
        normals : Tensor [num_tracks, 5]
            Parameters of a multivariate normal of the predicted position 
            with respect to the current position
        """
        num_tracks = len(obs2)
        track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0

    

        ## Masked Hidden Cell State
        hidden_cell_stacked = [
            torch.stack([h for m, h in zip(track_mask, hidden_cell_state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(track_mask, hidden_cell_state[1]) if m], dim=0),
        ]

        ## Mask current velocity & embed
        obs1_temp = obs1.clone()
        obs1_temp[:,-1]=0 # zero out current visibility
        curr_velocity = obs2 - obs1_temp # visibility depends on obs2
        curr_velocity = curr_velocity[track_mask]

        input_emb = self.input_embedding(curr_velocity)

        ## Mask & Pool per scene
        if self.pool is not None:
            hidden_states_to_pool = torch.stack(hidden_cell_state[0]).clone() # detach?
            batch_pool = []
            ## Iterate over scenes
            for (start, end) in zip(batch_split[:-1], batch_split[1:]):
                ## Mask for the scene
                scene_track_mask = track_mask[start:end]
                ## Get observations and hidden-state for the scene
                prev_position = obs1[start:end][scene_track_mask]
                curr_position = obs2[start:end][scene_track_mask]
                curr_hidden_state = hidden_states_to_pool[start:end][scene_track_mask]

                ## Provide track_mask to the interaction encoders
                ## Everyone absent by default. Only those visible in current scene are present
                interaction_track_mask = torch.zeros(num_tracks, device=obs1.device).bool()
                interaction_track_mask[start:end] = track_mask[start:end]
                self.pool.track_mask = interaction_track_mask

                ## Pool
                pool_sample = self.pool(curr_hidden_state, prev_position, curr_position)
                batch_pool.append(pool_sample)

            pooled = torch.cat(batch_pool)

            if self.pool_to_input:
                input_emb = torch.cat([input_emb, pooled], dim=1)
            else:
                hidden_cell_stacked[0] += pooled

        # LSTM step
        hidden_cell_stacked = lstm(input_emb, hidden_cell_stacked)
        normal_masked = self.hidden2normal(hidden_cell_stacked[0])

        # unmask [Update hidden-states and next velocities of pedestrians]
        normal = torch.full((track_mask.size(0),3), NAN, device=obs1.device)
        mask_index = [i for i, m in enumerate(track_mask) if m]
        for i, h, c, n in zip(mask_index,
                              hidden_cell_stacked[0],
                              hidden_cell_stacked[1],
                              normal_masked):
            hidden_cell_state[0][i] = h
            hidden_cell_state[1][i] = c
            normal[i] = n

        return hidden_cell_state, normal

    def forward(self, observed, batch_split, prediction_truth=None, n_predict=None):
        """Forecast the entire sequence

        Parameters
        ----------
        observed : Tensor [obs_length, num_tracks, 2] # truly its [obs_length, num_tracks, 3] because of visibility
            Observed sequences of x-y coordinates of the pedestrians
        goals : Tensor [num_tracks, 2]
            Goal coordinates of the pedestrians
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene
        prediction_truth : Tensor [pred_length - 1, num_tracks, 2]
            Prediction sequences of x-y coordinates of the pedestrians
            Helps in teacher forcing wrt neighbours positions during training
        n_predict: Int
            Length of sequence to be predicted during test time

        Returns
        -------
        rel_pred_scene : Tensor [pred_length, num_tracks, 5]
            Predicted velocities of pedestrians as multivariate normal
            i.e. positions relative to previous positions
        pred_scene : Tensor [pred_length, num_tracks, 2]
            Predicted positions of pedestrians i.e. absolute positions
        """

        # import pdb; pdb.set_trace()
        assert ((prediction_truth is None) + (n_predict is None)) == 1
        if n_predict is not None:
            # -1 because one prediction is done by the encoder already
            prediction_truth = [None for _ in range(n_predict - 1)]

        ego_invis_frames = np.argwhere((observed[:,batch_split[:-1],-1].cpu()==0))

        if ego_invis_frames.size(1) != 0:
            for f in range(ego_invis_frames.size(1)):
                temp_t = ego_invis_frames[0,-(f+1)]
                temp_id = ego_invis_frames[1,-(f+1)]
                observed[temp_t,batch_split[:-1][temp_id]] = observed[temp_t+1,batch_split[:-1][temp_id]]

        # initialize: Because of tracks with different lengths and the masked
        # update, the hidden state for every LSTM needs to be a separate object
        # in the backprop graph. Therefore: list of hidden states instead of
        # a single higher rank Tensor.
        num_tracks = observed.size(1)
        hidden_cell_state = (
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(num_tracks)],
            [torch.zeros(self.hidden_dim, device=observed.device) for _ in range(num_tracks)],
        )

        ## Reset LSTMs of Interaction Encoders.
        if self.pool is not None:
            self.pool.reset(num_tracks, device=observed.device)

        # list of predictions
        normals = []  # predicted normal parameters for both phases
        positions = []  # true (during obs phase) and predicted positions

        if len(observed) == 2:
            positions = [observed[-1]]

        # encoder
        for obs1, obs2 in zip(observed[:-1], observed[1:]):
            ##LSTM Step

            hidden_cell_state, normal = self.step(self.encoder, hidden_cell_state, obs1, obs2, batch_split)

            normals.append(normal)
            obs2_temp = obs2.clone()
            obs2_temp[:,-1] = 0 # zero out visibility
            positions.append(obs2_temp + normal[:, :3])  # no sampling, just mean

        # initialize predictions with last position to form velocity. DEEP COPY !!!
        prediction_truth = copy.deepcopy(list(itertools.chain.from_iterable(
            (observed[-1:], prediction_truth)
        )))

        # decoder, predictions
        for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
            if obs1 is None:
                obs1 = positions[-2].detach()  # DETACH!!!
            else:
                for primary_id in batch_split[:-1]:
                    obs1[primary_id] = positions[-2][primary_id].detach()  # DETACH!!!
            if obs2 is None:
                obs2 = positions[-1].detach()
            else:
                for primary_id in batch_split[:-1]:
                    obs2[primary_id] = positions[-1][primary_id].detach()  # DETACH!!!

            hidden_cell_state, normal = self.step(self.decoder, hidden_cell_state, obs1, obs2, batch_split)

            normals.append(normal)
            obs2_temp = obs2.clone()
            obs2_temp[:,-1] = 0
            positions.append(obs2_temp + normal[:, :3])  # no sampling, just mean

        rel_pred_scene = torch.stack(normals, dim=0)
        pred_scene = torch.stack(positions, dim=0)
        return rel_pred_scene, pred_scene

class LSTMPredictor(object):
    def __init__(self, model):
        self.model = model

    def save(self, state, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)

        # # during development, good for compatibility across API changes:
        # # Save state for optimizer to continue training in future
        with open(filename + '.state', 'wb') as f:
            torch.save(state, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return torch.load(f, map_location='cpu')


    def __call__(self, paths, n_predict=12, modes=1, predict_all=True, obs_length=9, start_length=0, args=None):


        self.model.eval()
        with torch.no_grad():
            xy = paths_to_xy(paths)
            if xy[obs_length-2,0,-1]==0 or xy[obs_length-1,0,-1]==0:
                return {}
            xy = keep_valid_neigh(xy)
            observed_xy = xy
            batch_split = [0, observed_xy.shape[1]]
            observed_xy = torch.Tensor(observed_xy)
            batch_split = torch.Tensor(batch_split).long()

            multimodal_outputs = {}
            for num_p in range(modes):
                _, output_scenes = self.model(observed_xy[start_length:obs_length], batch_split, n_predict=n_predict)
                output_scenes = output_scenes.numpy()
    
                output_primary = output_scenes[-n_predict:, 0]
                output_neighs = output_scenes[-n_predict:, 1:]
                ## Dictionary of predictions. Each key corresponds to one mode
                multimodal_outputs[num_p] = [output_primary, output_neighs]

        ## Return Dictionary of predictions. Each key corresponds to one mode
        return multimodal_outputs

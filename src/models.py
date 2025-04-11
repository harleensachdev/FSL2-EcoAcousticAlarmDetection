
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (EMBEDDING_DIM, N_WAY, N_SUPPORT, 
                   PROTO_WEIGHT, RELATION_WEIGHT, TEMPERATURE)
class CombinedFreqTemporalCNNEncoder(nn.Module):
    # Frequency attention and temporal structure encoder

    def __init__(self, n_classes=N_WAY, rnn_type = 'gru', bidirectional = True, hidden_size = 128):
        super().__init__()
        # Input format - [batch, 1, n_mels, time] -> typical mel-spectrogram format

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # preserve time dimension for temporal modelling, so we use adaptive pooling only on frequency dimensions
        # pooling over frequency compresses each frame into 4 representative frequency bins, retain time resolution
        # network learns what frequency ranges are important, doesn't discard what time

        self.time_preserve_pool = nn.AdaptiveAvgPool2d((4, None))

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if rnn_type.lower() == 'lstm': # can try
            self.rnn = nn.LSTM(
                input_size = 128 * 4, # 128 channels * 4 frequency bins
                hidden_size = hidden_size,
                batch_first = True,
                bidirectional = bidirectional
        )
        else:  # Default to GRU -> very high accuracy
            # input to gru (gated recurrent unit) which scans across time to  understand patterns/sequences
            # bidirectional grue can see both past and future context
            # this allows Grus to model temporal dependencies as bird alarm calls aren't just sounds, but squences with rhythmici patterns
            # input shape into gru [batch, time_steps, 128*4]
            self.rnn = nn.GRU(
                input_size=128 * 4,  # 128 channels x 4 frequency bins
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=bidirectional
            )

        # attention mechanism - focus on the important tiem  step, ignore irrelevant frames
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 1),
            nn.Tanh()
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_size * self.num_directions, EMBEDDING_DIM)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(EMBEDDING_DIM, n_classes)  # Classification layer
        self.log_softmax = nn.LogSoftmax(dim=1)

    
    def forward(self, input_data, return_embedding = True):
        # Handle multiple possible input tensor shapes
        
        # If input is 6D, try to reshape
        if input_data.dim() == 6:
            # Reshape from [batch, k_shot, 1, 1, height, width] to [batch * k_shot, 1, height, width]
            batch_size, k_shot, channels, _, height, width = input_data.shape
            input_data = input_data.view(batch_size * k_shot, channels, height, width)
        
        # Handle 5D input tensor 
        elif input_data.dim() == 5:
            # Reshape from [batch, n_samples, channels, height, width] to [batch * n_samples, channels, height, width]
            batch_size, n_samples, channels, height, width = input_data.shape
            input_data = input_data.view(batch_size * n_samples, channels, height, width)
        
        # Ensure input has proper dimensions for spectrogram
        # Check if batch dimension exists, if not add it
        if input_data.dim() == 2:  # Single spectrogram with shape [n_mels, time]
            input_data = input_data.unsqueeze(0)  # Add batch dimension: [1, n_mels, time]
            input_data = input_data.unsqueeze(1)  # Add channel dimension: [1, 1, n_mels, time]
        elif input_data.dim() == 3:
            # This could be [batch, n_mels, time] or [1, n_mels, time]
            if input_data.size(0) == 1 and len(input_data) == 1:
                # It's likely [1, n_mels, time], add channel dim
                input_data = input_data.unsqueeze(1)
            else:
                # It's likely [batch, n_mels, time], add channel dim to each
                input_data = input_data.unsqueeze(1)
        
        # Ensure input is 4D [batch, channels, height, width]
        if input_data.dim() != 4:
            raise ValueError(f"Unexpected input tensor shape: {input_data.shape}")
        
        batch_size = input_data.size(0)  # remember original batch size for shaping

        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Preserve the time dimension by only pooling frequency
        x = self.time_preserve_pool(x) # [batch, 128, 4, T]
        # T = number of time steps (depends on length of audio (1 sec) , hop size, and downsampling of cnn)

        # reshape for rnn [batch, timesteps, features]
        # transpose to get time as the second dimension (in x)
        x = x.permute(0, 3, 1, 2 ) # -> 0,3,1,2 = order
        time_steps = x.size(1)
        x = x.reshape(batch_size, time_steps, -1) #[batch, T, 128*4], flattening channel and frequency feature into single feature vector per time step 

        # process with rnn, using 128 neurons in the hidden layer of the rnn, learning a 128 dimensional represntation, and outputting 256 (bidirectional)
        rnn_out, _ = self.rnn(x) #[batch, T, hidden_size*2], (2 for bidirectional)

        # apply attention on important time stpes
        attention_weights = self.attention(rnn_out) # small neural network (linear layer + tanh, for each time step, outputs score on importance, ignoring silence /noise)
        attention_weights = F.softmax(attention_weights, dim = 1) # probability distribution of how likely each frame is to be important

        # get context vector with attention weight, removing irrelevant time slices
        # weighted sum of time step inputs 
        # rnn_out : batch ,t, hidden_size * 2
        # attention_weights: [batch, t, 1]

        # broadcast, scaling time step by how important it is
        # torch.sum sums it over time into a single vector per sample, filtered to emphasize relevant parts
        context = torch.sum(rnn_out * attention_weights, dim = 1) # [batch, hidden_size * num_directions]
        
        # Get embeddings
        embedding = self.fc1(context)
        
        # Return embeddings if requested
        if return_embedding:
            return embedding
        
        # Otherwise continue with classification
        x = self.dropout(embedding)
        logits = self.fc2(x)
        return self.log_softmax(logits)

class RelationNetwork(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        
        # The relation module takes concatenated embeddings from two samples
        self.relation_module = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output similarity score between 0 and 1
        )
    
    def forward(self,x):
        # seperate/ preconcatenated
        if isinstance(x, tuple) or isinstance(x,list):
            embedding1, embedding2 = x
                
            # Ensure both embeddings have the same shape
            if embedding1.dim() == 1:
                embedding1 = embedding1.unsqueeze(0)
            if embedding2.dim() == 1:
                embedding2 = embedding2.unsqueeze(0)
            
            # Concatenate the two embeddings
            combined = torch.cat([embedding1, embedding2], dim=1)
        else:
            combined = x
        
        # Pass through relation module to get similarity score
        return self.relation_module(combined)


class PrototypicalNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, support_images, support_labels, query_images, n_way, n_support):
        """
        Implements the prototypical network for classification
        
        Parameters:
            support_images: [n_way*n_support, C, H, W] or [n_way*n_support, H, W] support images
            support_labels: [n_way*n_support] support labels
            query_images: [n_query, C, H, W] or [n_query, H, W] query images
            n_way: number of classes
            n_support: number of examples per class in support set
            
        Returns:
            log_p_y: [n_query, n_way] log probabilities for each query
        """
        # Extract feature embeddings for support and query sets
        support_embeddings = self.encoder(support_images, return_embedding=True)
        query_embeddings = self.encoder(query_images, return_embedding=True)
        
        # Get unique classes
        unique_labels = torch.unique(support_labels)
        
        # Ensure we have the right number of classes
        if len(unique_labels) != n_way:
            raise ValueError(f"Expected {n_way} unique classes but got {len(unique_labels)}")
        
        # Compute prototypes
        prototypes = torch.zeros(n_way, support_embeddings.shape[1], device=support_embeddings.device)
        for i, label in enumerate(unique_labels):
            mask = support_labels == label
            prototypes[i] = support_embeddings[mask].mean(dim=0)

        query_embeddings_norm = F.normalize(query_embeddings, p=2, dim=1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    
        # Calculate cosine similarity with temperature scaling
        cosine_sim = torch.mm(query_embeddings_norm, prototypes_norm.t()) * TEMPERATURE

        dists = 1 - cosine_sim  # convert similarity to distance
        # Convert distances to log probabilities
        log_p_y = F.log_softmax(-dists, dim=1)
        
        return log_p_y
    
    def classify(self, support_images, support_labels, query_images, n_way, n_support):
        """
        Perform classification using prototypical network
        
        Returns:
            predicted_labels: [n_query] predicted class indices for each query
        """
        log_p_y = self.forward(support_images, support_labels, query_images, n_way, n_support)
        _, predicted_labels = torch.max(log_p_y, dim=1)
        return predicted_labels


class EnsembleModel(nn.Module):
    def __init__(self, encoder, relation_net = None):
        super().__init__()
        self.encoder = encoder
        self.proto_net = PrototypicalNet(encoder)
        if relation_net is None:
            self.relation_net = RelationNetwork(EMBEDDING_DIM)
        else:
            self.relation_net = relation_net
    
    def forward(self, x):
        """
        Standard forward method for typical classification
        
        Parameters:
            x: input images [batch, channels, height, width]
        
        Returns:
            class probabilities
        """
        # Use encoder's standard forward method
        return self.encoder(x)
    
    def few_shot_classify(self, support_images, support_labels, query_images, 
                       n_way=N_WAY, n_support=N_SUPPORT,
                       proto_weight=PROTO_WEIGHT, relation_weight=RELATION_WEIGHT):
        """
        Few-shot classification method that matches the previous implementation
        
        Parameters:
            support_images: [n_way*n_support, C, H, W] support 
            support_labels: [n_way*n_support] support labels
            query_images: [n_query, C, H, W] query images
            n_way: number of classes
            n_support: number of examples per class in support set
            proto_weight: weight for prototypical network predictions
            relation_weight: weight for relation network predictions
            
        Returns:
            predicted_labels: [n_query] predicted class indices for each query
        """
        proto_log_probs = self.proto_net.forward(support_images, support_labels, query_images, n_way,n_support)
        proto_probs = torch.exp(proto_log_probs)
        
        # Extract embeddings for relation network
        support_embeddings = self.encoder(support_images, return_embedding=True)
        query_embeddings = self.encoder(query_images, return_embedding=True)
        
        # Get unique classes
        unique_labels = torch.unique(support_labels)
        
        # Ensure we have the right number of classes
        if len(unique_labels) != n_way:
            raise ValueError(f"Expected {n_way} unique classes but got {len(unique_labels)}")
        
        # Compute prototypes for each class
        prototypes = torch.zeros(n_way, support_embeddings.shape[1], 
                                device=support_embeddings.device)
        for i, label in enumerate(unique_labels):
            mask = support_labels == label
            prototypes[i] = support_embeddings[mask].mean(dim=0)
        
        # Calculate relation scores for each query-prototype pair
        relation_scores = torch.zeros(query_embeddings.size(0), n_way, 
                                     device=query_embeddings.device)
        
        for i, query_emb in enumerate(query_embeddings):
            for j, prototype in enumerate(prototypes):
                score = self.relation_net(query_emb.unsqueeze(0), prototype.unsqueeze(0))
                relation_scores[i, j] = score
        
        # Normalize relation scores to sum to 1 (convert to probabilities)
        relation_probs = relation_scores / relation_scores.sum(dim=1, keepdim=True)
        
        # Combine predictions using weighted average
        ensemble_probs = proto_weight * proto_probs + relation_weight * relation_probs
        
        # Return predicted labels
        _, predicted_labels = torch.max(ensemble_probs, dim=1)
        return predicted_labels
# FSL2-EcoAcousticAlarmDetection

FSL2-EcoAcousticAlarmDetection is a few-shot learning model designed to classify ecological audio recordings into three categories: alarm, non-alarm, and background. The model begins by converting MP3 or WAV files into Mel spectrograms and, for each episode, randomly splits samples into a support set (5 samples per class), query set (6 samples per class), and test set (30 samples per class). Using an episodic batch sampler, 100 training episodes are generated. A CNN encoder with four convolutional blocks extracts embeddings from spectrograms, optimized via the Adam optimizer and cross-entropy loss. However, to mantain temporal structure, encoder employs an adaptive pooling layer that compresses frequency dimension down into 4 representative bins, preserving time resolution. This is then passed into an RNN module (GRU by default) in a bidirectional configuration to capture temporal dynamic and sequences in bird calls. An attention mechanism - comprised of single linear layer with Tanh activation, assigns importance to each time step, effectively filtering less informative frames. These embeddings are used by a Prototypical Network, which computes class prototypes from the support set and compares them to query embeddings using COSINE distance and temperature scaling (10.0), converting distances into log-probabilities for classification. Simultaneously, a Relation Network composed of fully connected layers (256 -> 128 -> 64 -> 1) takes concatenated embeddings of each query and prototype pair to compute similarity scores, optimized using mean squared error (MSE) loss. During evaluation, the model processes the test set over 100 episodes, extracting embeddings and producing final predictions using a weighted combination of prototypical probabilities (60%) and relation similarities (40%).

This hybrid approach achieves a classification accuracy of 97% on a 30-sample test set.

Differences from FSL1:
-> Class prototypes from the support set is compared to query embeddings using COSINE distance with temperature scaling (10.0), instead of EUCALIDEAN distance
-> Mantains temporal structure by pooling layer that compresses frequency dimension into 4 representative bins, preserving time, instead of pooling over both frequency and time and flattening into a single vector
-> Simple attention mechanism - comprised of single linear layer with Tanh activation to assign importance to each time step and filter out less useful frames

Differences from FS3:
-> Simplified attention mechanism - only one linear Tanh layer, compared to multihead temporal attention (see fs3)
-> Utilizes ensemble approach (prototypical + relation weighted predictions), not just prototypical network
-> Normalization is not learnable
-> Uses constant temperature scaling instead of temperature decay

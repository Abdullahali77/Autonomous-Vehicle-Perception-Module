class ResNetLSTM(nn.Module):
    """
    ResNet (last 2 layers fine-tuned) -> LSTM -> FC
    Always outputs sequence-level prediction for 2-frame input
    """

    def __init__(self, num_classes=3, lstm_hidden_dim=256, lstm_num_layers=2, dropout=0.3):
        super().__init__()

        # Load pre-trained ResNet50
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Freeze all ResNet layers first
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze last 2 layers (layer3 and layer4)
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        # Remove classification head, keep up to avgpool
        self.feature_extractor = nn.Sequential(
            *list(self.resnet.children())[:-1]
        )

        self.feature_dim = 2048

        # LSTM for sequence of 2 frames
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False
        )

        # FC classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, 2, 3, 224, 224) - Batch of 2-frame sequences

        Returns:
            output: (B, num_classes) - Sequence-level predictions
        """
        B, T, C, H, W = x.shape

        # Process all frames through ResNet
        x_flat = x.view(B * T, C, H, W)
        features_flat = self.feature_extractor(x_flat)
        features_flat = features_flat.view(B * T, self.feature_dim)
        features_seq = features_flat.view(B, T, self.feature_dim)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(features_seq)

        # Take last hidden state from last layer
        last_hidden = hidden[-1]

        # Classify
        output = self.classifier(last_hidden)

        return output

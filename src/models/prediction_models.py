import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')


class WirelessDataset(Dataset):
    """Dataset class for wireless network time series data"""

    def __init__(self,
                 channel_gains: np.ndarray,
                 traffic_demands: np.ndarray,
                 sequence_length: int = 10,
                 prediction_horizon: int = 1):
        """
        Initialize the dataset.

        Args:
            channel_gains: Historical channel gains (time_steps, n_users, n_rbs)
            traffic_demands: Historical traffic demands (time_steps, n_users)
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict ahead
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Flatten spatial dimensions for easier processing
        self.channel_gains = channel_gains  # (time_steps, n_users, n_rbs)
        self.traffic_demands = traffic_demands  # (time_steps, n_users)

        # Prepare sequences
        self.sequences = self._create_sequences()

    def _create_sequences(self):
        """Create input-output sequences for training"""
        sequences = []
        n_time_steps = self.channel_gains.shape[0]

        for i in range(n_time_steps - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            input_channels = self.channel_gains[i:i + self.sequence_length]
            input_traffic = self.traffic_demands[i:i + self.sequence_length]

            # Target (future values)
            target_channels = self.channel_gains[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
            target_traffic = self.traffic_demands[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]

            sequences.append({
                'input_channels': input_channels,
                'input_traffic': input_traffic,
                'target_channels': target_channels,
                'target_traffic': target_traffic
            })

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return (
            torch.FloatTensor(seq['input_channels']),
            torch.FloatTensor(seq['input_traffic']),
            torch.FloatTensor(seq['target_channels']),
            torch.FloatTensor(seq['target_traffic'])
        )


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class LSTMPredictor(nn.Module):
    """LSTM-based predictor for channel gains and traffic demands"""

    def __init__(self,
                 n_users: int,
                 n_rbs: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 prediction_horizon: int = 1):
        super().__init__()
        self.n_users = n_users
        self.n_rbs = n_rbs
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon

        # Input features: channel gains + traffic demands
        self.input_size = n_users * n_rbs + n_users

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Output projections
        self.channel_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_users * n_rbs * prediction_horizon)
        )

        self.traffic_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_users * prediction_horizon)
        )

    def forward(self, channel_gains, traffic_demands):
        """
        Forward pass.

        Args:
            channel_gains: (batch_size, seq_len, n_users, n_rbs)
            traffic_demands: (batch_size, seq_len, n_users)

        Returns:
            Predicted channel gains and traffic demands
        """
        batch_size, seq_len = channel_gains.shape[:2]

        # Flatten spatial dimensions
        channels_flat = channel_gains.view(batch_size, seq_len, -1)
        traffic_flat = traffic_demands.view(batch_size, seq_len, -1)

        # Concatenate features
        x = torch.cat([channels_flat, traffic_flat], dim=-1)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Use last output for prediction
        last_output = lstm_out[:, -1, :]

        # Generate predictions
        channel_pred = self.channel_predictor(last_output)
        traffic_pred = self.traffic_predictor(last_output)

        # Reshape predictions
        channel_pred = channel_pred.view(batch_size, self.prediction_horizon, self.n_users, self.n_rbs)
        traffic_pred = traffic_pred.view(batch_size, self.prediction_horizon, self.n_users)

        return channel_pred, traffic_pred


class TransformerPredictor(nn.Module):
    """Transformer-based predictor for channel gains and traffic demands"""

    def __init__(self,
                 n_users: int,
                 n_rbs: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 prediction_horizon: int = 1):
        super().__init__()
        self.n_users = n_users
        self.n_rbs = n_rbs
        self.d_model = d_model
        self.prediction_horizon = prediction_horizon

        # Input features: channel gains + traffic demands
        self.input_size = n_users * n_rbs + n_users

        # Input projection
        self.input_projection = nn.Linear(self.input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # Output projections
        self.channel_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_users * n_rbs * prediction_horizon)
        )

        self.traffic_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_users * prediction_horizon)
        )

    def forward(self, channel_gains, traffic_demands):
        """
        Forward pass.

        Args:
            channel_gains: (batch_size, seq_len, n_users, n_rbs)
            traffic_demands: (batch_size, seq_len, n_users)

        Returns:
            Predicted channel gains and traffic demands
        """
        batch_size, seq_len = channel_gains.shape[:2]

        # Flatten spatial dimensions
        channels_flat = channel_gains.view(batch_size, seq_len, -1)
        traffic_flat = traffic_demands.view(batch_size, seq_len, -1)

        # Concatenate features
        x = torch.cat([channels_flat, traffic_flat], dim=-1)

        # Input projection
        x = self.input_projection(x)

        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)

        # Transformer encoding
        encoded = self.transformer_encoder(x)

        # Use last output for prediction
        last_output = encoded[:, -1, :]

        # Generate predictions
        channel_pred = self.channel_predictor(last_output)
        traffic_pred = self.traffic_predictor(last_output)

        # Reshape predictions
        channel_pred = channel_pred.view(batch_size, self.prediction_horizon, self.n_users, self.n_rbs)
        traffic_pred = traffic_pred.view(batch_size, self.prediction_horizon, self.n_users)

        return channel_pred, traffic_pred


class WirelessPredictor:
    """Wrapper class for training and using wireless network predictors"""

    def __init__(self,
                 n_users: int,
                 n_rbs: int,
                 model_type: str = 'lstm',
                 sequence_length: int = 10,
                 prediction_horizon: int = 1,
                 **model_kwargs):
        """
        Initialize the predictor.

        Args:
            n_users: Number of users
            n_rbs: Number of resource blocks
            model_type: Type of model ('lstm' or 'transformer')
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict ahead
            **model_kwargs: Additional arguments for the model
        """
        self.n_users = n_users
        self.n_rbs = n_rbs
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon

        # Initialize model
        if model_type == 'lstm':
            self.model = LSTMPredictor(
                n_users=n_users,
                n_rbs=n_rbs,
                prediction_horizon=prediction_horizon,
                **model_kwargs
            )
        elif model_type == 'transformer':
            self.model = TransformerPredictor(
                n_users=n_users,
                n_rbs=n_rbs,
                prediction_horizon=prediction_horizon,
                **model_kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Training utilities
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.channel_scaler = StandardScaler()
        self.traffic_scaler = StandardScaler()
        self.is_trained = False

    def prepare_data(self,
                    channel_gains: np.ndarray,
                    traffic_demands: np.ndarray,
                    test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for training"""

        # Normalize data
        original_shape_channels = channel_gains.shape
        original_shape_traffic = traffic_demands.shape

        # Flatten for scaling
        channels_flat = channel_gains.reshape(-1, original_shape_channels[-1] * original_shape_channels[-2])
        traffic_flat = traffic_demands.reshape(-1, original_shape_traffic[-1])

        # Fit scalers and transform
        channels_scaled = self.channel_scaler.fit_transform(channels_flat)
        traffic_scaled = self.traffic_scaler.fit_transform(traffic_flat)

        # Reshape back
        channels_scaled = channels_scaled.reshape(original_shape_channels)
        traffic_scaled = traffic_scaled.reshape(original_shape_traffic)

        # Create dataset
        dataset = WirelessDataset(
            channels_scaled,
            traffic_scaled,
            self.sequence_length,
            self.prediction_horizon
        )

        # Split data
        train_size = int(len(dataset) * (1 - test_size))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, test_loader

    def train(self,
              train_loader: DataLoader,
              test_loader: DataLoader,
              epochs: int = 100,
              lr: float = 1e-3,
              patience: int = 10) -> Dict[str, List[float]]:
        """Train the model"""

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        # Loss function
        criterion = nn.MSELoss()

        train_losses = []
        test_losses = []
        best_loss = float('inf')
        patience_counter = 0

        print(f"Training {self.model_type} model...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                input_channels, input_traffic, target_channels, target_traffic = batch
                input_channels = input_channels.to(self.device)
                input_traffic = input_traffic.to(self.device)
                target_channels = target_channels.to(self.device)
                target_traffic = target_traffic.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                pred_channels, pred_traffic = self.model(input_channels, input_traffic)

                # Calculate loss
                channel_loss = criterion(pred_channels, target_channels)
                traffic_loss = criterion(pred_traffic, target_traffic)
                loss = channel_loss + traffic_loss

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    input_channels, input_traffic, target_channels, target_traffic = batch
                    input_channels = input_channels.to(self.device)
                    input_traffic = input_traffic.to(self.device)
                    target_channels = target_channels.to(self.device)
                    target_traffic = target_traffic.to(self.device)

                    pred_channels, pred_traffic = self.model(input_channels, input_traffic)

                    channel_loss = criterion(pred_channels, target_channels)
                    traffic_loss = criterion(pred_traffic, target_traffic)
                    loss = channel_loss + traffic_loss

                    test_loss += loss.item()

            # Average losses
            train_loss /= len(train_loader)
            test_loss /= len(test_loader)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            scheduler.step(test_loss)

            # Early stopping
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        self.is_trained = True

        return {'train_losses': train_losses, 'test_losses': test_losses}

    def predict(self,
                channel_history: np.ndarray,
                traffic_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()

        # Normalize input
        channels_flat = channel_history.reshape(-1, channel_history.shape[-1] * channel_history.shape[-2])
        traffic_flat = traffic_history.reshape(-1, traffic_history.shape[-1])

        channels_scaled = self.channel_scaler.transform(channels_flat)
        traffic_scaled = self.traffic_scaler.transform(traffic_flat)

        channels_scaled = channels_scaled.reshape(channel_history.shape)
        traffic_scaled = traffic_scaled.reshape(traffic_history.shape)

        # Convert to tensors
        channels_tensor = torch.FloatTensor(channels_scaled).unsqueeze(0).to(self.device)
        traffic_tensor = torch.FloatTensor(traffic_scaled).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_channels, pred_traffic = self.model(channels_tensor, traffic_tensor)

        # Convert back to numpy and denormalize
        pred_channels = pred_channels.cpu().numpy().squeeze(0)
        pred_traffic = pred_traffic.cpu().numpy().squeeze(0)

        # Denormalize
        for t in range(self.prediction_horizon):
            channels_flat = pred_channels[t].reshape(-1, pred_channels.shape[-1] * pred_channels.shape[-2])
            traffic_flat = pred_traffic[t].reshape(-1, pred_traffic.shape[-1])

            channels_denorm = self.channel_scaler.inverse_transform(channels_flat)
            traffic_denorm = self.traffic_scaler.inverse_transform(traffic_flat)

            pred_channels[t] = channels_denorm.reshape(pred_channels[t].shape)
            pred_traffic[t] = traffic_denorm.reshape(pred_traffic[t].shape)

        return pred_channels, pred_traffic

    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'n_users': self.n_users,
            'n_rbs': self.n_rbs,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'channel_scaler': self.channel_scaler,
            'traffic_scaler': self.traffic_scaler
        }, filepath)

    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.channel_scaler = checkpoint['channel_scaler']
        self.traffic_scaler = checkpoint['traffic_scaler']
        self.is_trained = True


if __name__ == "__main__":
    # Example usage
    from src.data.wireless_generator import WirelessDataGenerator

    # Generate sample data
    generator = WirelessDataGenerator(n_users=5, n_rbs=10)
    dataset = generator.generate_dataset(n_time_slots=1000)

    # Initialize predictor
    predictor = WirelessPredictor(
        n_users=5,
        n_rbs=10,
        model_type='lstm',  # or 'transformer'
        sequence_length=10,
        prediction_horizon=1
    )

    # Prepare data
    train_loader, test_loader = predictor.prepare_data(
        dataset['channel_gains'],
        dataset['traffic_demands']
    )

    # Train model
    history = predictor.train(train_loader, test_loader, epochs=50)

    # Make predictions
    sample_channels = dataset['channel_gains'][-10:]
    sample_traffic = dataset['traffic_demands'][-10:]

    pred_channels, pred_traffic = predictor.predict(sample_channels, sample_traffic)
    print(f"Predicted channel gains shape: {pred_channels.shape}")
    print(f"Predicted traffic demands shape: {pred_traffic.shape}")
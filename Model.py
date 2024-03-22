import torch


class MidiTranscriptionModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hp = params
        self.device = params['device']

        self.FC = torch.nn.Linear(params['input_size'], params['output_size'])

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=params['learning_rate'])
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.FC(x)
        return x

    def training_step(self, batch):
        # switch to train mode
        self.train()
        # Reset gradients
        self.optimizer.zero_grad()
        audio, midi = batch[0], batch[1]
        audio.to(self.device)
        midi.to(self.device)

        # Prediction
        pred_midi = self.forward(audio)
        loss = self.loss_fn(pred_midi, midi)
        # Backpropagation
        loss.backward()
        # Update parameters
        self.optimizer.step()

        return loss.item()

    def validation_step(self, batch):
        # Set model to evaluation mode
        self.eval()
        with torch.no_grad():
            audio, midi = batch[0], batch[1]
            audio.to(self.device)
            midi.to(self.device)
            # Prediction
            pred_midi = self.forward(audio)
            loss = self.loss_fn(pred_midi, midi)

        return loss.item()

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

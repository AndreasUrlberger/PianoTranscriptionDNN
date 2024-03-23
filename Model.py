import torch


class MidiTranscriptionModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.hp = params
        self.device = params['device']

        self.FC1 = torch.nn.Linear(params['input_size'], params['hidden_size_1'])
        self.FC2 = torch.nn.Linear(params['hidden_size_1'], params['hidden_size_2'])
        self.FC3 = torch.nn.Linear(params['hidden_size_2'], params['hidden_size_3'])
        self.FC4 = torch.nn.Linear(params['hidden_size_3'], params['hidden_size_4'])
        self.FC5 = torch.nn.Linear(params['hidden_size_4'], params['output_size'])

        self.activation = torch.nn.LeakyReLU()

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=params['learning_rate'])
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.activation(self.FC1(x))
        x = self.activation(self.FC2(x))
        x = self.activation(self.FC3(x))
        x = self.activation(self.FC4(x))
        x = self.FC5(x)
        return x
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
            x = torch.sigmoid(x)
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

    def save_state(self, path):
        """
        Save model state to the given path. Conventionally the
        path should end with "*.pt".

        Later to restore:
            model.load_state_dict(torch.load(filepath))
            model.eval()

        Inputs:
        - path: path string
        """
        torch.save(self.state_dict(), path)
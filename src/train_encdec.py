from citylearn.agents.base import Agent as RandomAgent
from citylearn.citylearn import CityLearnEnv
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn.functional as F

from agents.autoencoder import SACEncoder

root_directory = Path("data/citylearn_challenge_2023_phase_1")
schema_path = root_directory / "schema.json"
env = CityLearnEnv(schema=schema_path, root_directory=root_directory, central_agent=True)
random_model = RandomAgent(env)
encoder = SACEncoder(observation_space_dim=77, output_dim=30, hidden_dim=50)

# ---------

# TODO: Add all data to train on to get the encoder as we did in NN
# best loss: ~ 70k xd

# -------

best_loss = float('inf')  # Initialize to a very high value
best_model_path = "best_sac_encoder.pth"

optimizer = optim.Adam(encoder.parameters(), lr=0.00005)
for epoch in range (100):
    observations = env.reset()
    epoch_loss = 0

    while not env.done:
        
        observations_decoded = encoder(observations)
        observations_tensor = torch.tensor(observations, dtype=torch.float32)

        # Calculate loss
        loss = F.mse_loss(observations_decoded.squeeze(), observations_tensor.squeeze())

        #print(observations_tensor.squeeze(), observations_decoded_tensor.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()


        actions = random_model.predict(observations)
        observations, _, _, done = env.step(actions)

    print(f"Epoch {epoch + 1}: Loss = {epoch_loss:.4f}")

    # Save the model if it's the best one so far
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(encoder.state_dict(), best_model_path)
        print(f"New best model saved with loss {best_loss:.4f}")


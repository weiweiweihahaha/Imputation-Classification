import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


def _columnwise_mean_fill(values):
    filled = np.asarray(values, dtype=float).copy()
    valid_counts = np.sum(~np.isnan(filled), axis=0)
    column_sums = np.nansum(filled, axis=0)
    column_means = np.divide(
        column_sums,
        valid_counts,
        out=np.zeros_like(column_sums, dtype=float),
        where=valid_counts > 0,
    )
    column_means = np.where(np.isfinite(column_means), column_means, 0.0)
    missing_mask = np.isnan(filled)
    if missing_mask.any():
        filled[missing_mask] = np.take(column_means, np.where(missing_mask)[1])
    return filled


def _masked_mse_loss(prediction, target, observed_mask):
    squared_error = (prediction - target) ** 2
    masked_error = squared_error * observed_mask
    observed_count = max(float(observed_mask.sum()), 1.0)
    return float(masked_error.sum() / observed_count)


def _coerce_numeric_array(missing_data):
    if isinstance(missing_data, pd.DataFrame):
        frame = missing_data.copy()
    else:
        frame = pd.DataFrame(np.asarray(missing_data))
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.to_numpy(dtype=float, copy=True)


def _torch_backend_available():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception:
        return None
    return {
        "torch": torch,
        "nn": nn,
        "optim": optim,
    }


def _run_torch_mida(scaled_data, observed_mask, *, theta, num_epochs, dropout_ratio, batch_size):
    backend = _torch_backend_available()
    if backend is None:
        raise RuntimeError("PyTorch is unavailable")

    torch = backend["torch"]
    nn = backend["nn"]
    optim = backend["optim"]

    _, cols = scaled_data.shape
    train_data = torch.tensor(scaled_data.tolist(), dtype=torch.float32)
    observed_mask_tensor = torch.tensor(observed_mask.astype(np.float32).tolist(), dtype=torch.float32)
    train_dataset = torch.utils.data.TensorDataset(train_data, observed_mask_tensor)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=min(batch_size, max(len(train_data), 1)),
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Autoencoder(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.drop_out = nn.Dropout(p=dropout_ratio)
            self.encoder = nn.Sequential(
                nn.Linear(dim + theta * 0, dim + theta * 1),
                nn.Tanh(),
                nn.Linear(dim + theta * 1, dim + theta * 2),
                nn.Tanh(),
                nn.Linear(dim + theta * 2, dim + theta * 3),
            )
            self.decoder = nn.Sequential(
                nn.Linear(dim + theta * 3, dim + theta * 2),
                nn.Tanh(),
                nn.Linear(dim + theta * 2, dim + theta * 1),
                nn.Tanh(),
                nn.Linear(dim + theta * 1, dim + theta * 0),
            )

        def forward(self, x):
            x = x.view(-1, self.dim)
            x_missed = self.drop_out(x)
            z = self.encoder(x_missed)
            out = self.decoder(z)
            return out.view(-1, self.dim)

    model = Autoencoder(dim=cols).to(device)
    optimizer = optim.SGD(model.parameters(), momentum=0.99, lr=0.01, nesterov=True)

    for _ in range(num_epochs):
        early_stop = False
        for batch_data, batch_mask in train_loader:
            batch_data = batch_data.to(device)
            batch_mask = batch_mask.to(device)
            reconst_data = model(batch_data)
            squared_error = (reconst_data - batch_data) ** 2
            masked_error = squared_error * batch_mask
            observed_count = torch.clamp(batch_mask.sum(), min=1.0)
            cost = masked_error.sum() / observed_count
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            if cost.item() < 1e-06:
                early_stop = True
                break
        if early_stop:
            break

    model.eval()
    with torch.no_grad():
        return model(train_data.to(device)).cpu().numpy()


def _run_numpy_mida(scaled_data, observed_mask, *, theta, num_epochs, dropout_ratio, batch_size):
    rng = np.random.default_rng(0)
    rows, cols = scaled_data.shape
    layer_sizes = [cols, cols + theta, cols + theta * 2, cols + theta * 3, cols + theta * 2, cols + theta, cols]
    weights = [
        rng.normal(0.0, 0.05, size=(layer_sizes[idx], layer_sizes[idx + 1]))
        for idx in range(len(layer_sizes) - 1)
    ]
    biases = [np.zeros((1, layer_sizes[idx + 1]), dtype=float) for idx in range(len(layer_sizes) - 1)]
    velocity_w = [np.zeros_like(weight) for weight in weights]
    velocity_b = [np.zeros_like(bias) for bias in biases]
    learning_rate = 0.01
    momentum = 0.99
    keep_prob = max(1.0 - dropout_ratio, 1e-6)
    linear_layers = {2, 5}

    if rows == 0:
        return scaled_data.copy()

    for _ in range(num_epochs):
        order = rng.permutation(rows)
        early_stop = False
        for start in range(0, rows, min(batch_size, rows)):
            batch_idx = order[start:start + min(batch_size, rows)]
            batch_data = scaled_data[batch_idx]
            batch_mask = observed_mask[batch_idx].astype(float)
            dropout_mask = (rng.random(batch_data.shape) < keep_prob).astype(float) / keep_prob

            activations = [batch_data]
            preactivations = []

            x = batch_data * dropout_mask
            activations.append(x)
            for layer_idx, (weight, bias) in enumerate(zip(weights, biases)):
                z = activations[-1] @ weight + bias
                preactivations.append(z)
                if layer_idx in linear_layers:
                    a = z
                else:
                    a = np.tanh(z)
                activations.append(a)

            prediction = activations[-1]
            loss = _masked_mse_loss(prediction, batch_data, batch_mask)
            if loss < 1e-06:
                early_stop = True
                break

            observed_count = max(batch_mask.sum(), 1.0)
            delta = 2.0 * (prediction - batch_data) * batch_mask / observed_count
            grad_w = [None] * len(weights)
            grad_b = [None] * len(biases)

            for layer_idx in range(len(weights) - 1, -1, -1):
                if layer_idx not in linear_layers:
                    delta = delta * (1.0 - np.tanh(preactivations[layer_idx]) ** 2)
                grad_w[layer_idx] = activations[layer_idx + 1].T @ delta
                grad_b[layer_idx] = delta.sum(axis=0, keepdims=True)
                if layer_idx > 0:
                    delta = delta @ weights[layer_idx].T
                else:
                    delta = (delta @ weights[layer_idx].T) * dropout_mask

            for layer_idx in range(len(weights)):
                velocity_w[layer_idx] = momentum * velocity_w[layer_idx] - learning_rate * grad_w[layer_idx]
                velocity_b[layer_idx] = momentum * velocity_b[layer_idx] - learning_rate * grad_b[layer_idx]
                weights[layer_idx] += velocity_w[layer_idx]
                biases[layer_idx] += velocity_b[layer_idx]

        if early_stop:
            break

    x = scaled_data
    for layer_idx, (weight, bias) in enumerate(zip(weights, biases)):
        x = x @ weight + bias
        if layer_idx not in linear_layers:
            x = np.tanh(x)
    return x


def MIDA(missing_data):
    theta = 7
    num_epochs = 20
    dropout_ratio = 0.5
    batch_size = 4  # not in the paper

    X = _coerce_numeric_array(missing_data)
    _, cols = X.shape
    observed_mask = ~np.isnan(X)

    initialized = _columnwise_mean_fill(X)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(initialized)
    try:
        filled_data = _run_torch_mida(
            scaled_data,
            observed_mask,
            theta=theta,
            num_epochs=num_epochs,
            dropout_ratio=dropout_ratio,
            batch_size=batch_size,
        )
    except Exception:
        filled_data = _run_numpy_mida(
            scaled_data,
            observed_mask,
            theta=theta,
            num_epochs=num_epochs,
            dropout_ratio=dropout_ratio,
            batch_size=batch_size,
        )
    filled = scaler.inverse_transform(filled_data)

    imputed = np.asarray(X, dtype=float).copy()
    missing_mask = np.isnan(imputed)
    imputed[missing_mask] = filled[missing_mask]

    return imputed

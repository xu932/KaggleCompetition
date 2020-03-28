import numpy as np
import time

import torch


def data_augmentation_shift(features, labels, transformation, **kwargs):
    def transform(data):
        return transformation(data.reshape(kwargs['shape']), kwargs['shift'])
    features_shifted = np.apply_along_axis(transform, 1, features)
    labels_shifted = labels.copy()
    return features_shifted, labels_shifted


def data_augmentation_rotate(features, labels, transformation, **kwargs):
    def transform(data):
        return transformation(data.reshape(kwargs['shape']), kwargs['angle'], reshape=kwargs['reshape'] if 'reshape' in kwargs else False)
    features_shifted_ck = np.apply_along_axis(transform, 1, features)
    kwargs['angle'] = -kwargs['angle']
    features_shifted_cck = np.apply_along_axis(transform, 1, features)
    labels_shifted = np.r_[labels.copy(), labels.copy()]
    return np.r_[features_shifted_ck, features_shifted_cck], labels_shifted


def validate_model(model, loss, val_loader):
    correct = 0
    total = 0
    total_loss = 0.0
    model.eval()
    for features, labels in val_loader:
        # Forward propagation
        outputs = model(features)
        compute_loss = loss(outputs, labels)

        # Get predictions from the maximum value
        predicted = torch.max(outputs.data, 1)[1]

        # Total number of labels
        total += len(labels)
        correct += (predicted == labels).sum()
        total_loss += compute_loss.data
    return total_loss, float(correct) / total


def train_and_validate_model(model, optimizer, loss, train_loader, val_loader, num_epochs, log=0):
    history = {'loss': [], 'accu': []}

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        for features, labels in train_loader:
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(features)

            # Calculate the loss for current epoch
            compute_loss = loss(outputs, labels)

            # Calculating gradients
            compute_loss.backward()

            # Update parameters
            optimizer.step()
        end_time = time.time()
        val_loss, val_accu = validate_model(model, loss, val_loader)
        history['loss'].append(val_loss)
        history['accu'].append(val_accu)

        if val_loss < model.min_loss:
            print(f'record: epoch {epoch + 1} time: {end_time - start_time:.2f} loss: {val_loss:.4f} accu: {val_accu:.4f}')
            model.record_state(val_loss)
        elif log > 0 and epoch % log == 0:
            print(f'        epoch {epoch + 1} time: {end_time - start_time:.2f} loss: {val_loss:.4f} accu: {val_accu:.4f}')

    return history


def test_model(model, data_loader):
    predictions = []
    model.eval()
    for feature in data_loader:
        # Forward propagation
        output = model(feature[0])
        predictions += (torch.max(output.data, 1)[1]).tolist()

    return predictions

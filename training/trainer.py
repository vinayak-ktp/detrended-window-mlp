import torch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def train(model, train_dl, val_dl, criterion, optimizer, device, num_epochs, save_path, patience=10):
    history = {
        "train_loss": [],
        "val_loss": []
    }
    best_val_loss = float('inf')
    wait = 0

    for epoch in range(1, num_epochs+1):
        epoch_train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device)
        epoch_val_loss = evaluate(model, val_dl, criterion, device)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        print(f"epoch {epoch}/{num_epochs}   train={epoch_train_loss:.4}   val={epoch_val_loss:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stop at epoch {epoch} (patience={patience})")
                break

    model.load_state_dict(torch.load(save_path, weights_only=True))
    return history

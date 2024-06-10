import torch

def train_one_epoch(model, 
                    train_dl,
                    loss_fn,
                    optimizer,
                    device):
    model.train()
    model.to(device)

    train_loss = 0
    counter = 0 

    for i, (x, y) in enumerate(train_dl):
        counter += 1
        optimizer.zero_grad()

        batch_size, seq_len = x.shape
        x = x.to(device)
        labels = y.to(device)

        outputs = model(x)

        outputs = outputs.view(batch_size * seq_len, -1)
        labels = labels.view(batch_size * seq_len)
        # print(f'y_pred shape : {outputs.shape}')
        # print(f'labels shape : {labels.shape}')

        loss = loss_fn(outputs, labels)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss = train_loss/len(train_dl)

    return train_loss


def test_one_epoch(model, 
                  val_dl,
                  loss_fn,
                  device): 
    
    model.eval()
    model.to(device)

    val_loss = 0
    counter = 0 

    with torch.inference_mode():
        for i, (x, y) in enumerate(val_dl):
            counter += 1

            batch_size, seq_len = x.shape
            x = x.to(device)
            labels = y.to(device)

            outputs = model(x)

            outputs = outputs.view(batch_size * seq_len, -1)
            labels = labels.view(batch_size * seq_len)

            loss = loss_fn(outputs, labels)

            val_loss += loss.item()

    val_loss = val_loss/len(val_dl)

    return val_loss


# import wandb
import wandb

def train(model,
          train_dl,
          val_dl,
          loss_fn,
          optimizer,
          device,
          epochs,
          save_wandb=None,
          project_name=None,
          experiment_name=None,
          hyper_param_config=None):
    
# ------------------------------------------------------------------------------------------------
    if save_wandb : 
        try:
            wandb.init(
            # Set the project where this run will be logged
            project=project_name, 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"exp_{experiment_name}", 
            # Track hyperparameters and run metadata
            config=hyper_param_config
            )
        except:
            print('Can not initialize to wandb')
# --------------------------------------------------------------------------------

    results = {
        'train_loss':[],
        'val_loss':[]
        }

    for i in range(epochs):

        train_loss = train_one_epoch(model=model,
                                     train_dl=train_dl,
                                     loss_fn=loss_fn,
                                     optimizer=optimizer,
                                     device=device)

        val_loss = test_one_epoch(model=model,
                                   val_dl=val_dl,
                                   loss_fn=loss_fn,
                                   device=device)
# --------------------------------------------------------------------------------
        if save_wandb : 
            wandb.log({"train_loss": train_loss, 
                        "val_loss": val_loss
                        })
            
# --------------------------------------------------------------------------------

        print(f'epoch {i+1}/{epochs} | '
              f'train_loss:{train_loss:.2f} | '
              f'val_loss:{val_loss:.2f}'
              )
    
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)    
# --------------------------------------------------------------------------------
    if save_wandb :
        wandb.finish()
# --------------------------------------------------------------------------------

    return results


    



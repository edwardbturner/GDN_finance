import torch
from torch.optim import Adam



def custom_loss(y_hat, y, gamma, delta):
    loss = delta*torch.mean(-torch.tanh(gamma*y_hat)*y) + (1-delta)*torch.mean((y_hat-y)**2)
    return loss

def pnl_function(y_hat, y):
    pnl = torch.mean(torch.sign(y_hat)*y)
    return pnl



def rolling_train_test(model, data, GDN_epochs, DDPM_epochs, gamma, delta, GDN_lr, DDPM_lr, DDPM_lb, training_lb, GCN):    
    train_losses, train_pnls, test_losses, test_pnls, test_ys, all_batches = [], [], [], [], [], []
    T = data.snapshot_count
    
    for t in range(training_lb, T):
        print(f' ----- On train/test for day {t} ----- ')
        
        if GCN==True:
            b = T
        else:
            b = DDPM_lb+training_lb
        
        if t<b:
            model_passing_outputs = model_passing(model, data, GDN_epochs, gamma, delta, GDN_lr, training_lb,
                                                  t, False, GCN)
            all_batches.append(model_passing_outputs[0])
            train_losses.append(model_passing_outputs[1])
            train_pnls.append(model_passing_outputs[2])
            test_losses.append(model_passing_outputs[3])
            test_pnls.append(model_passing_outputs[4])
            test_ys.append(model_passing_outputs[5])
            print(f'Current test pnl = {model_passing_outputs[4]}')
            
        else:
            model.train()
            previous_batches = np.array(all_batches[-DDPM_lb:])
            previous_batches = torch.tensor(previous_batches)
            
            DDPMoptim = Adam(model._DDPM.parameters(), lr=DDPM_lr)
            mse = nn.MSELoss()
            num_steps = model.num_steps
            
            for epoch in range(DDPM_epochs):
                epoch_loss = 0.0
                
                for batch in previous_batches:
                    x0 = batch                    
                    n = len(x0)
                    epsilon = torch.randn_like(x0)
                    time = torch.randint(0, num_steps, (n,))
                    noisy_imgs = model._DDPM(x0, time, epsilon)
                    epsilon_theta = model._DDPM.backward(noisy_imgs, time.reshape(n, -1))
                    DDPM_loss = mse(epsilon_theta, epsilon)
                    
                    DDPMoptim.zero_grad()
                    DDPM_loss.backward()
                    DDPMoptim.step()
                    
                    epoch_loss += DDPM_loss.item()

                print(f"Loss at DDPM epoch {epoch + 1}: {epoch_loss/DDPM_lb:.6f}")
            
            model_passing_outputs = model_passing(model, data, GDN_epochs, gamma, delta, GDN_lr, training_lb,
                                                  t, True, GCN)
            all_batches.append(model_passing_outputs[0])
            train_losses.append(model_passing_outputs[1])
            train_pnls.append(model_passing_outputs[2])
            test_losses.append(model_passing_outputs[3])
            test_pnls.append(model_passing_outputs[4])
            test_ys.append(model_passing_outputs[5])
            print(f'Current test pnl = {model_passing_outputs[4]:.6f}')
            
    return train_losses, train_pnls, test_losses, test_pnls, np.array(test_ys)



def model_passing(model, data, GDN_epochs, gamma, delta, GDN_lr, training_lb, t, use_DDRM, GCN):    
    current_batch, train_t_losses, train_t_pnls = [], [], []
    
    params = list(model._GCN.parameters()) + list(model._linear.parameters())
    optim = Adam(params, lr=GDN_lr)
    
    model.eval()
    x_data = []
    for i in range(t-training_lb, t+1):
        m = data[i].x
        
        if use_DDRM==True:
            print(f'Sampling for day {i-t+training_lb+1}/{training_lb+1} which is day {i}')
            std_dev = m.std()
            
            # Below we standardise before denoising the sample via the DDRM and then scale back
            # up post-DDRM
            m = m/std_dev            
            x_data.append((model._DDRM(m, 0.05).detach().numpy())*std_dev.item())
        else:
            x_data.append(m.detach().numpy())
    
    x_data = np.array(x_data)
    x_data = torch.tensor(x_data)
    
    
    model.train()
    for epoch in range(GDN_epochs):
        train_epoch_loss = 0
        train_pnl_loss = 0
        
        for i, t_ in enumerate(range(t-training_lb, t)):
            snapshot = data[t_]
            y_hat = model(x_data[i], snapshot.edge_index, snapshot.edge_type, snapshot.edge_attr, False)
            train_loss = custom_loss(y_hat, snapshot.y, gamma, delta)
            train_pnl_loss += pnl_function(y_hat, snapshot.y).item()
                
            optim.zero_grad()
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()
        
        print(f'On epoch {epoch+1} of day {t} training, loss = {train_epoch_loss/training_lb:.8f}')

        # We divide by training_lb so can compare fairly for varying training_lb size
        train_t_losses.append(train_epoch_loss/training_lb)
        train_t_pnls.append(train_pnl_loss/training_lb)
    
    
    model.eval()
    if GCN==False:
        for i, t_ in enumerate(range(t-training_lb, t+1)):
            snapshot = data[t_]
            x_0_hat = model(x_data[i], snapshot.edge_index, snapshot.edge_type, snapshot.edge_attr, True)
            x_0_hat = x_0_hat/x_0_hat.std()  # So all of the DDPM training samples are standardised
            x_0_hat = x_0_hat[None, :]
            current_batch.append(x_0_hat.detach().numpy())
    
    snapshot = data[t]
    y_hat_t = model(x_data[-1], snapshot.edge_index, snapshot.edge_type, snapshot.edge_attr, False)
    test_t_loss = custom_loss(y_hat_t, snapshot.y, gamma, delta).item()
    test_t_pnl = pnl_function(y_hat_t, snapshot.y).item()
    
    return current_batch, train_t_losses, train_t_pnls, test_t_loss, test_t_pnl, (y_hat_t.detach().numpy(), snapshot.y.detach().numpy())

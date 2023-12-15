import argparse
import torch
import os
import random
import gc
import time
from datetime import datetime
from torch.utils.data import DataLoader
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.fine_tune_data import ShapE_Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# setup argparse args
# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_folder', type=str, 
                    help='path to data folder', required=True)
parser.add_argument('-o', '--output_folder', type=str, 
                    help='path to output folder', required=True)
parser.add_argument('-b', '--batch_size', type=int, default=6, 
                    help='batch size')
parser.add_argument('-g', '--grad_accumulation_steps', type=int, default=11, 
                    help='gradient accumulation steps')
parser.add_argument('--epochs', type=int, default=100, 
                    help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=7e-5*0.4, 
                    help='learning rate')
parser.add_argument('--epoch_test_freq', type=int, default=5, 
                    help='frequency (in epochs) in which to test model')
parser.add_argument('--num_timesteps', type=int, default=1024, 
                    help='number of timesteps (1024 in paper)')
parser.add_argument('--num_control_layers', type=int, default=24, 
                    help='number of control residual blocks in transformer backbone')
parser.add_argument('--freeze_outproj', action='store_true', 
                    help='whether to also freeze the output projection layer')
parser.add_argument('--reverse', action='store_true', 
                    help='whether to train the first layers')
parser.add_argument('--rescaled_mse', action='store_true', 
                    help='whether to use wandb')
parser.add_argument('--full_backbone', action='store_true', 
                    help='whether to also train layers after the control layers')
parser.add_argument('--cond_drop_prob', type=float, default=0.5, 
                    help='chance for model to ignore text condition during training')

# main function
def train_controlnet(args):
    data_path = args.data_folder
    output_path = args.output_folder
    batch_size = args.batch_size
    num_epochs = args.epochs
    num_timesteps = args.num_timesteps
    test_freq = args.epoch_test_freq
    grad_acc_steps = args.grad_accumulation_steps
    num_control_layers = args.num_control_layers
    learning_rate = args.learning_rate
    reverse = args.reverse
    full_backbone = args.full_backbone
    cond_drop_prob = args.cond_drop_prob
    no_one_conv = True
    cross_mode = True
    full_backbone_hard = True
    train_outproj = not args.freeze_outproj


    # load model
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    model.wrapped.backbone.make_ctrl_layers(num_control_layers, 
                                            reverse=reverse,
                                            cross_mode=cross_mode,
                                            conditional=True,
                                            no_one_conv=no_one_conv)
    model.wrapped.set_up_controlnet_cond()
    # this next line should force the model to rely more on the condition
    
    model.wrapped.cond_drop_prob = cond_drop_prob
    diffusion = diffusion_from_config(load_config('diffusion'))
    if args.rescaled_mse:
        diffusion.loss_type = 'rescaled_mse'
    
    # prepare model for training
    model.prepare_ctrlnet_for_training(out_proj=train_outproj, 
                                       full_backbone=full_backbone,
                                       full_backbone_hard=full_backbone_hard)
    print(f"************* Trainable Param Status *************")
    model.print_parameter_status()

    # if model file exists in output folder, load it
    first_epoch = 0
    if os.path.exists(os.path.join(output_path, 'model_final.pt')):
        print("loading model from output folder")
        model.load_state_dict(torch.load(os.path.join(output_path, 'model_final.pt')))
        # get last epoch by looking at file prefixes
        render_path = os.path.join(output_path, 'rendered_samples')
        first_epoch = max([int(f.split('_')[0]) for f in os.listdir(render_path) if f.endswith('.mp4')])
        print(f"starting from epoch: {first_epoch + 1}")

    # set up datasets
    print(f"====== Setting up training dataset ======")
    train_dataset = ShapE_Dataset(os.path.join(data_path, 'train'), load_gray=True)
    train_dataloader = DataLoader(
                train_dataset, 
                batch_size=batch_size,
                shuffle=True, 
                num_workers=0
            )
    print(f"====== Setting up test-unseen dataset ======")
    test_dataset = ShapE_Dataset(os.path.join(data_path, 'test'), load_gray=True)
    test_dataloader = DataLoader(
                test_dataset, 
                batch_size=1,
                shuffle=False, 
                num_workers=0
            )

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # create output dir if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # fine tune
    for epoch in range(first_epoch, first_epoch + num_epochs):
        print(f"====== epoch: {epoch + 1} ======")
        start_time = time.time()
        loss_for_epoch = 0
        grad_acc_step = 0
        while grad_acc_step < grad_acc_steps:
            for batch in train_dataloader:
                cond = batch['latent_gray'].to(device).detach()
                x_start = batch['latent'].to(device).detach()
                prompts = batch['prompt']
                latents_batch_size = x_start.shape[0]
                model_kwargs = dict(x_start=prompts, cond=cond)
                timesteps = torch.tensor(random.sample(range(num_timesteps), 
                                                       latents_batch_size), 
                                                       dtype=torch.int32).to(device).detach()
                losses = diffusion.training_losses(
                    model=model,
                    x_start=x_start,
                    t=timesteps,
                    model_kwargs=model_kwargs,
                )
                loss = (losses['loss'] / grad_acc_steps)
                loss.backward()
                loss_for_epoch += loss.item()
                grad_acc_step += 1
        
                if grad_acc_step % grad_acc_steps == 0:
                    optimizer.step()
                    model.zero_grad()
        
                del x_start
                del cond
                del prompts
                del timesteps
                torch.cuda.empty_cache()
                gc.collect()
        
        # logging
        elapsed_time = time.time() - start_time
        print(f'elapsed time: {elapsed_time // 60:.0f}:{elapsed_time % 60:.0f} minutes')
        print(f"loss for epoch: {loss_for_epoch}")

                
        # test model
        if ((epoch + 1) % test_freq == 0) or (epoch == 0):
            print(f"====== testing and saving model at epoch: {epoch + 1} ======")
            random.seed(10)
            acc_test_loss = 0
            with torch.no_grad():
                for sample in test_dataloader:
                    if sample == -1:
                        print(f"issue loading validation sample")
                        break
                    cond = batch['latent_gray'].to(device).detach()
                    x_start = batch['latent'].to(device).detach()
                    prompts = sample['prompt'] * batch_size
                    model_kwargs = dict(texts=prompts, cond=cond)
                    timesteps = torch.tensor(random.sample(range(num_timesteps), 
                                                           batch_size), 
                                                           dtype=torch.int32).to(device).detach()
                    test_losses = diffusion.training_losses(
                        model=model,
                        x_start=x_start,
                        t=timesteps,
                        model_kwargs=model_kwargs,
                    )
                    test_loss = (test_losses['loss'] / grad_acc_steps)
                    acc_test_loss += test_loss.item()
                
                    # clear cache
                    del sample
                    torch.cuda.empty_cache()
                    gc.collect()

                # logging
                print(f"test loss for epoch: {acc_test_loss}")

            # save model
            torch.save(model.state_dict(), os.path.join(output_path, f'model_final.pt'))
            torch.save(model.state_dict(), os.path.join(output_path, f'model_epoch_{epoch}.pt'))
            random.seed(str(datetime.now()))
            

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parser.parse_args()
    train_controlnet(args)
    
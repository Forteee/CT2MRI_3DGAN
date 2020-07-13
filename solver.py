from model import StarGenerator3D
from model import StarDiscriminator3D
#from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import matplotlib.pyplot as plt


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, dataloader, config):
        """Initialize configurations."""

        # Data loader.
        self.dataloader = dataloader

        # Model configurations.
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs
        
        self.criterionL1 = torch.nn.L1Loss()

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = StarGenerator3D()
        self.D = StarDiscriminator3D()
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

        
    def save_pictures(self, real_CT, real_MR, fake_MR, sample_path):
        real_A = real_CT.cpu().numpy()[0, 0]
        real_B = real_MR.cpu().numpy()[0, 0]
        fake_B = fake_MR.cpu().numpy()[0, 0]
        
        fig, axs = plt.subplots(3, 12, sharex=True, figsize=(120, 30))
        plt.subplots_adjust(hspace =0)
        for i in range(0, 12):
            axs[0, i].imshow(real_A[:, :, i], cmap=plt.cm.gray)
            axs[0, i].axis('off')
            #axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    
            axs[1, i].imshow(real_B[:, :, i], cmap=plt.cm.gray)
            axs[1, i].axis('off')
            #axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    
            axs[2, i].imshow(fake_B[:, :, i], cmap=plt.cm.gray)
            axs[2, i].axis('off')
        plt.tight_layout()
        plt.savefig(sample_path)
        plt.close()

    def train(self):
        # Set data loader.
        data_loader = self.dataloader
        data_iter = iter(data_loader)

        # Learning rate cache for decaying.
        g_lr = self.g_lr  # 0.0001
        d_lr = self.d_lr  # 0.0001

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            try:
                input_data = next(data_iter)
            except:
                data_iter = iter(data_loader)
                input_data = next(data_iter)
            
            real_CT = input_data['CT'].to(self.device).float()
            real_MR = input_data['MR'].to(self.device).float()
            image_paths = input_data['CT_path']
          
            # Train the discriminator 

            # Compute loss with real images.
            out_src = self.D(real_MR)
            d_loss_real = - torch.mean(out_src)
            # Compute loss with fake images.
            fake_MR = self.G(real_CT)
            out_src = self.D(fake_MR.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(real_MR.size(0), 1, 1, 1, 1).to(self.device)
            x_hat = (alpha * real_MR.data + (1 - alpha) * real_CT.data).requires_grad_(True)
            out_src = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_gp'] = d_loss_gp.item()
            
            # train the generator                                

            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                fake_MR = self.G(real_CT)
                out_src = self.D(fake_MR)
                g_loss_fake = - torch.mean(out_src)
                
                # L1 loss
                g_loss_L1 = self.criterionL1(fake_MR, real_MR) * 100

                # Backward and optimize.
                g_loss = g_loss_fake + g_loss_L1
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_L1'] = g_loss_L1.item()

            #  Miscellaneous                                    

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    patient_ID = image_paths[0].split('/')[-1].split('.')[0]
            
                    sample_path = os.path.join(self.sample_dir, '{0}-images_{1}.jpg'.format(i+1, patient_ID))
                    self.save_pictures(real_CT, real_MR, fake_MR, sample_path)
                    #save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    

    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        data_loader = self.dataloader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
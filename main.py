import os
import torch
from tqdm import tqdm

from models import ResChanAttnMultiScaleAttn, make_model
from utils import AverageMeter, tup2list

class Args:
    # Pseudocode
    def __init__(self):
        self.enc_cpt_path = 'path/enc_cpt'
        self.exp_prefix = 'res_channel_attn_multi_scale'
        self.G_n_voxels = 3540
        self.im_res = 112
        self.G_n_chan_dec_output = 3
        self.gn = 2
        self.loss_weights = [1,1,1]
        self.reg_loss = 1
        self.n_epochs = 350
        
class Trainer:
    def __init__(
        self,
        dec: torch.nn.Module,
        enc: torch.nn.Module,
        data_loaders,
        optimizer: torch.optim.Optimizer,
        critions,
        reg_losses,
        datasets, 
        args
    ) -> None:
        self.gpu_id = 0
        self.enc = enc.to(self.gpu_id)
        self.dec = dec.to(self.gpu_id)
        self.data_loaders = data_loaders
        self.optimizer = optimizer
        self.critions = critions
        self.datasets = datasets
        self.reg_losses = reg_losses

        self.global_step = 0
        self.epoch = 0
        self.args = args

    def _run_epoch_train(self, epoch):
        self.enc.eval()
        self.dec.train()
        losses_names = ['total'] + ['criteria', 'D', 'ED', 'DE', 'geom'] + list(self.reg_losses.keys())
        losses = dict(zip(losses_names, [AverageMeter() for _ in range(len(losses_names))]))
        

        with tqdm(desc = 'Train', total = len(self.data_loaders['train'])) as bar:
            for batch_idx, (images_gt, fmri_gt) in enumerate(self.data_loaders['train']):
                unlabeled_fmri = next(iter(self.data_loaders['unlabeled_fmri']), None)[0]
                unlabeled_images = next(iter(self.data_loaders['external_images']), None)[0]
                
                images_gt, fmri_gt, unlabeled_fmri, unlabeled_images = \
                    map(lambda x: x.to(self.gpu_id), [images_gt, fmri_gt, unlabeled_fmri, unlabeled_images])
                ''' supervised learning train  : dec (fMRI 2 image)'''
                images_D = self.dec(fmri_gt)
                
                loss_criteria = 0
                self.critions['image'].eval()
                loss_D_list = self.critions['image'](images_D, images_gt)
                loss_D = sum(tup2list(loss_D_list, 1))
                losses['D'].update(loss_D.data)

                ''' unsupervised learning train '''
                # image 2 image
                with torch.no_grad():
                    fmri_E = self.enc(unlabeled_images).detach()
                images_ED = self.dec(fmri_E)
                
                # fMRI 2 fMRI
                fmri_DE = self.enc(self.dec(unlabeled_fmri))
                # unsupervised loss
                loss_ED = loss_DE = 0
                loss_ED_list = self.critions['image'](images_ED, unlabeled_images)
                loss_ED = sum(tup2list(loss_ED_list, 1))
                losses['ED'].update(loss_ED.data)

                loss_DE = self.critions['fmri'](fmri_DE, unlabeled_fmri)
                losses['DE'].update(loss_DE.data)
                loss_criteria += sum([float(w) * l for w, l in zip(self.args.loss_weights, [loss_D, loss_ED, loss_DE])])
                    
                losses['criteria'].update(loss_criteria.data)


                if self.args.reg_loss:
                    reg_loss_tot = 0
                    for loss_name, (w, reg_loss_func) in self.reg_losses.items():
                        reg_loss = reg_loss_func(self.dec)
                        losses[loss_name].update(reg_loss.data)
                        # if callable(w):
                        #     w = w(global_step)
                        reg_loss_tot += w * reg_loss
                    loss = loss_criteria + reg_loss_tot
                else:
                    loss = loss_criteria

                ''' total loss '''
                losses['total'].update(loss.data)

                ''' back propagation '''
                self.critions['image'].zero_grad()
                self.dec.zero_grad()
                self.enc.zero_grad()
                loss.backward()
                

    def run_train(self):
        # with SummaryWriter(comment='Decoder training') as sum_writer:
        for epoch in range(self.args.n_epochs):
            self.epoch = epoch
            self._run_epoch_train(epoch)


def get_model(args, name = "enc", pertrained = False):
    if name == "enc":
        model = make_model('SeparableEncoderVGG19ml', args.G_n_voxels, args.random_crop_pad_percent, drop_rate=0.25)
        if pertrained:
            state_dict = torch.load(args.enc_cpt_path)['state_dict']
            model.load_state_dict(state_dict)
    elif name == 'res_channel_attn_multi_scale':
        model = ResChanAttnMultiScaleAttn(args.G_n_voxels, args.im_res, start_CHW=(64, 14, 14), out_chan_ls=[64, 64, 64], n_chan_output=args.G_n_chan_dec_output, G = args.gn)

def get_dataloader():
    # Pseudocode, omitting the specific implementation of the function.
    pass
def get_optimizer():
    # Pseudocode, omitting the specific implementation of the function.
    pass
def get_criterion():
    # Pseudocode, omitting the specific implementation of the function.
    pass
def get_regularization_loss():
    # Pseudocode, omitting the specific implementation of the function.
    pass
def main():    
    args = Args()
    ''' Get Dataset '''
    # Pseudocode, omitting the specific implementation of the get_dataloader function.
    data_loaders, datasets = get_dataloader()
    ''' load models '''
    # enc: for pretraining
    enc = get_model(args, name = "enc", pertrained = args.enc_cpt_path != '')    
    dec = get_model(args, name = args.exp_prefix, pertrained = False)

    optimizer = get_optimizer()
    critions = get_criterion()
    reg_losss = get_regularization_loss()
    trainer = Trainer(dec, enc, data_loaders, optimizer, critions, reg_losss, datasets, args)
    trainer.run_train()


if __name__ == "__main__":
    main()
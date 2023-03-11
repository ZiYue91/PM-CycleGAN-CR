import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class ZyResult2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(netG='physics_GJ', no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--netGtGA', type=str, default='physics_GtGA', help='weight for cycle loss (A -> B -> A)')
        if is_train:
            parser.add_argument('--lambda_D', type=float, default=1, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_G_A', type=float, default=1, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_rec_trans', type=float, default=10, help='use identity mapping.')
            parser.add_argument('--lambda_rec_atmos', type=float, default=10, help='use identity mapping.')
            parser.add_argument('--lambda_recA', type=float, default=10, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_recB', type=float, default=10, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_idt', type=float, default=1, help='use identity mapping.')
            parser.add_argument('--lambda_perceA', type=float, default=1, help='use identity mapping.')
            parser.add_argument('--lambda_perceB', type=float, default=1, help='use identity mapping.')
        else:
            parser.add_argument('--lambda_recB', type=float, default=1, help='weight for cycle loss (A -> B -> A)')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D', 'G_A', 'idt_A', 'idt_transmission', 'cycle_transmission', 'cycle_atmospheric', 'cycle_A',  'cycle_B',
                           'perce_A', 'perce_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'transmission', 'fake_B1', 'fake_B2']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'rec_transmission']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'GtGA', 'D_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'GtGA']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netGtGA = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netGtGA, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.styleLossNet = networks.StyleLossNet1234().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netGtGA.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.transmission, self.atmospheric = self.netGtGA(self.real_A)
        self.atmospheric = self.atmospheric.unsqueeze(-1).unsqueeze(-1)

        self.rec_A = self.fake_B + self.atmospheric * self.transmission

        self.fake_B1 = self.real_A - self.atmospheric * self.transmission
        fake_B = torch.clamp(self.fake_B, 0, 1.0)
        fake_B1 = torch.clamp(self.fake_B1, 0, 1.0)
        self.fake_B2 = fake_B * 0.5 + fake_B1 * 0.5

        self.transmission_real = self.transmission.detach()
        self.atmospheric_real = self.atmospheric.detach()
        self.fake_A = self.real_B + self.atmospheric_real * self.transmission_real  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

        self.rec_transmission, self.rec_atmospheric = self.netGtGA(self.fake_A)
        self.rec_atmospheric = self.rec_atmospheric.unsqueeze(-1).unsqueeze(-1)



    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_B = self.fake_B_pool.query(self.fake_B)
        pred_fake = self.netD_A(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        pred_real = self.netD_A(self.real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_D
        self.loss_D.backward()

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        # Identity loss
        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        self.idt_A = self.netG_A(self.real_B)
        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.opt.lambda_idt
        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        self.idt_transmission, self.idt_atmospheric = self.netGtGA(self.real_B)
        self.loss_idt_transmission = self.criterionIdt(self.idt_transmission, self.idt_transmission+1-self.idt_transmission) * self.opt.lambda_idt

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) * self.opt.lambda_G_A

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.opt.lambda_recA
        self.loss_perce_A = self.styleLossNet.calculate_perceptual_loss(self.rec_A, self.real_A) * self.opt.lambda_perceA

        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.opt.lambda_recB
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_transmission = self.criterionCycle(self.rec_transmission, self.transmission_real) * self.opt.lambda_rec_trans
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_atmospheric = self.criterionCycle(self.rec_atmospheric, self.atmospheric_real) * self.opt.lambda_rec_atmos
        self.loss_perce_B = self.styleLossNet.calculate_perceptual_loss(self.rec_B, self.real_B) * self.opt.lambda_perceB

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_cycle_transmission + self.loss_cycle_atmospheric + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_transmission+\
                      self.loss_perce_A + self.loss_perce_B

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(self.netD_A, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad(self.netD_A, True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights

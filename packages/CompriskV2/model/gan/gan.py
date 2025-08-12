from typing import List

import numpy
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import spectral_norm

from CompriskV2.model.gan import GANAutoEncoder
from CompriskV2.model.gan.gan_classificator import GANClassificator

from CompriskV2.model import LinkedParameterModel
from Dataflux import get_logger, logging_mode
from Dataflux.pytorch import BATCH_TYPE

logger = get_logger(__name__, path="./logfile", level=logging_mode)


class GAN(LinkedParameterModel):

    def __init__(
            self,
            num_class_steps: int = 4,  # number of classifier steps per autoencoder step
            features_in: List[str] = None,
            features_out: List[str] = None,
            autoencoder_dropout_prob: float = 0.15,
            gradient_clipping_val: float = 1.0,
            gen_lr: float = 1e-5,
            disc_lr: float = 1e-5,
            *args,
            **kwargs
    ) -> None:

        super().__init__(features_in=features_in, features_out=features_out,*args, **kwargs)
        self._num_class_steps = num_class_steps
        self._gradient_clipping_val = gradient_clipping_val
        self._gen_lr = gen_lr
        self._disc_lr = disc_lr
        self.automatic_optimization = False
        #self._loss = torch.nn.BCEWithLogitsLoss()
        self._loss_name = "BCELoss"
        self.lambda_smooth = 0.35
        self.lambda_mse = 1
        self._steps = 0
        self._class_steps_per_gen_step = num_class_steps
        self._lr_decay = 0.9

        self.training_step_outputs = {
            "Generator MSE": [],
            "D real": [],
            "D loss real": [],
            "D fake": [],
            "D loss fake": [],
            "D fool": [],
            "D loss fool": [],
        }


        self.generator = GANAutoEncoder(
            num_features=self._features_in,
            num_features_out=self._features_out,
        )
        #self.generator = torch.load("packages/CompriskV2/model/gan/pretrained_gen_3E_B256.pt")

        self.discriminator = GANClassificator(
            num_features=self._features_in,
            sequence_input_size=self._sequence_out,
        )

    @property
    def labels(self) -> tuple[str]:
        return ("y_scaled", "y_hat", "score")

    @property
    def embedding(self) -> tuple[torch.Tensor]:
        return (
            torch.zeros((1, self._sequence_in, self._features_in)),
        )

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict]]:
        g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self._gen_lr)
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self._disc_lr)

        g_lr_scheduler = ExponentialLR(g_optimizer,gamma = self._lr_decay)
        d_lr_scheduler = ExponentialLR(d_optimizer, gamma = self._lr_decay)

        return [g_optimizer, d_optimizer], [{"scheduler": g_lr_scheduler, "interval": "epoch"}, {"scheduler": d_lr_scheduler, "interval": "epoch"}]

    def disc_loss_with_smoothness_penalty(self, y_hat, x):
        smoothness_loss = torch.mean((y_hat[:,1:,:] - y_hat[:,:-1,:])**2) # temporal difference penalty
        mse_loss = torch.nn.BCELoss()(y_hat, x)
        return mse_loss + smoothness_loss * self.lambda_smooth

    def gen_loss_with_smoothness_penalty(self, y_hat, x):
        smoothness_loss = torch.mean((y_hat[:,1:,:] - y_hat[:,:-1,:])**2) # temporal difference penalty
        mse_loss = torch.nn.functional.binary_cross_entropy(y_hat, x)
        return mse_loss + smoothness_loss * self.lambda_smooth

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x, _ = super().forward(x, x)
        y_hat = self.generator(x)
        score = torch.abs(x - y_hat)
        c = self.discriminator(x)
        # TODO beides ausgeben oder nur eins
        return x, y_hat, score

    def on_train_start(self) -> None:
        print("Device: ", self.device)
        print("Discriminator learning rate: ", self._disc_lr)
        print("Generator learning rate: ", self._gen_lr)
        print("Discriminator steps per Generator step: ", self._num_class_steps)

    def training_step(self, batch: BATCH_TYPE, **kwargs):
        #logger.debug("Training step")

        x, y = batch
        x = self._input_transformer(x)
        y = self._output_transformer(y)
        g_opt, d_opt = self.optimizers()
        #label_randomizer = ["flipped", "noisy"]
        #choice = random.choice(label_randomizer)

        real_label = torch.ones((y.shape[0], 1, 1), device=self.device)
        fake_label = torch.zeros((y.shape[0], 1, 1), device=self.device)

        # Soft or flipped labels to mess with discriminator
        #if (self._steps + 1) % 4 == 0:
        #    match choice:
        #        case "flipped":
        #            real_label = torch.zeros((y.shape[0], 60, 1), device=self.device)
        #            fake_label = torch.ones((y.shape[0], 60, 1), device=self.device)
        #        case "softened":
        #            real_label = torch.ones((y.shape[0], 60, 1), device=self.device).add(random.uniform(-0.3, 0))
        #            fake_label = torch.zeros((y.shape[0], 60, 1), device=self.device).add(random.uniform(0, 0.3))



        # generate fake batch for D
        y_hat = spectral_norm(self.generator(x))
        self.training_step_outputs["Generator MSE"].append(torch.nn.MSELoss()(y_hat, x).item())
        self.log("training/gen_MSE", torch.nn.MSELoss()(y_hat, x).item())

        #########################
        # Update Discriminator: #
        #########################
        self.toggle_optimizer(d_opt)
        # Forward pass real batch through D
        d_output_real = spectral_norm(self.discriminator(x))
        # Calculate loss for real batch
        lossD_real = self._loss(d_output_real, real_label)
        # safe real loss for training monitoring
        self.log("training/D_loss_real", lossD_real.item())
        self.log("training/D_output_real", torch.mean(d_output_real))
        self.training_step_outputs["D real"].append(torch.mean(d_output_real))
        self.training_step_outputs["D loss real"].append(lossD_real.item())
        # Calculate gradients for D
        self.manual_backward(lossD_real)
        torch.nn.utils.clip_grad_value_(self.discriminator.parameters(), self._gradient_clipping_val)


        # Forward pass fake batch through D
        d_output_fake = spectral_norm(self.discriminator(y_hat))
        # Calculate loss for fake batch
        lossD_fake = self._loss(d_output_fake, fake_label)
        # safe fake loss for training monitoring
        self.log("training/D_loss_fake", lossD_fake.item())
        self.log("training/D_output_fake", torch.mean(d_output_fake))
        self.training_step_outputs["D fake"].append(torch.mean(d_output_fake))
        self.training_step_outputs["D loss fake"].append(lossD_fake.item())
        # Calculate gradients for D
        self.manual_backward(lossD_fake)
        torch.nn.utils.clip_grad_value_(self.discriminator.parameters(), self._gradient_clipping_val)
        # Update Discriminator
        d_opt.step()
        d_opt.zero_grad()
        self.untoggle_optimizer(d_opt)


        ###################
        # Update Generator #
        ###################
        if (self._steps + 1) % self._class_steps_per_gen_step == 0:
            self.toggle_optimizer(g_opt)
            y_hat = spectral_norm(self.generator(x))

            #for param in self.discriminator.parameters():
            #    param.requires_grad = False

            # forward pass fake data through updated Discriminator
            d_output_fool = spectral_norm(self.discriminator(y_hat))
            # generate loss on how well the Generator fooled to Discriminator
            lossG = torch.nn.functional.binary_cross_entropy_with_logits(d_output_fool, real_label)
            #lossG_mse = torch.nn.MSELoss()(y_hat, x)
            #lossG = lossG + lossG_mse * self.lambda_mse
            self.log("training/D_loss_fool", lossG.item())
            self.log("training/D_output_fool", torch.mean(d_output_fool))
            self.training_step_outputs["D fool"].append(d_output_fool)
            self.training_step_outputs["D loss fool"].append(lossG.item())
            self.manual_backward(lossG)
            torch.nn.utils.clip_grad_value_(self.generator.parameters(), self._gradient_clipping_val)
            g_opt.step()
            g_opt.zero_grad()
            self.untoggle_optimizer(g_opt)

            #for param in self.discriminator.parameters():
            #    param.requires_grad = True


        self._steps += 1

    def on_train_epoch_end(self) -> None:
        #training_losses = torch.stack(self.training_step_outputs)

        print("Generator MSE", numpy.average(self.training_step_outputs["Generator MSE"]))
        #print("D real", numpy.average(self.training_step_outputs["D real"]))
        #print("D fake", numpy.average(self.training_step_outputs["D fake"]))
        #print("D fool", numpy.average(self.training_step_outputs["D fool"]))
        print("D loss real", numpy.average(self.training_step_outputs["D loss real"]))
        print("D loss fake", numpy.average(self.training_step_outputs["D loss fake"]))
        print("D loss fool", numpy.average(self.training_step_outputs["D loss fool"]))

        for _, value in self.training_step_outputs.items():
            value.clear()

        self._steps = 0

    def validation_step(self, batch: BATCH_TYPE, **kwargs):
        x, y = batch
        x = self._input_transformer(x)
        y = self._output_transformer(y)

        self.generator.eval()
        self.discriminator.eval()
        y_hat = self.generator(x)
        gen_loss = torch.nn.MSELoss()(y_hat, x)
        real_label = torch.ones((y.shape[0], 1, 1), device=self.device)
        fake_label = torch.zeros((y.shape[0], 1, 1), device=self.device)

        #loss = self._loss(y_hat, real_label)
        #self.log(f"validate_generator/{self._loss_name}", loss)
        #self.log(f"validate/{self._loss_name}", loss)


        x_class_output = self.discriminator(x)
        loss_class_real = self._loss(x_class_output, real_label)
        self.log("disc_output/real", x_class_output.mean())
        self.log(f"validate_discriminator/real_{self._loss_name}", loss_class_real)

        y_class_output = self.discriminator(y_hat)
        loss_class_fake = self._loss(y_class_output, fake_label)
        self.log("disc_output/fake", y_class_output.mean())
        self.log(f"validate_discriminator/fake_{self._loss_name}", loss_class_fake)

        self.log(f"validate/{self._loss_name}", loss_class_real+loss_class_fake)
        self.log(f"validate/MSELoss", gen_loss)

        #self.log("real_batch", x[1,:,1])
        #self.log("reconstructed_batch", y_hat[1,:,1])

        self.generator.train()
        self.discriminator.train()

        return loss_class_real+loss_class_fake

    def predict_step(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self(x)

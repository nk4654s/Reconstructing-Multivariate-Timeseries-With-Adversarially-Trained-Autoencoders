import gc
import os
import sys
from typing import List, Any

import numpy as np
from lightning.pytorch.callbacks import model_checkpoint
import pandas as pd
import torch
from statistics import mean
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
from Comprisk.helper.structure.loss import r2_score, batch_wasserstein

from CompriskV2.model.gan import GANAutoEncoder
from CompriskV2.model.gan.gan_classificator import GANClassificator

from CompriskV2.model import LinkedParameterModel
from Dataflux import get_logger, logging_mode
from Dataflux.pytorch import BATCH_TYPE

logger = get_logger(__name__, path="./logfile", level=logging_mode)


class GAN(LinkedParameterModel):

    def __init__(
            self,
            features_in: List[str] = None,
            features_out: List[str] = None,
            disc_output_linear_hidden_dim: int = 128,

            gen_lr: float = 1e-5,
            disc_lr: float = 1e-5,
            gen_lr_decay: float = 0.9,
            disc_lr_decay: float = 0.9,
            gradient_clipping_val: float = 1.0,

            num_class_steps: int = 4,  # number of classifier steps per autoencoder step
            autoencoder_dropout_rate: float = 0.0,
            label_smoothing: float = 0.0,
            *args,
            **kwargs
    ) -> None:

        super().__init__(features_in=features_in, features_out=features_out,*args, **kwargs)
        self._disc_output_linear_hidden_dim = disc_output_linear_hidden_dim

        self._gen_lr = gen_lr
        self._disc_lr = disc_lr
        self._gen_lr_decay = gen_lr_decay
        self._disc_lr_decay = disc_lr_decay
        self._gradient_clipping_val = gradient_clipping_val

        self._class_steps_per_gen_step = num_class_steps
        self._autoencoder_dropout_rate = autoencoder_dropout_rate
        self._label_smoothing = label_smoothing

        self._steps = 0 # counting number of steps for comparison with _class_steps_per_gen_step
        self.automatic_optimization = False

        #self.lambda_smooth = 0.35
        #self.lambda_mse = 1
        #self.custom_loss = CustomBCELossWithLogits()

        self.training_step_outputs = {
            "G MSE": [],
            "G Wasserstein": [],
            "D output real": [],
            "D output fool": [],
            "D output fake": [],
            "D loss real": [],
            "D loss fake": [],
            "D loss fool": [],
        }


        self.generator = GANAutoEncoder(
            num_features_in=self._features_in,
            num_features_out=self._features_out,
            weight_init=self._weight_init,
            dropout_rate=self._autoencoder_dropout_rate,
        )

        self.discriminator = GANClassificator(
            num_features_in=self._features_in,
            sequence_in=self._sequence_in,
            output_linear_hidden_dim= self._disc_output_linear_hidden_dim,
            weight_init=self._weight_init,
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

        g_optimizer = self._optim(self.generator.parameters(), lr=self._gen_lr)
        d_optimizer = self._optim(self.discriminator.parameters(), lr=self._disc_lr)

        g_lr_scheduler = ExponentialLR(g_optimizer,gamma = self._gen_lr_decay)
        d_lr_scheduler = ExponentialLR(d_optimizer, gamma = self._disc_lr_decay)

        return [g_optimizer, d_optimizer], [{"scheduler": g_lr_scheduler, "interval": "epoch"}, {"scheduler": d_lr_scheduler, "interval": "epoch"}]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x, _ = super().forward(x, x)
        x_hat = self.generator(x)
        score = torch.abs(x - x_hat)
        c = self.discriminator(x)
        # TODO beides ausgeben oder nur eins
        return x, x_hat, score

    def on_epoch_start(self) -> None:
        self._steps = 0

    def on_train_start(self) -> None:
        print("Device: ", self.device)
        print("Discriminator learning rate: ", self._disc_lr)
        print("Generator learning rate: ", self._gen_lr)
        print("Discriminator steps per Generator step: ", self._class_steps_per_gen_step)

    def training_step(self, batch: BATCH_TYPE, **kwargs):
        #logger.debug("Training step")
        #self.log("training/D_loss_fake", lossD_fake.item())

        x, y = batch
        x = self._input_transformer(x)
        y = self._output_transformer(y)
        g_opt, d_opt = self.optimizers()

        real_label = torch.ones((y.shape[0], 1), device=self.device)
        fake_label = torch.zeros((y.shape[0], 1), device=self.device)
        # real_label = torch.squeeze(torch.nn.functional.one_hot(real_label.long(), num_classes=2).type(torch.float32))
        # fake_label = torch.squeeze(torch.nn.functional.one_hot(fake_label.long(), num_classes=2).type(torch.float32))
        #apply label smoothing
        #real_label = real_label * (1 - (self._label_smoothing * 2)) + self._label_smoothing
        #fake_label = fake_label * (1 - (self._label_smoothing * 2)) + self._label_smoothing

        # generate fake batch for D
        x_hat = self.generator(x).detach() # prevent disc update from flowing back into generator


        #########################
        # Update Discriminator: #
        #########################
        self.toggle_optimizer(d_opt)
        d_opt.zero_grad()

        # Forward pass real batch through D
        d_output_real = self.discriminator(x)
        # Calculate loss for real batch
        d_loss_real = self._loss(d_output_real, real_label)
        # safe real loss for training monitoring


        # Forward pass fake batch through D
        d_output_fake = self.discriminator(x_hat)
        # Calculate loss for fake batch
        d_loss_fake = self._loss(d_output_fake, fake_label)
        # Calculate gradients for D
        loss = d_loss_fake + d_loss_real
        self.manual_backward(loss)
        self.clip_gradients(d_opt, self._gradient_clipping_val, "norm")
        # Update Discriminator
        d_opt.step()

        self.untoggle_optimizer(d_opt)
        #########################
        # Update Discriminator: #
        #########################

        # TODO fix memory leak ??? exploding weights?
        #####################
        # Update Generator: #
        #####################
        self.toggle_optimizer(g_opt)
        g_opt.zero_grad()

        x_hat = self.generator(x)
        # forward pass fake data through updated Discriminator
        d_output_fool = self.discriminator(x_hat)
        # generate loss on how well the Generator fooled to Discriminator
        d_loss_fool = self._loss(d_output_fool, real_label)
        if self._steps % self._class_steps_per_gen_step == 0:
            self.manual_backward(d_loss_fool)
            self.clip_gradients(g_opt, self._gradient_clipping_val, "norm")
            g_opt.step()

        self.untoggle_optimizer(g_opt)
        #####################
        # Update Generator: #
        #####################

        self._steps += 1

        #####################
        # Log Step Results: #
        #####################
        self.log("discriminator_train/output_real", d_output_real.mean().item())
        self.log("discriminator_train/output_fake", d_output_fake.mean().item())
        self.log("discriminator_train/output_fool", d_output_fool.mean().item())

        self.log("discriminator_train/loss_real", d_loss_real.item())
        self.log("discriminator_train/loss_fake", d_loss_fake.item())
        self.log("discriminator_train/loss_fool", d_loss_fool.item())

        gen_mse = torch.nn.MSELoss()(x_hat, x).detach()
        gen_r2 = r2_score(x_hat, x).detach()
        ws_loss = batch_wasserstein(x_hat, x).detach()

        self.log("generator_train/mse_loss", gen_mse.item())
        self.log("generator_train/r2_score", gen_r2.item())
        self.log("generator_train/wasserstein_distance", ws_loss.item())

        self.training_step_outputs["G MSE"].append(gen_mse.item())
        self.training_step_outputs["G Wasserstein"].append(ws_loss.item())
        self.training_step_outputs["D output real"].append(d_output_real.mean().item())
        self.training_step_outputs["D output fool"].append(d_output_fool.mean().item())
        self.training_step_outputs["D output fake"].append(d_output_fake.mean().item())
        self.training_step_outputs["D loss real"].append(d_loss_real.item())
        self.training_step_outputs["D loss fool"].append(d_loss_fool.item())
        self.training_step_outputs["D loss fake"].append(d_loss_fake.item())
        #####################
        # Log Step Results: #
        #####################

    def on_train_epoch_end(self) -> None:
        print("Generator MSE: ", mean(self.training_step_outputs["G MSE"]))
        print("Generator Wasserstein: ", mean(self.training_step_outputs["G Wasserstein"]))
        print("Discriminator Output Real: ", mean(self.training_step_outputs["D output real"]))
        print("Discriminator Output Fool: ", mean(self.training_step_outputs["D output fool"]))
        print("Discriminator Output Fake: ", mean(self.training_step_outputs["D output fake"]))
        print("Discriminator Loss Real: ", mean(self.training_step_outputs["D loss real"]))
        print("Discriminator Loss Fool: ", mean(self.training_step_outputs["D loss fool"]))
        print("Discriminator Loss Fake: ", mean(self.training_step_outputs["D loss fake"]))

        for k in self.training_step_outputs.keys():
            self.training_step_outputs[k].clear()
        print(sys.getsizeof(self.training_step_outputs))

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.generator.eval()
        checkpoint_folder = self.trainer.checkpoint_callback.__dict__["dirpath"]
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        file = "environment/dev/datasets/test_dataset_march_24/0b3426b3-94af-4b8c-89c1-da57493c099a_2024-03-23.feather"

        df = pd.read_feather(file)
        df.drop(columns=["ts"], inplace=True)
        df_tensor = torch.tensor(df.values).to(self.device)
        data = self._input_transformer(df_tensor)
        num_features = data.shape[1]
        num_sequences = data.shape[0] // self._sequence_in
        #trim data to be divisible by num_sequence
        data = data[:num_sequences * self._sequence_in]
        data = data.reshape(num_sequences, self._sequence_in, num_features)

        output = []
        input_list = []
        with torch.no_grad():
            for i in range(data.shape[0]):
                x = data[i]
                x_hat = self.generator(x).detach().cpu().tolist()
                input_list.extend(x.cpu().tolist())
                output.extend(x_hat)

        # TODO mach schön
        input_list = np.array(input_list).T
        output = np.array(output).T


        fig, axs = plt.subplots(self._features_out, 1, sharey="row", figsize=(12, 6 * self._features_out))

        for feature in range(self._features_out):
            axs[feature].plot(input_list[feature], zorder=1)
            axs[feature].set_title(f"{df.columns[feature]}")
            axs[feature].plot(output[feature], alpha=0.5, zorder=2)

        plt.savefig(fname=f"{checkpoint_folder}/checkpoint_on_epoch_{self.current_epoch}_monitoring_graphs.jpg")

        plt.close("all")
        gc.collect()

        del input_list, output, df, df_tensor, data, x_hat, x

        self.generator.train()
        return

    def validation_step(self, batch: BATCH_TYPE, **kwargs):
        x, y = batch
        x = self._input_transformer(x)
        y = self._output_transformer(y)
        real_label = torch.ones((y.shape[0], 1), device=self.device)
        fake_label = torch.zeros((y.shape[0], 1), device=self.device)
        # real_label = torch.squeeze(torch.nn.functional.one_hot(real_label.long(), num_classes=2).type(torch.float32))
        # fake_label = torch.squeeze(torch.nn.functional.one_hot(fake_label.long(), num_classes=2).type(torch.float32))

        self.generator.eval()
        self.discriminator.eval()

        x_hat = self.generator(x)
        d_output_real = self.discriminator(x)
        d_output_fake = self.discriminator(x_hat)

        d_loss_real = self._loss(d_output_real, real_label)
        d_loss_fake = self._loss(d_output_fake, fake_label)
        d_loss = d_loss_fake + d_loss_real
        with torch.no_grad():
            gen_loss = torch.nn.MSELoss()(x_hat, x).detach()
            gen_r2 = r2_score(x_hat, x).detach()
            ws_loss = batch_wasserstein(x_hat, x).detach()

        self.log("discriminator_validation/output_real", d_output_real.mean())
        self.log("discriminator_validation/loss_real", d_loss_real.item())

        self.log("discriminator_validation/output_fake", d_output_fake.mean())
        self.log("discriminator_validation/loss_fake", d_loss_fake.item())

        self.log("generator_validation/mse_loss", gen_loss.item())
        self.log("generator_validation/r2_score", gen_r2.item())
        self.log("generator_validation/wasserstein_distance", ws_loss.item())

        self.generator.train()
        self.discriminator.train()

        return d_loss

    def predict_step(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self(x)

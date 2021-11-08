
# support both parallel training and unparallel

from datasets.dataset_generator import get_train_val_dataset
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import torch.nn as nn
import time
import numpy as np
from utils.utils import createDirectory, createCheckpointDir, convert_secs2time, AverageMeter, time_string, load_helpers
from utils.utils import calculate_similarity, calculate_eer
from network.resnet51 import resnet51
from tqdm import tqdm
from torch.optim import lr_scheduler
from training.model_manager import ModelManager



class Trainer(ModelManager):

    def __init__(self, config):
 
        super().__init__(config)
        
        self.datasets = get_train_val_dataset(self.config)

        # Is the instance for Debugging
        self.debug = self.config.basic.debug_mode
        
        # Directory where model checkpoints will be saved
        self.checkpoint_dir = createCheckpointDir(outputFolderPath = self.config.basic.result_dir, debug_mode=self.debug)
        

        # Open file for logging
        self.fh = open(os.path.join(self.checkpoint_dir, "log_file.txt"), "a")

        # Create a tensorboard summary writter
        self.summary_writer = self._create_summary_writer()

        # Create the specified model and the optimizer
        self.optimizer = self._get_optimizer() 
        self.lr_scheduler = self._get_lr_scheduler()

        self.loss_fn = self._get_loss_fn()

        # Try to load a pre-trained model.
        self.curr_epoch, self.best_epoch, self.EER, self.train_accuracy, self.train_loss = load_helpers()
        
        # Print config file
        self.print_to_log('------------------------------------------')
        for keys,values in self.config.items():
            self.print_to_log( str(keys) + " : " +  str(values) )
            self.print_to_log('\n')
        self.print_to_log('------------------------------------------')
        


    def _get_optimizer(self):
        """
        Get different optimizers for different experiments
        :return: Optimizer
        """

        # SGD Optimizer
        if self.config.training.optimizer.type == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config.training.optimizer.learning_rate, 
                                        weight_decay=self.config.training.optimizer.weight_decay, 
                                        momentum=self.config.training.optimizer.momentum)

        # RMSprop Optimizer
        elif self.config.training.optimizer.type == "RMSprop":
            optimizer = torch.optim.RMSprop(self.model.parameters(),
                                    lr=self.config.training.optimizer.learning_rate,
                                    momentum=self.config.training.optimizer.momentum)

        # Adam Optimizer
        elif self.config.training.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(),
                                lr=self.config.training.optimizer.learning_rate)
        
        else:
            raise Exception("Optimizer {} does not exist".format(self.config.training.optimizer.type))
        
        if self.config.training.resume:
            optimizer.load_state_dict(self.old_checkpoint['optimizer'])
        
        return optimizer



    def _get_lr_scheduler(self):
        """
        get the learning rate scheduler
        :return: LR Scheduler
        """

        # Step learning rate
        if self.config.training.lr_scheduler.type == "StepLR":
            return lr_scheduler.StepLR(self.optimizer,
                                       step_size=self.config.training.lr_scheduler.step_size,
                                       gamma=self.config.training.lr_scheduler.gamma)

        # Reduce on plateau learning rate
        elif self.config.training.lr_scheduler.type == "ReduceLROnPlateau":
            return lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                  patience=self.config.training.lr_scheduler.patience,
                                                  factor=self.config.training.lr_scheduler.gamma, cooldown=1)
        else:
            raise Exception("Scheduler {} does not exist".format(self.config.training.lr_scheduler.type))


    def _get_loss_fn(self):

        # when model is resnet51, use Cross Entropy Loss
        if self.config.model.name == "resnet51":
            return nn.CrossEntropyLoss()
        
        # ArcFace loss losses.ArcFaceLoss(num_classes, embedding_size, margin=28.6, scale=64, **kwargs)
        if self.config.model.name == "arcface":  
            return nn.NLLLoss()

        raise Exception("Criterion {} does not exist".format(self.config.training.criterion.type))

   
    def _create_summary_writer(self):
        return SummaryWriter(log_dir=os.path.join(self.checkpoint_dir, "tensorboard_logs"))



    def train_val_test(self):
        # To meassure average epoch processing time
        epoch_time = AverageMeter()
        # Time of the beginning of an epoch
        start_time = time.time()  

        for curr_epoch in range(1, self.config.training.epochs + 1): 
            # Update current epoch count
            self.curr_epoch = curr_epoch

            # Log the estimated time for finishing the experiment
            self._log_estimated_epoch_time(epoch_time=epoch_time, lr=self.optimizer.param_groups[0]["lr"])

            # Train for one epoch
            train_loss, train_accuracy = self.train_epoch()

            # Validate model on the validation set
            EER = self.validate_epoch()

            # Adjust the learning rate after each epoch, according to the lr scheduler
            self.lr_scheduler.step(train_loss)

            if self.summary_writer is not None:
                # Log metrics to the tensorboard
                self.summary_writer.add_scalars(f"loss",
                                                {"train": train_loss},
                                                self.curr_epoch)
                self.summary_writer.add_scalars(f"Metric",
                                                {"train": train_accuracy, "val": EER},
                                                self.curr_epoch)

            # Calculate epoch time, and restart timer
            epoch_time.update(time.time() - start_time)
            start_time = time.time()

            # Save a checkpoint of the model after each epoch
            self.save_checkpoint()

            # Check if the training should stop due to early stopping
            if (self.curr_epoch - self.best_epoch) == self.config.training.early_stop: 
                self.print_to_log("EARLY STOPPING \n")
                break

        # Log the training report 
        self.training_end_report()

        # Close the file for logging
        self.fh.close()

        return self.checkpoint_dir 


    def train_epoch(self):
        # Train for one epoch
        #train_loss, train_metric = self.run_one_epoch(which="train")
        data_loader = self.datasets['dataloaders']['train']
        self.model.train()

        # For averaging batch processing times over the epoch
        batch_time = AverageMeter() 
        # For averaging data loading time over the epoch
        data_time = AverageMeter() 
        # For averaging losses over the epoch
        losses = AverageMeter()
        # For storing all (prediction, target) pairs in the epoch 
        metrics = AverageMeter()

        # Measure the beginning of the batch (and also beginiing of data loading)
        batch_start_time = time.time()  
        # Loop over the whole dataset once
            # for i, sample in enumerate(tqdm(data_loader))
        for i, sample in enumerate(tqdm(data_loader)):
            # Measure data loading/pre-processing time
            data_time.update(time.time() - batch_start_time)  

            # Transfer input and labels to GPU/Device
            inputs = sample[0].to(self.device)
            labels = sample[1].to(self.device)

            # Delete calculated gradients
            self.optimizer.zero_grad() 

            # only save computation grahy when training
            with torch.set_grad_enabled(True):
                _, predictions = self.model(inputs)
                
                # Compute the loss 
                loss = self.loss_fn(predictions, labels) 

            # Calculate the loss gradients
            loss.backward()
            # Update network weights with calculated gradients  
            self.optimizer.step() 

            # Update epoch loss averaging
            losses.update(loss.item())  

            arg_max = torch.argmax(predictions, 1)
            metric = (arg_max == labels).sum() / self.config.preprocessing.dataloader.batch_size

            # Update epoch metric averaging
            metrics.update(metric.item())
            
            del inputs, labels, predictions, loss, metric
            torch.cuda.empty_cache()

            # Measure the time it took to process the batch
            batch_time.update(time.time() - batch_start_time)
            # Measure the beginning of the next ba
            batch_start_time = time.time()  

        # Calculate average epoch loss
        loss_avg = losses.get_average()
        # Calculate the average of the metric (accuracy) of this epoch
        accuracy = metrics.get_average()

        self.print_to_log(f"train loss: {loss_avg:.6f}")
        self.print_to_log(f"train accuracy: {accuracy:.6f}")

        # Append the loss and acc to the list (save one value for evey epoch)
        self.train_loss.append(loss_avg)
        self.train_accuracy.append(accuracy)

        return loss_avg, accuracy


    def validate_epoch(self):
        # Validate for one epoch
        #val_loss, val_metric = self.run_one_epoch(which="val")
        data_loader = self.datasets['dataloaders']['val']
        self.model.eval()

        # For averaging batch processing times over the epoch
        batch_time = AverageMeter() 
        # For averaging data loading time over the epoch
        data_time = AverageMeter() 
        # For averaging losses over the epoch
        losses = AverageMeter()
        # For storing all (prediction, target) pairs in the epoch 
        metrics = AverageMeter()

        # Measure the beginning of the batch (and also beginiing of data loading)
        batch_start_time = time.time()  
       
        similarities = []
        labels = []
        # Loop over the whole dataset once
        for i, sample in enumerate(tqdm(data_loader)):
            # Measure data loading/pre-processing time
            data_time.update(time.time() - batch_start_time)  

            # Transfer input and labels to GPU/Device
            X1 = sample[0].to(self.device)
            X2 = sample[1].to(self.device)
            y = sample[2].to(self.device)

            # only save computation grahy when training
            with torch.set_grad_enabled(False):
                features_1, _ = self.model(X1)
                features_2, _ = self.model(X2)

            # calculate and append cosine similarity of each pair
            similarity = calculate_similarity(features_1, features_2)
            #print("similarity: ", similarity)
            #similarity.detach().cpu().numpy()
            similarities.append(similarity.tolist())
            labels.append(y.tolist())

            del X1, X2, features_1, features_2
            torch.cuda.empty_cache()

            # Measure the time it took to process the batch
            batch_time.update(time.time() - batch_start_time)
            # Measure the beginning of the next ba
            batch_start_time = time.time()  

        # calculate EER
        similarities = [item for sublist in similarities for item in sublist]
        labels = [item for sublist in labels for item in sublist]

        EER = calculate_eer(similarities, labels)

        self.print_to_log(f"EER: {EER:.6f}")

        # If the current validation loss is better than all previous epochs
        if self.curr_epoch == 1:
            self.best_epoch = 1
        elif EER < np.min(self.EER):
            self.best_epoch = self.curr_epoch

        # Append the loss and acc to the list (save one value for evey epoch)
        self.EER.append(EER)

        return EER




    def save_checkpoint(self):
        self.model.eval()  # Switch the model to evaluation mode

        if self.config.training.parallel:
            state = {'state_dict': self.model.module.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'curr_epoch': self.curr_epoch,
                 'best_epoch': self.best_epoch,
                 'train_accuracy': self.train_accuracy,
                 'train_loss': self.train_loss,
                 'EER': self.EER}
        else:
            state = {'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'curr_epoch': self.curr_epoch,
                 'best_epoch': self.best_epoch,
                 'train_accuracy': self.train_accuracy,
                 'train_loss': self.train_loss,
                 'EER': self.EER}
                 
        torch.save(state, os.path.join(self.checkpoint_dir, "checkpoint.pth"))
        if self.curr_epoch == self.best_epoch:
            torch.save(state, os.path.join(self.checkpoint_dir, "best_checkpoint.pth"))

        return


    def print_to_log(self, message):
        print(message, file=self.fh)
        print(message)
        self.fh.flush()

    def training_end_report(self):
        # Best epoch indexes to extract best metrics from arrays
        best_tr_epoch_i = int(np.argmin(self.train_loss))
        best_val_epoc_i = self.best_epoch - 1  # Minus one because epochs start from 1, and list indexing starts from 0

        self.print_to_log(" ")
        self.print_to_log("---------------------------------------------------------------------------------")
        self.print_to_log("End of training report:")
        self.print_to_log("---------------------------------------------------------------------------------")
        self.print_to_log("Best training loss: {:0.4f}".format(self.train_loss[best_tr_epoch_i]))
        self.print_to_log("Best training accuracy: {:0.4f}".format(self.train_accuracy[best_tr_epoch_i]))
        self.print_to_log("Best validation EER: {:0.4f}".format(self.EER[best_val_epoc_i]))
        self.print_to_log("Epoch with the best training loss: {}".format(best_tr_epoch_i + 1))
        self.print_to_log("Epoch with the smallest EER: {}".format(self.best_epoch))
        self.print_to_log("---------------------------------------------------------------------------------")
        self.print_to_log(" ")
        self.print_to_log("Finished training. Starting the evaluation.")
        self.print_to_log("---------------------------------------------------------------------------------")
        self.print_to_log(" ")
        return
    
    def _log_estimated_epoch_time(self, epoch_time, lr):
        """
        self.Print_to_logself.print_to_log the estimated time to finish this experiment, as well as the lr for the current epoch.

        :param epoch_time: average time per one epoch
        :param lr: current lr
        """
        # Info about the last epoch
        # (Do not self.print_to_log before first epoch happens)
        if epoch_time.val != 0.0:

            epoch_h, epoch_m, epoch_s = convert_secs2time(epoch_time.val)
            self.print_to_log('Epoch processing time: {:02d}:{:02d}:{:02d} (H:M:S) \n'.format(epoch_h, epoch_m, epoch_s))

        # Info about the beginning of the current epoch
        remaining_seconds = epoch_time.get_average() * (self.config.training.epochs - self.curr_epoch)  
        need_hour, need_mins, _ = convert_secs2time(remaining_seconds)
        need_time = '[Need: {:02d}:{:02d} (H:M)]'.format(need_hour, need_mins)
        self.print_to_log('{:3d}/{:3d} ----- [{:s}] {:s} LR={:}'.format(self.curr_epoch, self.config.training.epochs, time_string(), need_time, lr))









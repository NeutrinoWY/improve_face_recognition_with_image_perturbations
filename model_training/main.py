# run parallel or unparallel training
import os
from training import trainer
from utils.config_utils import get_config
import torch



def main():
    config = get_config()

    if not os.path.isdir(config.basic.result_dir):
        os.mkdir(config.basic.result_dir)
    if not os.path.isdir(config.basic.save_dir):
        os.mkdir(config.basic.save_dir)
    

    experiment = trainer.Trainer(config)

    checkpoint_dir = experiment.train_val_test()

    BEST_MODEL = os.path.join(checkpoint_dir, "best_checkpoint.pth")

    print(BEST_MODEL)

    return BEST_MODEL

    

if __name__ == "__main__":
    main()
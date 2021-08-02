#!/usr/bin/python3
"""
Main File to start training the AFFFE Network
"""
from utils.config_utils import get_config
from utils.utils import init_environment
from datasets import dataset_generator


def main():
    """
    Run training for a specific model
    """
    # Load configuration for training
    config = get_config()
    # Init environment, use GPU if available, set random seed
    device = init_environment(config)

    # test preprocess
    results = dataset_generator.get_train_val_dataset(config)
    print(results)

    dataloader_train = results['dataloaders']['train']
    print(next(iter(dataloader_train)))

    dataloader_val = results['dataloaders']['val']
    print(next(iter(dataloader_val)))


if __name__ == '__main__':
    main()

"""
Configuration Utils Function Collection
"""
import os
import argparse
import collections
from datetime import datetime
import yaml
from dotmap import DotMap


def get_config(config_name="train"):
    """
    Reads the configuration and returns a configuration DotMap
    :param config_name: Optional name of the configuation file
    :return: Configuration DotMap with arguments parsed form terminal/cmd
    """

    # Read arguments
    args = _read_arguments()

    # Read the config name specified
    if not config_name:
        config_name = args.__dict__['config.name']

    # load the file and parse it to a DotMap
    with open("model_training/config/" + config_name + ".yml", "r") as file:
        config_dict = yaml.safe_load(file)
    config = DotMap(config_dict)

    # Overwrite default values
    return _overwrite_defaults(config, args)


def _overwrite_defaults(config, args):
    """
    TODO: check before submission
    Overwrite the default values in the configuration DotMap
    :param config: configuration file
    :param args: command line arguments
    :return: DotMap Configuration with new values
    """

    # Overwrite all arguments that are set via terminal/cmd
    for argument, argument_value in args.__dict__.items():
        if argument_value is not None:
            config = _replace_value_in_config(config, argument, argument_value)
    return config


def _read_arguments():
    """
    Read the arguments from the command line/terminal

    :return: ArgParser
    """
    parser = argparse.ArgumentParser(description='Arguments for PyAffact')
    parser.add_argument('--config.name', default='train', type=str)

    parser.add_argument('--basic.cuda_device_name', default=None, type=str)
    parser.add_argument('--basic.experiment_name', default=None, type=str)
    parser.add_argument('--basic.experiment_description', default=None,type=str)
    parser.add_argument('--basic.result_dir', default=None, type=str)
    parser.add_argument('--basic.save_dir', default=None, type=str)
    parser.add_argument('--basic.debug_mode', default=None, type=int)
    parser.add_argument('--basic.checkpoint', default=None, type=str)

    parser.add_argument('--dataset.partition_filename', default=None, type=str)
    parser.add_argument('--dataset.train_list_label', default=None, type=str)
    parser.add_argument('--dataset.test_list_label', default=None, type=str)
    parser.add_argument('--dataset.val_pairs', default=None, type=str)
    parser.add_argument('--dataset.image_dir', default=None, type=str)
    parser.add_argument('--dataset.loose_landmark_train', default=None, type=str)
    parser.add_argument('--dataset.loose_landmark_test', default=None, type=str)
    parser.add_argument('--dataset.loose_bb_train', default=None, type=str)
    parser.add_argument('--dataset.loose_bb_test', default=None, type=str)
    parser.add_argument('--dataset.bounding_box_mode', default=None, type=int)
    parser.add_argument('--dataset.meta_file', default=None, type=str)
    parser.add_argument('--dataset.bounding_box_scale', default=None, type=int)

    parser.add_argument('--model.name', default=None, type=str)
    parser.add_argument('--model.pretrained', default=None, type=int)
    parser.add_argument('--model.embedding_size', default=None, type=int)
    parser.add_argument('--model.num_classes', default=None, type=int)

    parser.add_argument('--training.epochs', default=None, type=int)
    parser.add_argument('--training.resume', default=None, type=int)
    parser.add_argument('--training.parallel', default=None, type=int)
    parser.add_argument('--training.optimizer.type', default=None, type=str)
    parser.add_argument('--training.optimizer.learning_rate', default=None, type=float)
    parser.add_argument('--training.optimizer.momentum', default=None, type=float)
    parser.add_argument('--training.optimizer.weight_decay', default=None, type=float)
    parser.add_argument('--training.criterion.type', default=None, type=str)
    parser.add_argument('--training.lr_scheduler.type', default=None, type=str)
    parser.add_argument('--training.lr_scheduler.step_size', default=None, type=int)
    parser.add_argument('--training.lr_scheduler.gamma', default=None, type=float)
    parser.add_argument('--training.lr_scheduler.patience', default=None, type=float)



    parser.add_argument(
        '--preprocessing.dataloader.batch_size',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.dataloader.shuffle',
        default=None,
        type=str)
    parser.add_argument(
        '--preprocessing.dataloader.num_workers',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.dataloader.prefetch_factor',
        default=None,
        type=int)

    parser.add_argument(
        '--preprocessing.save_preprocessed_image.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.save_preprocessed_image.frequency',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.crop_size.x',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.crop_size.y',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.random_bounding_box.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.val_scale',
        default=None,
        type=float)
    parser.add_argument(
        '--preprocessing.transformation.scale_jitter.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.scale_jitter.normal_distribution.mean',
        default=None,
        type=float)
    parser.add_argument(
        '--preprocessing.transformation.scale_jitter.normal_distribution.std',
        default=None,
        type=float)
    parser.add_argument(
        '--preprocessing.transformation.angle_jitter.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.angle_jitter.normal_distribution.mean',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.angle_jitter.normal_distribution.std',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.shift_jitter.enabled',
        default=None,
        type=float)
    parser.add_argument(
        '--preprocessing.transformation.shift_jitter.normal_distribution.mean',
        default=None,
        type=float)
    parser.add_argument(
        '--preprocessing.transformation.shift_jitter.normal_distribution.std',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.mirror.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.mirror.probability',
        default=None,
        type=float)
    parser.add_argument(
        '--preprocessing.transformation.gaussian_blur.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.gaussian_blur.normal_distribution.mean',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.gaussian_blur.normal_distribution.std',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.gamma.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.gamma.normal_distribution.mean',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.gamma.normal_distribution.std',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.temperature.enabled',
        default=None,
        type=int)
    parser.add_argument(
        '--preprocessing.transformation.normalization',
        default=None,
        type=int)

    args = parser.parse_args()
    return args


def _replace_value_in_config(config, argument, argument_value):
    """
    Replaces a value in the DotMap
    :param config: Configuration DotMap
    :param argument: Argument to overwrite
    :param argument_value: Argument value
    :return: new DotMap with new Values
    """

    # Recursive Help function which creates a nested dict
    def _create_nested_dict(key, value):
        value = {key[-1]: value}
        new_key_list = key[0:-1]
        if len(new_key_list) > 1:
            return _create_nested_dict(new_key_list, value)
        return {new_key_list[0]: value}

    # Recursive Help function which updates a value
    def _update(key, value):
        for k, val in value.items():
            if isinstance(val, collections.abc.Mapping):
                key[k] = _update(key.get(k, {}), val)
            else:
                key[k] = val
        return key

    argument_keys = argument.split('.')
    new_dict = _create_nested_dict(argument_keys, argument_value)
    return DotMap(_update(config.toDict(), new_dict))





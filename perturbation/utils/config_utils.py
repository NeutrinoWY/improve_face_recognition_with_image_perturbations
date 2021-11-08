import os
import argparse
import collections
from datetime import datetime
import yaml
from dotmap import DotMap


def get_config(config_name="perturbation"):
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
    with open("perturbation/config/" + config_name + ".yml", "r") as file:
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
    parser = argparse.ArgumentParser(description='Arguments for perturbation on test stage')
    parser.add_argument('--config.name', default='perturbation', type=str)

    parser.add_argument('--basic.bounding_box_mode', default=None, type=int)
    parser.add_argument('--basic.features_path', default=None, type=str)
    parser.add_argument('--basic.train_pairs_path', default=None, type=str)
    parser.add_argument('--basic.test_pairs_path', default=None,type=str)

    parser.add_argument('--load_model.AFFFE_model_path', default=None, type=str)
    parser.add_argument('--load_model.AFFFE_state_path', default=None, type=str)
    parser.add_argument('--load_model.checkpoint', default=None, type=str)

    parser.add_argument('--extractor_name', default=None, type=str)

    parser.add_argument('--result.perturbations_save_path', default=None, type=str)
    parser.add_argument('--result.results_save_path', default=None, type=str)
    parser.add_argument('--result.merged_results_path', default=None, type=str)
    parser.add_argument('--result.greedy_save_path', default=None, type=str)
    parser.add_argument('--result.simpleGA_save_path', default=None, type=str)

    parser.add_argument('--simpleGA.max_iteration', default=None, type=int)
    #parser.add_argument('--simpleGA.random_selection', default=None, type=int)
    parser.add_argument('--simpleGA.elite_selection', default=None, type=int)
    parser.add_argument('--simpleGA.weight_initialization.method', default=None, type=str)
    parser.add_argument('--simpleGA.weight_initialization.Gaussian_mean', default=None, type=float)
    parser.add_argument('--simpleGA.fitness_score', default=None, type=str)
    parser.add_argument('--simpleGA.generation_replacement.method', default=None, type=str)
    parser.add_argument('--simpleGA.generation_replacement.percentage', default=None, type=float)

    parser.add_argument('--scale_range.min_scale', default=None, type=float)
    parser.add_argument('--scale_range.max_scale', default=None, type=float)

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



    
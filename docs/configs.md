# Configs

We use a global dictionary hparams provided by the Configs module for convenient hyperparameter configuration

## The directory tree

The Configs directory is structured as follows:

config_base：Save the basic configuration of all algorithms, such as hidden_dim, whether to use skip-connection, etc.

- - <algorithm_name>.yaml . Important hyper-params include:
    - algorithm_path：it specifies the algorithm corresponds to the Agent class in the agents module
    - trainer_path：it specifies which Trainer class from the trainers module the algorithm uses for training
- scenarios：It save the basic configuration of all scenarios, as well as the special configuration of each algorithm in that scenario
  - <scenario_name>：Create subfolders to store configuration files for corresponding scenes
    - env.yaml：It saves the basic configuration of the scene, such as the number of agents, the size of the map, the length of the training episode, etc.
    - <algorithm_name>.yaml：Save the special configuration of the algorithm for this scene

## Inheritance and Override of Configuration

During actual training, we call the configuration file of configs/scenarios/<scenario_name>/<algorithm_name>.yaml to train the <algorithm_name> algorithm in the <scenario_name> scene.

We provide the base_configs in the yaml file for inheritance, and the code will automatically crawl the configuration file specified by base_config as the default configuration. Later, you can reset some specific configuration values in the yaml below the base_configs to overrides.

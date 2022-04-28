# Configs模块

我们使用Configs模块提供的一个全局字典hparams来进行便捷的超参数配置。

## 文件夹结构

Configs模块的分层如下：

- config_base：保存所有算法的基本配置，如hidden_dim、是否使用skip-connection等
  - <algorithm_name>.yaml 具体保存了对应算法的配置，重要配置参数有
    - algorithm_path：指定该算法的Agent对应的是agents模块中的那个Agent类
    - trainer_path：指定该算法使用trainers模块中的哪个Trainer类来训练
- scenarios：保存所有场景的基本配置，以及各个算法在该场景中的特殊配置
  - <scenario_name>：创建子文件夹来储存对应场景的配置文件
    - env.yaml：保存该场景的基本配置，如智能体的数量、地图的大小、训练episode的长度等
    - <algorithm_name>.yaml：保存该算法在该场景的特殊配置

## 配置信息的继承和覆盖

我们在实际训练时，调用的是configs/scenarios/<scenario_name>/<algorithm_name>.yaml的配置文件，以实现在<scenario_name>场景里训练<algorithm_name>算法。

我们在yaml文件里提供了base_configs的功能，代码会自动爬取base_config指定的配置文件，作为预设配置。随后，你可以在该yaml里重新设定某些特定配置的值，实现覆盖。
# 编写异构MADRL算法和场景的注意事项

异构的MADRL算法，异构二字，指的是我们通过Hierarchical Graph Attention (HGAT) Layer实现了不同group (也就是异构的) 之间智能体的通信，从而在算法层面上实现了异构智能体之间的协作。

一般的同构MADRL算法 (如DQN、DGN、Soft-DRGN等) ，往往应用在只存在同构智能体的场景（比如Surviving、UAV-MBS等），因此只需要在Group内部进行通信即可，这时候Graph Attention Layer就可以满足需求。然而，对存在异构智能体的场景（如Cooperative Treasure Collection，CTC），往往需要在异构的智能体之间进行协作才能实现较好的性能，因此实现跨Group的智能体间通信是很有必要的。

为了实现跨Group的智能体间通信，我们采用了HAMA（AAAI 2021）中提出的Hierarchical Graph Attention（HGAT） Layer，将通信分为两个阶段：组内通信 (Intra-Group) 阶段和组间通信 (Inter-Group) 阶段。其中组内通信使用Graph Attention，组间通信则是直接将每个Group的embedding进行拼接。具体HGAT的理论可以去看HAMA论文或者我的[Soft-HGRNs论文(preprint)](https://www.researchgate.net/publication/354400471_Soft_Hierarchical_Graph_Recurrent_Networks_for_Many-Agent_Partially_Observable_Environments)。

## 基础模块：HGAT Layer

下面给出一张HGAT Layer的具体流程图，我们的代码（具体实现可以见modules/hgn.HGATLayer类）也是这么实现的：

![image-20211129141722201](imgs/HGAT_layer.png)

我们将上面的HGAT Layer用Pytorch进行了封装，大家只需要直接调用HGAT的forward接口就好。具体来说，HGAT Layer的forward分为了以下两个步骤：inner_group_forward()和inter_group_forward()，这些我们都在内部做了比较智能的识别与实现。

总的来说，HGAT Layer实现了所有Groupwise通信的功能，是构建HGN等异构MADRL算法的基石，因此也不需要做什么修改，大家只需要知道调用HGAT.forward(obs_lst, mask_lst, groupwise_connection)就可以啦。

## 网络结构：HGNNetwork

接下来我们展示如何基于HGATLayer写一个HGN网络结构。我们以CTC场景为例，有3类智能体。我们给所有智能体都提供encoder和linear regressor，所有类的encoder和regressor可以存到nn.ModuleList里面：

```python
self.encoders = nn.ModuleList()
self.linears = nn.ModuleList()
for idx in range(self.num_groups):
    self.encoders.append(nn.Linear(self.in_dim[idx], self.hidden_dim))
    if self.skip_connect:
        self.linears.append(nn.Linear(3 * self.hidden_dim, self.action_dim[idx]))
    else:
        self.linears.append(nn.Linear(self.hidden_dim, self.action_dim[idx]))
```

具体的实现可以见modules/hgn/HGNNetwork类。

注意这里HGNNetwork与其同构变种DGNNetwork有一点区别，即它需要提供一个init_from_groupwise_connection()函数，它可以在训练开始之前从输入的sample中推断出当前环境中每个group之间可通信情况，进而初始化对应的HGATLayer。一般来说，只要大家写的新Network继承了这里的HGNNetwork，就不用自己实现啦。

## Agent类：HGNAgent

与一般的同构算法的Agent不同，我们将所有group的智能体的Network定义在了一个Network类里面，相应的，我们也将所有group的智能体与环境交互、以及计算loss定义在了一个Agent类里面。

具体来说很简单，不过是遍历了一遍所有的group，依次计算action，或者依次计算loss，返回给trainer，就结束了。

如果想看实例，可以参见agents/hgn/HGNAgent类。

## 数据格式约定：Agent类与环境的交互、与Network的交互

#### 环境类输出数据的格式

数据格式约定是值得注意的。在异构环境中，我们有多个group的智能体，其中每个group中智能体的obs等参数可以被存成一个tensor。因此，考虑CTC这个具有3类智能体的场景，我们将每个group按照0到K进行标号（具体怎么标由环境类自己定义），然后存储信息为如下格式：

```python
# 这里只给出obs的存储方式，其他诸如rew、act、next_obs、done等同理。
obs = {}
obs['obs_0'], obs['obs_1'], obs['obs_2'] = Tensor(shape=[n_agent_0, obs_dim_0]),  Tensor(shape=[n_agent_1, obs_dim_1]),  Tensor(shape=[n_agent_2, obs_dim_2])
```

下面给出adj的存储方式，如果3个group每个group都能从其他group获得信息，那么理应存如3*3=9个邻接矩阵进入sample字典中。如果i号group不能从j号group获得信息，那么只需要在sample不存入key为'adj_i_j'的任何信息就好，我们的HGAT会自动爬取这个信息，就不会定义从groupi向groupj获取信息的网络结构了。

```python
# 为了节省空间我们只给出一个例子，意思是group0是否可以从group1获得信息
adj = {}
adj['adj_0_1'] = Tensor(shape=[n_agent_0,n_agent_1]) 
```

#### Agent处理环境类的数据

经过replay buffer采样，我们得到的sample其实是上面定义的obs、adj等字典的并集，即sample这个字典里面既存有'obs_0'这样的obs数据，也存有‘adj_0_1’这样的adj数据。因此，Agent使用如下代码来从sample字典中提取需要的信息，并将obs、adj等数据转换为列表，再输入到Network中（转为列表的好处是我们底层的代码可以直接用group的序号来对对应的数据进行索引，从而大大提升运算效率）。

```python
def _parse_adj_dict_to_lst(self, sample, prefix):
        adj_lst = [[None for _ in range(self.num_group)] for _ in range(self.num_group)]
        for i in range(self.num_group):
            for j in self.groupwise_connection[i]:
                # adj_name为'adj_0_1'或'next_adj_0_1'这种格式的数据
                adj_name = prefix + '_' + str(i) + '_' + str(j)
                adj_lst[i][j] = sample[adj_name]
        return adj_lst

def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        obs = [sample['obs_' + str(i)] for i in range(self.num_group)]
        adj = self._parse_adj_dict_to_lst(sample, prefix='adj')
        act = [sample['act_' + str(i)] for i in range(self.num_group)]
        rew = [sample['rew_' + str(i)] for i in range(self.num_group)]
        done = [sample['done_' + str(i)] for i in range(self.num_group)]
        next_obs = [sample['next_obs_' + str(i)] for i in range(self.num_group)]
        next_adj = self._parse_adj_dict_to_lst(sample, prefix='next_adj')
```

可以看到对各变量的命名规则沿用了上一小节中环境输出数据格式的约定。

## Trainer类：HeteroValueBasedTrainer

由于我们之前的Trainer写的非常鲁棒，所以其实相比其基类ValueBasedTrainer没什么变化，只是根据异构场景的需求，增加了记录每个group的reward等指标的功能。理论上本Trainer也可以兼容HGRN算法。要实现Soft-HGRN可能还需要稍微修改一下本Trainer里记录alpha参数方面的地方，但是改动量也不大。


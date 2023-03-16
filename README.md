问题：
- DQN在小数据集上快速收敛----BCQ实现有问题
- reward.cumsum()的使用
- offline rl方面离散问题并不是很难处理？

TODO:
- 收集训练过程中的loss变化(完成)
- log名自动反映实验设置
- 需要统计每个回合累计奖励情况
- 使用hydra
- 理解多线程收集数据代码


观察到的现象：
- env不加seed, 每次求解情况不一样；根据SB score选分数和Configure不完全一样

数据集目录: "/home/chengpeng/chengpeng_BCQ_co/chengpeng_BCQ_co/data_generation"
- buffers_100:  LP迭代次数作为reward，100个数据，一般用于测试代码是否可行；
- buffers_100300: LP迭代次数作为reward,100300个数据；
- nodebuffers_10000：node作为reward，1W个数据；
- timebuffers_10000: solvingtime作为reward，1W个数据；
- nodemixedbuffers_10000: node作为reward，50%的专家数据，50%的弱专家数据，1W个数据；

https://shimo.im/sheets/QPtQtjYrYtpdrWPg/MODOC
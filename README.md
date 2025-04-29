# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/murtazahr/murmura/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                        |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| murmura/\_\_init\_\_.py                                                     |        0 |        0 |    100% |           |
| murmura/aggregation/aggregation\_config.py                                  |       25 |        0 |    100% |           |
| murmura/aggregation/coordination\_mode.py                                   |        4 |        0 |    100% |           |
| murmura/aggregation/strategies/fed\_avg.py                                  |       26 |        0 |    100% |           |
| murmura/aggregation/strategies/gossip\_avg.py                               |       28 |        0 |    100% |           |
| murmura/aggregation/strategies/trimmed\_mean.py                             |       41 |        4 |     90% |77-78, 103-104 |
| murmura/aggregation/strategy\_factory.py                                    |       27 |        0 |    100% |           |
| murmura/aggregation/strategy\_interface.py                                  |        8 |        1 |     88% |        30 |
| murmura/data\_processing/dataset.py                                         |      114 |       13 |     89% |129, 143, 201-206, 217-225 |
| murmura/data\_processing/partitioner.py                                     |       84 |        7 |     92% |32, 80, 123-124, 133-134, 150 |
| murmura/data\_processing/partitioner\_factory.py                            |       10 |        1 |     90% |        32 |
| murmura/model/model\_interface.py                                           |       25 |        7 |     72% |24, 39, 51, 60, 69, 78, 87 |
| murmura/model/pytorch\_model.py                                             |      102 |        1 |     99% |       143 |
| murmura/network\_management/topology.py                                     |       24 |        0 |    100% |           |
| murmura/network\_management/topology\_compatibility.py                      |       20 |        0 |    100% |           |
| murmura/network\_management/topology\_manager.py                            |       23 |        0 |    100% |           |
| murmura/node/client\_actor.py                                               |       74 |       31 |     58% |84-107, 116-120, 129-133, 146-159, 167-169, 177-179 |
| murmura/orchestration/cluster\_manager.py                                   |      110 |       46 |     58% |55, 66-69, 115-116, 132-139, 148-151, 160-163, 186, 199-220, 228-229, 236, 252, 272-290, 296-299 |
| murmura/orchestration/learning\_process/decentralized\_learning\_process.py |       67 |        1 |     99% |       121 |
| murmura/orchestration/learning\_process/federated\_learning\_process.py     |       67 |        3 |     96% |   118-123 |
| murmura/orchestration/learning\_process/learning\_process.py                |       73 |        3 |     96% |81-82, 118 |
| murmura/orchestration/orchestration\_config.py                              |       14 |        0 |    100% |           |
| murmura/orchestration/topology\_coordinator.py                              |      156 |       48 |     69% |52, 66-92, 105-148, 193, 205, 249, 261, 303, 315, 341, 381 |
| murmura/visualization/network\_visualizer.py                                |      356 |       83 |     77% |71, 124-128, 152, 205-206, 233-235, 281-282, 297-298, 314, 332, 336-340, 347, 351, 366-376, 384-389, 393-398, 404, 408, 466-468, 512, 526-565, 568-569, 599-600, 677-678, 708-709 |
| murmura/visualization/training\_event.py                                    |       38 |        0 |    100% |           |
| murmura/visualization/training\_observer.py                                 |       17 |        1 |     94% |        17 |
|                                                                   **TOTAL** | **1533** |  **250** | **84%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/murtazahr/murmura/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/murtazahr/murmura/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/murtazahr/murmura/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/murtazahr/murmura/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmurtazahr%2Fmurmura%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/murtazahr/murmura/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.
# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/murtazahr/murmura/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                             |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------- | -------: | -------: | ------: | --------: |
| murmura/\_\_init\_\_.py                          |        0 |        0 |    100% |           |
| murmura/aggregation/aggregation\_config.py       |       19 |        4 |     79% |     42-45 |
| murmura/aggregation/strategies/fed\_avg.py       |       22 |       17 |     23% |     28-54 |
| murmura/aggregation/strategies/trimmed\_mean.py  |       39 |       31 |     21% |22-27, 42-77, 86-103 |
| murmura/aggregation/strategy\_factory.py         |       15 |        8 |     47% |     26-35 |
| murmura/aggregation/strategy\_interface.py       |        6 |        1 |     83% |        26 |
| murmura/data\_processing/dataset.py              |      101 |        2 |     98% |  129, 143 |
| murmura/data\_processing/partitioner.py          |       84 |        7 |     92% |32, 80, 123-124, 133-134, 150 |
| murmura/data\_processing/partitioner\_factory.py |       10 |        1 |     90% |        32 |
| murmura/helper.py                                |       18 |        0 |    100% |           |
| murmura/model/model\_interface.py                |       25 |        7 |     72% |24, 39, 51, 60, 69, 78, 87 |
| murmura/network\_management/topology.py          |       24 |        0 |    100% |           |
| murmura/network\_management/topology\_manager.py |       23 |        0 |    100% |           |
| murmura/node/client\_actor.py                    |       74 |       39 |     47% |51-55, 63, 84-107, 116-120, 129-133, 146-159, 167-169, 177-179, 195, 203 |
| murmura/orchestration/cluster\_manager.py        |       71 |       25 |     65% |27, 53, 97-98, 114-121, 130-133, 142-145, 156-168, 176, 184-185, 192 |
| murmura/orchestration/orchestration\_config.py   |       14 |        0 |    100% |           |
|                                        **TOTAL** |  **545** |  **142** | **74%** |           |


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
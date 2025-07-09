# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Cloudslab/murmura/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                        |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| murmura/\_\_init\_\_.py                                                     |        0 |        0 |    100% |           |
| murmura/aggregation/\_\_init\_\_.py                                         |        0 |        0 |    100% |           |
| murmura/aggregation/aggregation\_config.py                                  |       25 |        0 |    100% |           |
| murmura/aggregation/coordination\_mode.py                                   |        4 |        0 |    100% |           |
| murmura/aggregation/strategies/\_\_init\_\_.py                              |        0 |        0 |    100% |           |
| murmura/aggregation/strategies/fed\_avg.py                                  |       30 |        2 |     93% |    46, 65 |
| murmura/aggregation/strategies/gossip\_avg.py                               |       32 |        2 |     94% |    56, 76 |
| murmura/aggregation/strategies/trimmed\_mean.py                             |       47 |        7 |     85% |54, 77, 96-97, 118, 132-133 |
| murmura/aggregation/strategy\_factory.py                                    |       27 |        0 |    100% |           |
| murmura/aggregation/strategy\_interface.py                                  |        8 |        1 |     88% |        34 |
| murmura/attacks/\_\_init\_\_.py                                             |        0 |        0 |    100% |           |
| murmura/attacks/scalability\_simulator.py                                   |      533 |      533 |      0% |   10-1274 |
| murmura/attacks/scalability\_simulator\_parallel.py                         |      134 |      134 |      0% |     6-313 |
| murmura/attacks/topology\_attacks.py                                        |      312 |      312 |      0% |     6-748 |
| murmura/data\_processing/\_\_init\_\_.py                                    |        0 |        0 |    100% |           |
| murmura/data\_processing/attack\_partitioner.py                             |      159 |      142 |     11% |29-32, 36-45, 51-93, 116-118, 124-140, 149-184, 188-205, 209, 226-229, 235-329 |
| murmura/data\_processing/data\_preprocessor.py                              |      196 |       12 |     94% |17, 22, 163, 278, 326-327, 355-357, 464-467, 481 |
| murmura/data\_processing/dataset.py                                         |      238 |       80 |     66% |52, 59, 66-68, 75-109, 116-123, 143-159, 266, 280, 344-349, 360-368, 375-389, 409, 433-435, 445-448, 468-495, 499-500, 508 |
| murmura/data\_processing/partitioner.py                                     |       84 |        7 |     92% |32, 80, 123-124, 133-134, 150 |
| murmura/data\_processing/partitioner\_factory.py                            |       44 |       28 |     36% |36-43, 52-93, 111-120, 131-156 |
| murmura/examples/\_\_init\_\_.py                                            |        0 |        0 |    100% |           |
| murmura/examples/dp\_decentralized\_ham10000\_example.py                    |      215 |      215 |      0% |     2-650 |
| murmura/examples/dp\_decentralized\_mnist\_example.py                       |      274 |      274 |      0% |     1-802 |
| murmura/examples/dp\_ham10000\_example.py                                   |      200 |      200 |      0% |     2-602 |
| murmura/examples/dp\_mnist\_example.py                                      |      184 |      184 |      0% |     1-532 |
| murmura/model/\_\_init\_\_.py                                               |        0 |        0 |    100% |           |
| murmura/model/model\_interface.py                                           |       25 |        7 |     72% |24, 39, 51, 60, 69, 78, 87 |
| murmura/model/pytorch\_model.py                                             |      182 |       42 |     77% |69-73, 102-107, 111, 115, 123-131, 135, 137, 149-152, 170-201, 279, 378, 389, 412 |
| murmura/models/\_\_init\_\_.py                                              |        3 |        0 |    100% |           |
| murmura/models/ham10000\_models.py                                          |      101 |        0 |    100% |           |
| murmura/models/mnist\_models.py                                             |       30 |        0 |    100% |           |
| murmura/network\_management/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| murmura/network\_management/topology.py                                     |       24 |        0 |    100% |           |
| murmura/network\_management/topology\_compatibility.py                      |       20 |        0 |    100% |           |
| murmura/network\_management/topology\_manager.py                            |       23 |        0 |    100% |           |
| murmura/node/\_\_init\_\_.py                                                |        0 |        0 |    100% |           |
| murmura/node/client\_actor.py                                               |      506 |      396 |     22% |17-18, 32-33, 41-42, 112, 152-212, 232-295, 304, 327-328, 336-549, 558-627, 641-838, 849-905, 914-939, 952-978, 986-995, 1003-1011, 1033-1035, 1055, 1075-1077, 1113-1117, 1127-1129 |
| murmura/node/resource\_config.py                                            |       16 |        0 |    100% |           |
| murmura/orchestration/\_\_init\_\_.py                                       |        0 |        0 |    100% |           |
| murmura/orchestration/cluster\_manager.py                                   |      598 |      429 |     28% |79-111, 138-140, 157-161, 178-200, 204, 209-262, 266, 277-284, 314, 325-327, 332, 348-349, 356-359, 412-416, 427-459, 470-598, 608-707, 713-850, 862-897, 912-964, 977-1011, 1015-1069, 1089-1131, 1135-1157, 1171, 1184-1244, 1250-1270, 1274-1289, 1294, 1310-1312, 1324, 1359-1362, 1378-1397, 1406-1407 |
| murmura/orchestration/learning\_process/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| murmura/orchestration/learning\_process/decentralized\_learning\_process.py |      130 |       29 |     78% |30, 47-75, 115, 153-154, 160-170, 345 |
| murmura/orchestration/learning\_process/federated\_learning\_process.py     |      123 |       32 |     74% |39, 52, 59-91, 134, 173-174, 180-190, 257-262 |
| murmura/orchestration/learning\_process/learning\_process.py                |      328 |      139 |     58% |44-45, 120-121, 149, 154, 178-192, 198-218, 258-260, 268-271, 319-332, 477, 498-499, 520-522, 532-542, 564, 572, 579, 612-662, 668-723, 732, 744-745, 755-795, 886-919 |
| murmura/orchestration/orchestration\_config.py                              |       62 |        0 |    100% |           |
| murmura/orchestration/topology\_coordinator.py                              |      175 |       61 |     65% |42, 61, 89-115, 131-184, 231, 243, 274, 298, 311, 340-357, 382, 395, 421, 461 |
| murmura/privacy/\_\_init\_\_.py                                             |        0 |        0 |    100% |           |
| murmura/privacy/dp\_aggregation.py                                          |      151 |       10 |     93% |136-137, 194, 262-263, 329, 393-394, 415-416 |
| murmura/privacy/dp\_config.py                                               |       72 |        8 |     89% |137, 140, 145, 258-263 |
| murmura/privacy/dp\_model\_wrapper.py                                       |      233 |       55 |     76% |20-22, 90, 107, 114-116, 121-122, 144-150, 154-155, 210-215, 218, 242, 252-255, 274, 299, 324-331, 344-352, 440-442, 477-481, 500-501, 506, 520-527, 556, 569, 582 |
| murmura/privacy/privacy\_accountant.py                                      |      148 |       21 |     86% |15-17, 102, 109-111, 163-166, 174-184, 452-467 |
| murmura/visualization/\_\_init\_\_.py                                       |        0 |        0 |    100% |           |
| murmura/visualization/network\_visualizer.py                                |      577 |      170 |     71% |93, 117-141, 258-274, 319, 490, 570-621, 625-640, 644-666, 670-681, 685-702, 706-723, 741-742, 769-771, 817-818, 833-834, 850, 868, 872-876, 883, 887, 902-912, 920-925, 929-934, 940, 944, 1005-1007, 1051, 1065-1104, 1107-1108, 1141-1142, 1219-1220, 1250-1251 |
| murmura/visualization/training\_event.py                                    |       93 |       38 |     59% |164-181, 187-210, 215-248 |
| murmura/visualization/training\_observer.py                                 |       18 |        1 |     94% |        17 |
|                                                                   **TOTAL** | **6384** | **3571** | **44%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/Cloudslab/murmura/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/Cloudslab/murmura/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Cloudslab/murmura/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/Cloudslab/murmura/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FCloudslab%2Fmurmura%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/Cloudslab/murmura/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.
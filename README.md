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
| murmura/data\_processing/\_\_init\_\_.py                                    |        0 |        0 |    100% |           |
| murmura/data\_processing/data\_preprocessor.py                              |      196 |       12 |     94% |17, 22, 163, 278, 326-327, 355-357, 464-467, 481 |
| murmura/data\_processing/dataset.py                                         |      238 |       80 |     66% |52, 59, 66-68, 75-109, 116-123, 143-159, 266, 280, 344-349, 360-368, 375-389, 409, 433-435, 445-448, 468-495, 499-500, 508 |
| murmura/data\_processing/partitioner.py                                     |       84 |        7 |     92% |32, 80, 123-124, 133-134, 150 |
| murmura/data\_processing/partitioner\_factory.py                            |       10 |        1 |     90% |        32 |
| murmura/examples/\_\_init\_\_.py                                            |        0 |        0 |    100% |           |
| murmura/examples/decentralized\_mnist\_example.py                           |      187 |      187 |      0% |     1-562 |
| murmura/examples/decentralized\_skin\_lesion\_example.py                    |      288 |      288 |      0% |    10-853 |
| murmura/examples/dp\_decentralized\_mnist\_example.py                       |      271 |      271 |      0% |     1-787 |
| murmura/examples/dp\_decentralized\_skin\_lesion\_example.py                |      347 |      347 |      0% |   11-1016 |
| murmura/examples/dp\_mnist\_example.py                                      |      181 |      181 |      0% |     1-517 |
| murmura/examples/dp\_skin\_lesion\_example.py                               |      235 |      235 |      0% |     1-642 |
| murmura/examples/mnist\_example.py                                          |      155 |      155 |      0% |     1-488 |
| murmura/examples/skin\_lesion\_example.py                                   |      267 |      267 |      0% |     1-786 |
| murmura/model/\_\_init\_\_.py                                               |        0 |        0 |    100% |           |
| murmura/model/model\_interface.py                                           |       25 |        7 |     72% |24, 39, 51, 60, 69, 78, 87 |
| murmura/model/pytorch\_model.py                                             |      182 |       42 |     77% |69-73, 102-107, 111, 115, 123-131, 135, 137, 149-152, 170-201, 279, 378, 389, 412 |
| murmura/models/\_\_init\_\_.py                                              |        0 |        0 |    100% |           |
| murmura/models/mnist\_models.py                                             |       30 |        0 |    100% |           |
| murmura/models/skin\_lesion\_models.py                                      |       98 |       98 |      0% |     1-237 |
| murmura/network\_management/\_\_init\_\_.py                                 |        0 |        0 |    100% |           |
| murmura/network\_management/topology.py                                     |       24 |        0 |    100% |           |
| murmura/network\_management/topology\_compatibility.py                      |       20 |        0 |    100% |           |
| murmura/network\_management/topology\_manager.py                            |       23 |        0 |    100% |           |
| murmura/node/\_\_init\_\_.py                                                |        0 |        0 |    100% |           |
| murmura/node/client\_actor.py                                               |      506 |      413 |     18% |17-18, 32-33, 41-42, 112, 152-212, 232-295, 304, 327-328, 336-549, 558-627, 641-838, 849-905, 914-939, 952-978, 986-995, 1003-1011, 1031-1033, 1049-1075, 1083-1127 |
| murmura/node/resource\_config.py                                            |       16 |        0 |    100% |           |
| murmura/orchestration/\_\_init\_\_.py                                       |        0 |        0 |    100% |           |
| murmura/orchestration/cluster\_manager.py                                   |      550 |      392 |     29% |87, 91, 100-101, 109-111, 138-140, 157-161, 178-181, 187, 192-235, 239, 271, 282-284, 289, 305-306, 313-316, 369-373, 384-416, 427-555, 565-664, 670-805, 817-852, 867-919, 932-966, 970-1024, 1044-1086, 1090-1112, 1126, 1132-1152, 1156-1171, 1176, 1187-1189, 1201, 1216-1232, 1236-1239, 1245, 1255-1274, 1283-1284 |
| murmura/orchestration/learning\_process/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| murmura/orchestration/learning\_process/decentralized\_learning\_process.py |      120 |       28 |     77% |30, 47-75, 115, 153-154, 160-170 |
| murmura/orchestration/learning\_process/federated\_learning\_process.py     |      123 |       32 |     74% |39, 52, 59-91, 134, 173-174, 180-190, 257-262 |
| murmura/orchestration/learning\_process/learning\_process.py                |      251 |      120 |     52% |41-42, 111-112, 140, 145, 169-183, 189-209, 220, 241-242, 263-265, 275-285, 307, 315, 322, 355-405, 411-466, 475, 487-488, 498-538, 629-662 |
| murmura/orchestration/orchestration\_config.py                              |       62 |        0 |    100% |           |
| murmura/orchestration/topology\_coordinator.py                              |      175 |       61 |     65% |42, 61, 89-115, 131-181, 226, 238, 269, 291, 303, 332-349, 372, 384, 410, 450 |
| murmura/privacy/\_\_init\_\_.py                                             |        0 |        0 |    100% |           |
| murmura/privacy/dp\_aggregation.py                                          |      151 |      151 |      0% |     1-418 |
| murmura/privacy/dp\_config.py                                               |       72 |        8 |     89% |137, 140, 145, 258-263 |
| murmura/privacy/dp\_model\_wrapper.py                                       |      227 |       52 |     77% |20-22, 100, 107-109, 130-136, 140-141, 196-201, 204, 228, 238-241, 260, 285, 310-317, 330-338, 426-428, 463-467, 486-487, 492, 506-513, 542, 555, 568 |
| murmura/privacy/privacy\_accountant.py                                      |      148 |       21 |     86% |15-17, 102, 109-111, 163-166, 174-184, 452-467 |
| murmura/visualization/\_\_init\_\_.py                                       |        0 |        0 |    100% |           |
| murmura/visualization/network\_visualizer.py                                |      474 |       86 |     82% |81, 194-210, 255, 423, 508-509, 536-538, 584-585, 600-601, 617, 635, 639-643, 650, 654, 669-679, 687-692, 696-701, 707, 711, 772-774, 818, 832-871, 874-875, 908-909, 986-987, 1017-1018 |
| murmura/visualization/training\_event.py                                    |       40 |        0 |    100% |           |
| murmura/visualization/training\_observer.py                                 |       18 |        1 |     94% |        17 |
|                                                                   **TOTAL** | **5967** | **3555** | **40%** |           |


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
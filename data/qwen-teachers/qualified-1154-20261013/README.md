# 1154 张合格 Qwen 老师原生图

第四轮封存审核选定 1154 张合格版本；另 96 张继续迭代。划分：{'train': 1058, 'test': 48, 'validation': 48}，沿用来源划分。

images/ 保存原生 PNG 原字节；manifest、teachers JSONL/CSV 记录来源 UID/ID、来源与结果 SHA、版本、提示词、种子、配置、许可、作者和审核理由。generation-configs/ 与 provenance/ 保留参数和审核历史。sources 条目通过 source_archive_member 对应陈经完整 dataset.zip，源包 SHA：48aea188795570eb498d96a2bfc4ac47ba348872b24d59f2e530cb596643ec52。

逐项结果 SHA、完整解码及 ZIP CRC/inventory SHA 通过。轻微伪影记录见 artifact_presence/artifact_impact。审核目标仍为全 1250 张中重做至多 49 张；本包为用户授权提前发布的合格子集。

[下载完整结果包](https://github.com/Teddytututu/Perler_Generator/releases/download/qwen-teachers-1154-20261013/Qwen-Chen-qualified-teachers-1154-20261013.zip)

# V3 原生设备工程测速

设备：mps，批量4，序列512，FP32，排除2次预热，计时5次。

| 条件 | 参数 | 更新秒数中位数 | 输入token/s | 相对局部Transformer耗时 | 持久状态字节 |
|---|---:|---:|---:|---:|---:|
| trans_local | 333106 | 0.020275 | 101009 | 1.00 | 459328 |
| trans_full | 333106 | 0.018915 | 108275 | 0.93 | 1837120 |
| v3_static | 333136 | 0.315959 | 6482 | 15.58 | 360480 |
| v3_dynamic | 333136 | 0.409689 | 4999 | 20.21 | 360480 |

这是随机输入的计算测速，无法说明语言或检索能力。持久状态不包含训练激活及优化器。
V3使用向量节点，容量与旧版完整矩阵不同；相对旧版计时不能视作等容量算法提速。

CPU/原生设备及流式一致性：{'status': 'completed', 'config': {'vocab_size': 32, 'width': 16, 'layers': 2, 'heads': 2, 'ff': 32, 'nodes': 4, 'phase_bands': 3, 'segment_tokens': 4, 'context_limit': 128, 'dynamic_addresses': True}, 'logits_max_abs': 8.568167686462402e-08, 'gradients_max_abs': 1.1920928955078125e-07, 'gradient_tensors': 43, 'streaming_max_abs': 5.21540641784668e-08}
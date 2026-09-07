| cell (task/device/size/mode/phase) | off median | stream-only median | on median | stream-only/off | on/off | on/stream-only | checksum | on launch counters (captured/replayed/graph_launches/sgd_launches) |
|---|---|---|---|---|---|---|---|---|
| train/cuda/64/fresh/None | 520.3 us | 502.9 us | 496.9 us | 0.9665 | 0.9550 | 0.9880 | 完全一致 | 0/0/0/0 |
| train/cuda/64/reuse/None | 446.2 us | 430.3 us | 430.3 us | 0.9644 | 0.9644 | 1.0000 | 完全一致 | 2/98/100/2 |
| train_phases/cuda/64/reuse/backward | 197.2 us | 190.4 us | 190.3 us | 0.9657 | 0.9651 | 0.9994 | 完全一致 | 2/98/100/2 |
| train_phases/cuda/64/reuse/device_update | 90.9 us | 89.7 us | 88.4 us | 0.9866 | 0.9719 | 0.9851 | 完全一致 | 2/98/100/2 |
| train_phases/cuda/64/reuse/forward_resident | 159.2 us | 150.4 us | 149.6 us | 0.9448 | 0.9397 | 0.9947 | 完全一致 | 2/98/100/2 |
| train_phases/cuda/64/reuse/leaf_register | 0.1 us | 0.1 us | 0.1 us | 1.0000 | 1.0000 | 1.0000 | 完全一致 | 2/98/100/2 |
| train_phases/cuda/64/reuse/loss_readout | 0.0 us | 0.0 us | 0.0 us | 1.0000 | 1.0000 | 1.0000 | 完全一致 | 2/98/100/2 |
| train_phases/cuda/64/reuse/step_total | 451.1 us | 432.6 us | 430.9 us | 0.9589 | 0.9552 | 0.9962 | 完全一致 | 2/98/100/2 |
| train_phases/cuda/64/reuse/tape_build | 2.6 us | 2.8 us | 2.7 us | 1.0638 | 1.0395 | 0.9771 | 完全一致 | 2/98/100/2 |
| train_phases/cuda/64/reuse/tape_drop | 0.9 us | 0.9 us | 0.9 us | 0.9153 | 0.9322 | 1.0185 | 完全一致 | 2/98/100/2 |

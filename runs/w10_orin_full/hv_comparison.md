# HV comparison vs TetraRL (preference_ppo)

| Method | mean HV | std | n | p-value vs preference_ppo |
| --- | ---: | ---: | ---: | ---: |
| preference_ppo | 0.000022 | 0.000013 | 15 | - |
| duojoule | 0.000019 | 0.000007 | 15 | 0.4695 |
| envelope_morl | 0.000027 | 0.000010 | 15 | 0.2847 |
| focops | 0.000016 | 0.000014 | 15 | 0.2662 |
| max_action | 0.000054 | 0.000094 | 15 | 0.2061 |
| max_performance | 0.000054 | 0.000094 | 15 | 0.2061 |
| pcn | 0.000053 | 0.000074 | 15 | 0.1336 |
| ppo_lagrangian | 0.000019 | 0.000019 | 15 | 0.5573 |

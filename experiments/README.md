# Experiments

This folder contains code for the baslines models we ran on PACS, and additional code to run PACS-material. The results on PACS are as follows:

| **Model**                                                                                     | **With Audio (%)** | **Without Audio (%)** | **$\Delta$** |
|-----------------------------------------------------------------------------------------------|--------------------|-----------------------|--------------|
| Fusion (I+A+V)                                                                                | $51.9 \pm 1.1$     | -                     | -            |
| Fusion (Q+I)                                                                                  | -                  | $51.2 \pm 0.8$        | -            |
| Fusion (Q+A)                                                                                  | $50.9 \pm 0.6$     | -                     | -            |
| Fusion (Q+V)                                                                                  | -                  | $51.5 \pm 0.9$        | -            |
| Late Fusion                                                                                   | $55.0 \pm 1.1$     | $52.5\pm 1.6$         | 2.5          |
| [CLIP](https://github.com/openai/CLIP)/[AudioCLIP](https://github.com/AndreyGuzhov/AudioCLIP) | $60.0 \pm 0.9$     | $56.3 \pm 0.7$        | 3.7          |
| [UNITER](https://github.com/ChenRocks/UNITER) (L)                                             | -                  | $60.6 \pm 2.2$        | -            |
| [Merlot Reserve](https://github.com/rowanz/merlot_reserve) (B)                                | $66.5 \pm 1.4$     | $64.0 \pm 0.9$        | 2.6          |
| [Merlot Reserve](https://github.com/rowanz/merlot_reserve) (L)                                | $70.1 \pm 1.0$     | $68.4 \pm 0.7$        | 1.8          |
| Majority                                                                                      | 50.4               | 50.4                  | -            |
| Human                                                                                         | $96.3 \pm 2.1$     | $90.5 \pm 3.1$        | 5.9          |
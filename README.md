# SeeNet: Soft Emotion Expert with Pretrained model and Data Augmentation Method to Enhance Speech Emotion Recognition

> [Qifei Li](), [Yingming Gao](), [Yuhua Wen](), [Ziping Zhao](), [Ya Li]() and [Björn W. Schuller]()<br>
> School of Aritificial Intelligence, Beijing University of Posts and Telecommunications & Tianjin Normal University & GLAM – The Group on Language, Audio, and Music, Imperial College London<br>

## 📰 News
**[2024.06.13]** After the peer review process is completed, we will update the code. 


## ✨ Overview

![SeeNet](figs/framework.png)

Abstract:
Speech emotion recognition (SER) systems are designed to enable machines to recognize emotional states in human speech during human-computer interactions, enhancing the interactive experience. While considerable progress has been achieved in this field recently, the SER systems still encounter challenges related to performance and robustness, primarily stemming from the limitations of labeled data. In this end, we propose a novel multitask learning framework to learn distinctive and robust emotional representation, called ``Soft Emotion Expert Network (SeeNet)". The SeeNet consists of three components: pretrained model, auxiliary task soft emotion expert (SEE) module and the energy-based mixup (EBM) data augmentation module. The pretrained model and EBM module are employed to mitigate the challenges arising from limited labeled data, thereby enhancing model performance and bolstering robustness. The soft emotion expert module as auxiliary task is designed to assist the main task of emotion recognition to more efficiently enhance the distinction between samples exhibiting high similarity across categories to further improve the performance and robustness. To validate the effectiveness of our proposed method, we use different experimental setups to evaluate the performance and robustness of our method, such as within corpus, cross-corpus and the degree of noise immunity. The experimental results demonstrate that our proposed method surpasses the state-of-the-art (SOTA) methods in both performance and robustness.


## 🚀 Main Results

<p align="center">
  <img src="figs/performance.jpg" width=100%> <br>
   Comparison with state-of-the-art SER methods on 3 datasets.
</p>



## 🔨 Installation

Main prerequisites:

* `Python 3.8`
* `PyTorch`
* `transformers`
* `scikit-learn, scipy, pandas, numpy`
* `accelerate`
* `soundfile`
* `librosa`

If some are missing, please refer to [requirements.yml](requirements.ylm) for more details.

## 📍 Data Preparation
1. You should prepare training source, such as train.scp, valid.scp.

    Specifically, the format of `*.scp` file is typically like this:

    ```
    wave_index1 dataset_root/audio_1 label_1
    wave_index2 dataset_root/audio_2 label_2
    wave_index3 dataset_root/audio_3 label_3
    ...
    wave_indexN dataset_root/audio_N label_N
    ```

    An example of [train.scp] is shown as follows:

    ```
    Ses03F_impro01_F002 /home/samba/public/Datasets/IEMOCAP/IEMOCAP_full_release/Session3/sentences/wav/Ses03F_impro01/Ses03F_impro01_F002.wav neu
    Ses03F_impro01_M001 /home/samba/public/Datasets/IEMOCAP/IEMOCAP_full_release/Session3/sentences/wav/Ses03F_impro01/Ses03F_impro01_M001.wav neu
    Ses03F_impro01_M002 /home/samba/public/Datasets/IEMOCAP/IEMOCAP_full_release/Session3/sentences/wav/Ses03F_impro01/Ses03F_impro01_M002.wav neu
    Ses03F_impro01_M003 /home/samba/public/Datasets/IEMOCAP/IEMOCAP_full_release/Session3/sentences/wav/Ses03F_impro01/Ses03F_impro01_M003.wav neu
    ```

## ➡️ File Description

1. src/SeeNet.py. The file includes the dataset, training method, and the model of SeeNet. The file is for IEMOCAP and REVADESS.

  ```
    CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch SeeNet.py --train_src="/path/session1_train.scp" --valid_src="/path/session1_valid.scp --loss_rate=0.01"
  ```

2. src/SeeNetForMSP_IMPROVE.py. The file includes the dataset, training method, and the model of SeeNet. The file is for cross-dataset validation.

  ```
    CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch iemocap2msp.py --epochs=40 --train_src="/home/lqf/workspace/icassp2023/all_iemocap.scp" --valid_src="/home/lqf/workspace/wavlm-multi/part1_shuff_all_msp.scp" --test_src="/home/lqf/workspace/wavlm-multi/part2_shuff_all_msp.scp" --loss_rate=0.01 --is_augment
  ```

3. src/fine_tune_pretrained_model.py. The file includes the methods, which are used to fine-tune wav2vec2.0, HuBERT, WavLM and Data2vec.
  ```
    CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch SeeNet.py --max_length=6 --model="wavlm" --train_src="/path/session1_train.scp" --valid_src="/path/session1_valid.scp"
  ```

* `When you fine tune the Data2vec, the lr should be set as 0.00001.`


4. src/auxiliary_task_ablation_exp.py. The file is used to conduct comparative experiments on different auxiliary tasks.
  ```
    CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch auxiliary_task_ablation_exp.py --max_length=6 --epochs=30 --loss_rate=0.01 --num_class_a=4 --num_class_c=4 --train_src="/home/lqf/workspace/icassp2023/session1_train_info.scp" --valid_src="/home/lqf/workspace/icassp2023/session1_test_info.scp" --aux_task="SEE"
  ```

5. src/data_augmentation_ablation.py. The file is used to evalutae the performance different data augmentation method.

  ```
    CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch data_augmentation_ablation.py --train_src="/path/session1_train.scp" --valid_src="/path/session1_valid.scp --loss_rate=0 --is_augment=True --method_augment="SNR""
  ```

5. src/noise_exp.py. The file includes the methods, which are used to evaluate the noise immunity of SeeNet and its elements.

  ```
    CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch emotion_expert_version_02_msp.py --loss_rate=0.01 --db=10  --train_src="/home/lqf/workspace/wavlm-multi/session1_train.scp" --valid_src="/home/lqf/workspace/wavlm-multi/session1_test.scp"
  ```


## ☎️ Contact 

If you have any questions, please feel free to reach me out at `liqifei@bupt.edu.cn`.

## 👍 Acknowledgements

Thanks for the efforts of all the authors..

## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```
@article{li2025seenet,
  title={SeeNet: A Soft Emotion Expert and Data Augmentation Method to Enhance Speech Emotion Recognition},
  author={Li, Qifei and Gao, Yingming and Wen, Yuhua and Zhao, Ziping and Li, Ya and Schuller, Bjorn W},
  journal={IEEE Transactions on Affective Computing},
  number={01},
  pages={1--15},
  year={2025},
  publisher={IEEE Computer Society}
}
```
# A Lightweight Deep Autoencoder-based Approach for Unsupervised Anomaly Detection (Keras-Tensorflow Implementation)

This repository provides a Keras-Tensorflow implementation of the Intrusion detection using Autoencoder method presented in our paper ”A Lightweight Deep Autoencoder-based Approach for Unsupervised Anomaly Detection”.

# Citations and Contact.

You find a PDF of the **A Lightweight Deep Autoencoder-based Approach for Unsupervised Anomaly Detection** at: xxxx

If you use our work, please also cite the paper:

```
@article{xxxx,
  title={A Lightweight Deep Autoencoder-based Approach for Unsupervised Anomaly Detection},
  author={Gcinizwe Dlamini, Rufina Galieva and Muhammad Fahim},
  journal={xxxx},
  year={xxxx}
}
```

If you would like to get in touch, please contact .
m.fahim@innopolis.ru,
r.galieva@innopolis.university or g.dlamini@innopolis.university



# Abstract

>Unsupervised anomaly detection is an important area of research to find abnormal behavior and integral part of many systems. In this research, a lightweight deep autoencoder based approach is presented to detect anomalies in unsupervised manner. It has the ability to learn the model over the normal patterns and any deviation is considered as an anomaly. Consequently, it can relax the condition to have anomalous data patterns during the training phase of the model. In this work, we examine lightweight autoencoder for anomaly detection task in order to show that simple architecture can show good performance in terms of training, testing time, number of parameters and metrics. We apply autoencoder for binary classification problem (i.e., each data point considered either normal either abnormal). The reconstruction error is used to detect anomalies. The experiments are carried out over the particular class of cyber security domain known as intrusion detection systems. We evaluated our model on  standard publicly available benchmarks of KDD-99, NSL-KDD and UNSW-NB15 and achieved F1-score of 0.96, 0.88 and 0.95, respectively. It outperforms by a considerable margin when compared to state-of-the-art methods.



# Installation

This code is written in Python 3.6.7 using keras having tensorflow 1.x as backend

### Repository directory layout

    .
    ├── KDD99                   # Implementation for KDD-99 dataset
    │   ├── KDD99_Data          # KDD-99 Dataset folder
    │   ├── kdd99logs           # Model Training log files folder
    │   ├── preprocessing.py    # Data reprocessing file
    │   ├── kdd99.py            # Main file for training and testing model
    │   └── ...
    ├── NSL-KDD                 # Implementation for NSL-KDD dataset
    │   ├── NSL-KDD_Data        # NSL-KDD-99 Dataset folder
    │   ├── preprocessing.py    # Data reprocessing file
    │   ├── nslkdd.py           # Main file for training and testing model (binary classification)
    │   ├── multi_class.py      # Main file for training and testing model (multi-class classification)
    │   └── ...
    ├── UNSW-NB15               # Implementation for UNSW-NB15 dataset
    │   ├── dataset             # UNSW-NB15 Dataset folder
    │   ├── prerocessing.py     # Model training and testing
    │   ├──
    │   └── ...
    └── README.md


# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

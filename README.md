# **CONformal CHangepoint Localization (CONCH)**

**CONCH** is a distribution-free framework for offline changepoint localization that provides *exact finite-sample confidence sets*. This repository contains all simulation and real-data experiments from the [paper](https://rohanhore.github.io/research/CONCH.pdf) .

---

## **Repository Overview**

### **Simulations (`simulations/`)**

Notebooks for reproducing results on synthetic experiments in **Section 7.1**, **Appendix B.1–B.3**:

| Experiment | Description | Figures | File |
|------------------|--------------|----------|------|
| Gaussian mean-shift | CONCH confidence sets under Gaussian mean change for different CPP scores | Fig. 1, 6 | `Gaussian_sims.ipynb` |
| Calibrating Bootstrap sets | Calibration of Bootstrap confidence sets via CONCH | Fig. 2 | `Calibration.ipynb` |
| Multiple changepoint | CONCH to localize multiple changepoint with KCPD segmentation | Fig. 7 | `Multiple_chpt.ipynb` |
| Two-urn model | CONCH confidence sets for the two-urn setup | Fig. 8 | `Two_urns_sims.ipynb` |

---

### **Real-Data Experiments (`real_experiments/`)**

Notebooks for reproducing real-data results in **Section 7.2**, **Appendix B.4–B.5**:

| Task | Dataset | Figures | File |
|------|----------|----------|------|
| Domain shift | DomainNet | Fig. 3–4 | `Domainnet_expt.ipynb` |
| Sentiment shift | SST-2 | Fig. 5 | `SST2_expt.ipynb` |
| Digit shift | MNIST | Fig. 9–10 | `MNIST_expt.ipynb` |
| Class shift | CIFAR-100 | Fig. 11–12 | `CIFAR100_expt.ipynb` |

---

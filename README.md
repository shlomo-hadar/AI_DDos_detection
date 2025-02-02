# DDoS Attack Detection Using Artificial Neural Networks

This project implements a Deep Neural Network model to detect Distributed Denial of Service (DDoS) attacks in network traffic.

## Overview

DDoS attacks are a significant cybersecurity threat that disrupts online services by overwhelming networks with traffic. This model focuses on detecting two types of DDoS attacks:

- High-volume request floods to hosting servers
- Mass random data packet transmissions that overload networks

## Key Features

- **Lightweight Architecture**: Uses a simple yet effective neural network structure
- **Data Preprocessing**: Implements comprehensive data normalization and encoding
- **Visualization Tools**: Includes multiple data visualization functions for analysis

## Model Architecture

The model uses a simple but powerful architecture:

- Input Layer: 10 features
- Hidden Layer: 5 neurons with ReLU activation and L2 regularization (0.3)
- Output Layer: 1 neuron with sigmoid activation and L2 regularization (0.1)

### Why This Architecture?

1. **Simplicity**: The lightweight design allows for quick training and inference
2. **Regularization**: L2 regularization prevents overfitting
3. **Binary Classification**: Sigmoid activation effectively separates normal and attack traffic

### Key Hyperparameter Changes

1. **Learning Rate**:

   - Initial: 1e-4
   - Optimized: 1e-3
   - Impact: Faster convergence and better optimization

2. **Training Data Distribution**:
   - Initial: 50% test data
   - Optimized: 10% test data
   - Impact: More training data led to better generalization

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/shlomo-hadar/AI_DDos_detection
cd AI_DDos_detection
cd scripts
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Basic usage with default dataset:

```python
python ddos_attack_detection_using_ann.py
```

2. To use a custom dataset:

# Use custom dataset

In order to change the program's works on, in the file `const.py`, change the variable `dataset_file` inside the `Paths` dataclass: `dataset_file = os.path.join(dataset_dir, 'APA-DDoS-Dataset.csv')`.

### Dataset Requirements

Your dataset should include the following columns:

- ip.src: Source IP address
- tcp.srcport: Source port number
- tcp.dstport: Destination port number
- frame.len: Network frame length
- frame.time: Timestamp
- Packets: Number of packets
- Label: Traffic classification ('Benign', 'DDoS-PSH-ACK', 'DDoS-ACK')

## Results Visualization

The code includes several visualization functions:

- Attack distribution analysis
- Traffic patterns over time
- Packet size analysis
- Model performance metrics
- Training and validation accuracy/loss curves

##### Note

In order to display the corolation visualization, uncomment the
call to `show_corelation_analisys(data_set=data_set)`

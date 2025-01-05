import re
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt

from const import Paths, Const, Str


def data_acqusiotion_and_understanding():
    """
    These descriptions provide details about various attributes present in the dataset:
     ip.src: Source IP address
     tcp.srcport: Source port number for TCP (Transmission Control Protocol).
     tcp.dstport: Destination port number for TCP.
     ip.proto: IP protocol used (e.g., TCP, UDP).
     frame.len: Length of the network frame.
     tcp.flags.syn: TCP SYN flag.
     tcp.flags.reset: TCP RST flag.
     tcp.flags.push: TCP PUSH flag.
     tcp.flags.ack: TCP ACK flag.
     ip.flags.mf: IP More Fragments flag.
     ip.flags.df: IP Do Not Fragment flag.
     ip.flags.rb: Reserved bits in the IP header.
     tcp.seq: TCP sequence number.
     tcp.ack: TCP acknowledgment number.
     frame.time: Timestamp of the network frame.
     Packets: Number of packets in the network frame.
     Bytes: Number of bytes in the network frame.
     Tx Packets: Number of transmitted packets.
     Tx Bytes: Number of transmitted bytes.
     Rx Packets: Number of received packets.
     Rx Bytes: Number of received bytes.
     Label: The label or category assigned to the network event (e.g., 'DDoS-PSH-ACK', 'Benign', 'DDoS-ACK').

    # Data Exploration & Cleaning:
    """
    print(f'*Acuire Data from dataset file.')
    return pd.read_csv(Paths.dataset_file)


def show_dataset_distribution(dataset):
    """**Benign:** This label indicates that the network event or traffic is considered normal and does not pose any threat. In other words, it represents benign or legitimate network activity.
    **DDoS-PSH-ACK:** This label represents a specific type of DDoS attack characterized by the TCP flags PSH (Push) and ACK (Acknowledgment). In this type of attack, the attacker sends a high volume of TCP packets with the PSH and ACK flags set, overwhelming the target server or network with a large number of connection requests.
    **DDoS-ACK:** This label indicates another type of DDoS attack where the attacker floods the target server or network with a massive number of TCP packets with only the ACK (Acknowledgment) flag set. This flood of ACK packets consumes resources on the target system, leading to service disruption or denial of service for legitimate users.
    The **PSH flag** indicates that the data should be pushed immediately to the application layer, which can cause the receiving system to process data more rapidly, potentially overwhelming its resources.
    """
    print(f'**Show dataset request type distribution.')
    label_counts = dataset['Label'].value_counts()
    plt.figure(figsize=(3, 3))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of DDoS Attack Types')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()
    plt.clf()


def data_preprocessing(dataset):
    def normalize_time_column(data_set):
        data_set['frame.time'] = data_set['frame.time'].str.replace(' Mountain Daylight Time', '')
        data_set['frame.time'] = data_set['frame.time'].apply(lambda x: re.sub(r'\..*', '', x))
        data_set['frame.time'] = pd.to_datetime(data_set['frame.time'], format=' %d-%b %Y %H:%M:%S')
        data_set['frame.time'] = data_set['frame.time'].dt.tz_localize('UTC').dt.tz_convert('US/Mountain')

    print(f'*Preprocess dataset.')
    dataset.drop('ip.dst', axis=1, inplace=True)  # destination is always the same.
    # dataset['tcp.srcport'].value_counts()  # """These IP addresses represent the botnets utilized for the DDoS attacks. The consistent count of occurrences (10800) for each IP address implies an even distribution of DDoS attack traffic across various botnets."""
    constant_features = dataset.columns[dataset.nunique() == 1]
    dataset.drop(columns=constant_features, inplace=True)  # removing the features that are constant.
    normalize_time_column(data_set=dataset)


def get_features(data_set):
    binary_features = data_set.columns[data_set.nunique() == 2]
    numerical_features = [feature for feature in data_set.columns if data_set[feature].dtypes != 'O']
    categorical_features = [feature for feature in data_set.columns if data_set[feature].dtypes == 'O']
    return binary_features, numerical_features, categorical_features


def exploratory_data_analysis(data_set):
    def show_benign_vs_ddos_overtime(data_set):
        print(f'**Show dataset benign VS malicious.')
        connection_by_time = data_set[data_set['Label'] != 'Benign'].groupby(['frame.time']).size()
        connection_by_time_b = data_set[data_set['Label'] == 'Benign'].groupby(['frame.time']).size()
        plt.title('Benign & non-Benign DDoS Connections Over Time')
        plt.plot(connection_by_time.index, connection_by_time.values, marker='o', linestyle='-', label='Non-Benign')
        plt.plot(connection_by_time_b.index, connection_by_time_b.values, marker='o', linestyle='-', label='Benign')
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()
        plt.clf()

    def show_ip_req_by_time_distribution(data_set):
        connection_by_time_different_ip_src = data_set[data_set['Label'] != 'Benign'].groupby(
            ['ip.src', 'frame.time']).size()
        connection_by_time_different_ip_src.unstack(level=0).plot(kind='bar', stacked=True)
        plt.title('Count of Non-Benign DDoS Attacks by IP Source and Time')
        plt.xlabel('Frame Time')
        plt.ylabel('Count')
        plt.legend(title='IP Source')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        plt.clf()

    def show_bytes_over_time(data_set):
        data_set[data_set['Label'] == 'Benign'].groupby(['frame.time', 'Packets']).size()
        non_benign_data = data_set[data_set['Label'] != 'Benign']
        non_benign_bytes_by_time = non_benign_data.groupby(['frame.time', 'Bytes']).size().reset_index(name='count')
        non_benign_bytes_by_time = non_benign_bytes_by_time.groupby('frame.time')['Bytes'].mean()
        benign_data = data_set[data_set['Label'] == 'Benign']
        benign_bytes_by_time = benign_data.groupby(['frame.time', 'Bytes']).size().reset_index(name='count')
        benign_bytes_by_time = benign_bytes_by_time.groupby('frame.time')['Bytes'].mean()
        plt.plot(non_benign_bytes_by_time.index, non_benign_bytes_by_time.values, label='Non-Benign', linestyle='-',
                 marker='o')
        plt.plot(benign_bytes_by_time.index, benign_bytes_by_time.values, label='Benign', linestyle='-', marker='o')
        plt.title('Average Bytes Received Over Time for non benign connections')
        plt.xlabel('Time')
        plt.ylabel('Average Bytes')
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.clf()

    def show_avg_packets_over_time(data_set):
        benign = data_set[data_set['Label'] == "Bengin"].Packets
        ddos_ack = data_set[data_set['Label'] == "DDoS-ACK"].Packets
        ddos_psh_ack = data_set[data_set['Label'] == "DDoS-PSH-ACK"].Packets
        plt.xlabel("Numbre of Packets")
        plt.ylabel("")
        plt.title("Numbre of packets for label ddos Visualiztion")
        plt.hist([benign, ddos_ack, ddos_psh_ack], rwidth=0.95, color=['green', 'blue', 'red'],
                 label=['Bengin', 'ddos_ack', 'ddos_psh_ack'])
        plt.legend()

        # Filter non-benign and benign data
        non_benign_data = data_set[data_set['Label'] != 'Benign']
        benign_data = data_set[data_set['Label'] == 'Benign']

        # Group by 'frame.time' and 'Packets' and calculate the mean for non-benign traffic
        non_benign_packets_by_time = non_benign_data.groupby(['frame.time', 'Packets']).size().reset_index(name='count')
        non_benign_packets_by_time = non_benign_packets_by_time.groupby('frame.time')['Packets'].mean()

        # Group by 'frame.time' and 'Packets' and calculate the mean for benign traffic
        benign_packets_by_time = benign_data.groupby(['frame.time', 'Packets']).size().reset_index(name='count')
        benign_packets_by_time = benign_packets_by_time.groupby('frame.time')['Packets'].mean()

        # Plotting the line chart for non-benign and benign traffic
        plt.figure(figsize=(12, 6))
        plt.plot(non_benign_packets_by_time.index, non_benign_packets_by_time.values, label='Non-Benign', linestyle='-',
                 marker='o')
        plt.plot(benign_packets_by_time.index, benign_packets_by_time.values, label='Benign', linestyle='-', marker='o')
        plt.title('Average Packets Received Over Time')
        plt.xlabel('Time')
        plt.ylabel('Average Packets')
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.clf()

    def show_packets_over_time(data_set):
        # Filter non-benign and benign data
        non_benign_data = data_set[data_set['Label'] != 'Benign']
        benign_data = data_set[data_set['Label'] == 'Benign']

        # Plot histograms for non-benign and benign connections
        plt.figure(figsize=(12, 6))
        plt.hist(non_benign_data['Packets'], bins=30, alpha=0.7, color='red', label='Non-Benign')
        plt.hist(benign_data['Packets'], bins=30, alpha=0.7, color='blue', label='Benign')
        plt.title('Distribution of Packets Over Time')
        plt.xlabel('Number of Packets')
        plt.ylabel('Frequency')
        plt.legend()
        # plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.clf()

    def show_frequency_labels_over_time(data_set):
        label_counts = data_set.groupby(['frame.time', 'Label']).size().unstack(fill_value=0)
        label_counts['Benign'].plot(kind='line', figsize=(10, 6))
        label_counts['DDoS-PSH-ACK'].plot(kind='line', figsize=(10, 6))
        label_counts['DDoS-ACK'].plot(kind='line', figsize=(10, 6))

        plt.title('Frequency of Labels Over Time')
        plt.xlabel('Time')
        plt.xticks(rotation=90)
        plt.ylabel('Count')
        plt.legend()
        plt.show()
        plt.clf()

    def show_corelation_analisys(data_set):
        print(f'Showing correlation analisys. this operation takes some time.')
        sns.pairplot(data_set, hue='Label', vars=['frame.time', 'Packets', 'Tx Packets'])
        numerical_features = [feature for feature in data_set.columns if data_set[feature].dtypes != 'O']
        categorical_features = [feature for feature in data_set.columns if data_set[feature].dtypes == 'O']
        corr_matrix = data_set[numerical_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Heatmap of Numerical Features')
        plt.show()

    # def show_bytes_over_time(data_set):
    # def show_bytes_over_time(data_set):

    show_benign_vs_ddos_overtime(data_set=data_set)
    show_ip_req_by_time_distribution(data_set=data_set)
    # show_bytes_over_time(data_set=data_set)
    # show_avg_packets_over_time(data_set=data_set)
    # show_packets_over_time(data_set=data_set)
    # show_frequency_labels_over_time(data_set=data_set)
    show_corelation_analisys(data_set=data_set)


def normalize_data(dataset, numerical_features):

    def z_score_normalization(x):
        return (x - x.mean()) / x.std()

    def encode_ip_src_column(data_set, label_encoder):
        data_set['ip.src'] = label_encoder.fit_transform(data_set['ip.src'])

    def encode_label_column_1(data_set, label_encoder):
        label_mapping = {'Benign': 0, 'DDoS-PSH-ACK': 1, 'DDoS-ACK': 1}
        data_set['label_encoded'] = label_encoder.fit_transform(data_set['Label'])
        data_set['label_encoded'] = data_set['Label'].map(label_mapping)

    def encode_label_column_2(data_set):
        data1 = pd.get_dummies(dataset, columns=['Label'], drop_first=False, dtype='int64')
        return data1

    label_encoder = LabelEncoder()
    binary_features = dataset.columns[dataset.nunique() == 2]

    """We do not normalize binary features."""
    numerical_features.remove('Tx Bytes')
    numerical_features.remove('Rx Bytes')
    dataset.drop(['Tx Bytes', 'Rx Bytes'], axis=1, inplace=True)
    numerical_features = [feature for feature in numerical_features if feature not in binary_features]
    dataset[numerical_features] = dataset[numerical_features].apply(z_score_normalization)
    dataset[numerical_features].isnull().sum()

    """## Labeling Catgorical data:
    If we were to use sigmoid as the final activation of our neural network we have to encode the Label column using LabelEncoder.
    If we're using softmax a.k.a classifying into 3 classes, we use One Hot encoder.
    """
    encode_ip_src_column(data_set=dataset, label_encoder=label_encoder)
    encode_label_column_1(data_set=dataset, label_encoder=label_encoder)
    dataset = encode_label_column_2(data_set=dataset)
    return dataset


def train_model(dataset):
    y = dataset['label_encoded']
    X = dataset.drop(columns=['label_encoded', 'Label_Benign', 'Label_DDoS-ACK', 'Label_DDoS-PSH-ACK'])
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)

    """The distribution of classes between the two train and test sets is even.

    # Defining the deep learning model:
    """

    model = Sequential([
        Input(shape=(10,)),
        Dense(units=5, activation='relu', name='Hidden_layer_1', kernel_regularizer=L2(0.3)),
        # Dense(units=2,activation='relu',name='Hidden_layer_2',kernel_regularizer=L2(0.3)),
        Dense(units=1, activation='sigmoid', name='Output_layer', kernel_regularizer=L2(0.1))
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()

    """## Fitting the model:"""

    history_log = model.fit(
        X_train,
        y_train,
        batch_size=1024,
        epochs=100, verbose=2,
        callbacks=None,
        shuffle=True,
        validation_data=(X_val, y_val),
        class_weight=None,
        sample_weight=None,
        initial_epoch=0)
    return model, history_log


def main():
    """Problem definition
    Distributed Denial of Service: it's a cybersecurity menace which disrupts online services by sending an overwhelming amount of network traffic. These attacks are manually started with botnets that flood the target network. These attacks could have either of the following characteristics:
     The botnet sends a massive number of requests to the hosting servers.
     The botnet sends a high volume of random data packets, thus incapacitating the network.
     our goal is to detect the attack with certainty > 0.9"""
    ddos_data = data_acqusiotion_and_understanding()
    # show_dataset_distribution(dataset=ddos_data)
    data_preprocessing(dataset=ddos_data)
    # exploratory_data_analysis(data_set=ddos_data)
    binary_features, numerical_features, categorical_features = get_features(data_set=ddos_data)
    ddos_data = normalize_data(dataset=ddos_data, numerical_features=numerical_features)
    model, history_log = train_model(dataset=ddos_data)
    exit()

    """## Plotting the loss by epochs:"""
    loss = history_log.history['loss']
    val_loss = history_log.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'g', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Loss v/s No. of epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()

    """## Plotting the accuracy by epochs:"""

    accuracy = history_log.history['accuracy']
    val_accuracy = history_log.history['val_accuracy']
    plt.plot(epochs, accuracy, 'g', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Accuracy Scores v/s Number of Epochs')
    plt.xlabel('No. of Epochs')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.show()
    plt.clf()

    """## Model evaluation for test set:"""

    loss, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy of Deep neural Network on unseen data : %.2f' % (accuracy * 100))


if __name__ == '__main__':
    main()

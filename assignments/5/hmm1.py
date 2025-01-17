import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
from hmmlearn import hmm

def extract_mfcc(file_path, n_mfcc=13, n_fft=512, hop_length=256, target_sr=16000):
    y, sr = librosa.load(file_path, sr=target_sr)
    fmax = sr / 2 if sr / 2 < 8000 else 8000  # Cap fmax to 8 kHz for speech
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, fmax=fmax)
    return mfcc

def load_dataset(dataset_path, n_mfcc=13):
    data = {str(digit): [] for digit in range(10)}
    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.wav'):
            digit = file_name.split('_')[0]
            file_path = os.path.join(dataset_path, file_name)
            mfcc = extract_mfcc(file_path, n_mfcc=n_mfcc).T # Transpose for time-steps as rows
            data[digit].append(mfcc)
    return data

def predict_digit(test_mfcc, models):
    max_score = float('-inf')
    best_digit = None
    for digit, model in models.items():
        try:
            score = model.score(test_mfcc)
            if score > max_score:
                max_score = score
                best_digit = digit
        except:
            pass
    return best_digit

def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total = len(y_true)
    accuracy = correct / total if total > 0 else 0
    return accuracy

def evaluate_accuracy(data, models):
    y_true = []
    y_pred = []
    for digit, sequences in data.items():
        for mfcc in sequences:
            y_true.append(int(digit))
            predicted = predict_digit(mfcc, models)
            y_pred.append(int(predicted) if predicted is not None else -1)

    accuracy = calculate_accuracy(y_true, y_pred)
    return accuracy


def split_dataset(data, test_size=0.2):
    train_data = {}
    test_data = {}
    for digit, sequences in data.items():
        split_idx = int(len(sequences) * (1 - test_size))
        train_data[digit] = sequences[:split_idx]
        test_data[digit] = sequences[split_idx:]
    return train_data, test_data


def main():

    dataset_path = '../../data/external/spoken_digit_dataset/recordings/'
    recorded_dataset_path = '../../data/external/recorded_digit_dataset/'


    ## 3.2

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("MFCC Features for Digits 0-9", fontsize=16)

    for digit in range(10):
        sample_file = next(file for file in os.listdir(dataset_path) if file.startswith(str(digit)))
        file_path = os.path.join(dataset_path, sample_file)
        
        mfcc_features = extract_mfcc(file_path)
        
        ax = axes[digit // 5, digit % 5]
        img = librosa.display.specshow(mfcc_features, x_axis='time', ax=ax)
        ax.set_title(f"Digit: {digit}")
        ax.axis("off")

    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    fig.colorbar(img, cax=cbar_ax, orientation='horizontal')
    # plt.show()
    plt.savefig('figures/MFCC_Features.png')


    ## 3.3

    data = load_dataset(dataset_path)

    models = {}
    for digit, sequences in data.items():
        if not sequences:
            print(f"No sequences found for digit {digit}. Skipping model training.")
            continue
        X = np.vstack(sequences)
        lengths = [len(seq) for seq in sequences]
        model = hmm.GaussianHMM(n_components=5, covariance_type='diag', n_iter=100, random_state=42)
        model.fit(X, lengths)
        models[digit] = model


    ## 3.4
      
    train_data, test_data = split_dataset(data, test_size=0.2)
    
    accuracy = evaluate_accuracy(test_data, models)
    print(f"Accuracy on test sample: {accuracy * 100:.2f}%")

    if os.path.exists(recorded_dataset_path):
        user_data = load_dataset(recorded_dataset_path)
        accuracy = evaluate_accuracy(user_data, models)
        print(f"Accuracy on recorded dataset: {accuracy * 100:.2f}%")
    else:
        print(f"User-recorded dataset not found at {recorded_dataset_path}.")

        # Accuracy on test sample: 90.67%
        # Accuracy on recorded dataset: 20.00%

if __name__ == "__main__":
    main()

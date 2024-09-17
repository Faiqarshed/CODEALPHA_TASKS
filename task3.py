import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
    return result['encoding']

file_path = 'essays.csv'
encoding = detect_encoding(file_path)
print(f"Detected encoding: {encoding}")


data = pd.read_csv(file_path, encoding=encoding)

# Print the first few rows to verify
print(data.head())

# Preprocess the data
def preprocess_text(text):
    # Simple text preprocessing (lowercasing and stripping)
    return text.lower().strip()

# Apply preprocessing
data['processed_text'] = data['TEXT'].apply(preprocess_text)

# Define features and target
X_text = data['processed_text']
X_features = data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']]
y = data[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']].mean(axis=1)  # Assuming average score as target

# Tokenize and pad text sequences
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_text)
X_text_seq = tokenizer.texts_to_sequences(X_text)
X_text_pad = pad_sequences(X_text_seq, maxlen=max_len)

# Split the data
X_train_text, X_test_text, X_train_features, X_test_features, y_train, y_test = train_test_split(
    X_text_pad, X_features, y, test_size=0.2, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_test_features = scaler.transform(X_test_features)

# Build the model
text_input_shape = X_train_text.shape[1]
feature_input_shape = X_train_features.shape[1]

text_input = tf.keras.Input(shape=(text_input_shape,))
feature_input = tf.keras.Input(shape=(feature_input_shape,))

embedding_layer = Embedding(input_dim=max_words, output_dim=128, input_length=text_input_shape)(text_input)
lstm_layer = LSTM(64)(embedding_layer)
text_output = Dense(32, activation='relu')(lstm_layer)

feature_output = Dense(32, activation='relu')(feature_input)

combined = tf.keras.layers.concatenate([text_output, feature_output])
combined_output = Dense(1)(combined)

model = tf.keras.Model(inputs=[text_input, feature_input], outputs=combined_output)

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    [X_train_text, X_train_features], y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping]
)

# Evaluate the model
y_pred = model.predict([X_test_text, X_test_features])
mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")

# Example prediction
def predict_score(text, features):
    processed_text = preprocess_text(text)
    text_seq = tokenizer.texts_to_sequences([processed_text])
    text_pad = pad_sequences(text_seq, maxlen=max_len)
    features_scaled = scaler.transform([features])
    predicted_score = model.predict([text_pad, features_scaled])
    return predicted_score[0][0]

example_text = "Your example essay text goes here."
example_features = [0.5, 0.5, 0.5, 0.5, 0.5]  # Example features
predicted_score = predict_score(example_text, example_features)
print(f"Predicted Score: {predicted_score}")

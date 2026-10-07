# 📧 Email Spam Detection

A deep learning-based **Email Spam Detection System** that uses Natural Language Processing (NLP) and a **GRU (Gated Recurrent Unit) based Recurrent Neural Network** to classify email messages as **Spam** or **Ham (Not Spam)**.

The project includes an interactive **Streamlit web application** where users can enter an email message and receive a real-time prediction along with the spam probability.

---

## 🚀 Live Demo

🔗 [**Try the Email Spam Detector**](https://email-spam-detection-lh3apeiuzjgtjzphzqtkdd.streamlit.app/)

---

## 📌 Project Overview

Email spam is a common problem where unwanted, fraudulent, promotional, or potentially harmful messages are sent to users.

This project uses **Natural Language Processing and Deep Learning** to automatically analyze the content of an email and determine whether it is:

- 🚨 **Spam** — potentially unwanted or suspicious email
- ✅ **Ham** — legitimate email

The system processes the email text, converts it into numerical sequences using a trained tokenizer, and passes the padded sequence through a trained GRU-based neural network.

The model outputs a probability, which is then used to determine the final classification.

---

## ✨ Features

- 📧 Classifies emails as **Spam** or **Ham**
- 🧠 Uses a **GRU-based Recurrent Neural Network**
- 🔤 NLP-based text preprocessing
- 🔗 Removes hyperlinks from email text
- 🧹 Removes punctuation and stopwords
- 🔢 Converts text into numerical sequences
- 📏 Pads sequences to a fixed length
- ⚡ Real-time prediction
- 📊 Displays spam probability
- 🖥️ Interactive Streamlit interface
- 💾 Loads trained model and preprocessing artifacts
- 🌐 Deployed using Streamlit Community Cloud

---

## 🧠 Machine Learning Approach

The project uses a **GRU (Gated Recurrent Unit)** based Recurrent Neural Network for text classification.

GRU is a type of recurrent neural network designed to process sequential data such as text.

Unlike traditional machine learning approaches that rely heavily on manually engineered text features, the GRU model can learn patterns and relationships within sequences of words.

### Why GRU?

GRU networks are useful for text classification because they:

- Process sequential text data
- Capture relationships between words
- Handle variable-length sequences
- Require fewer parameters than some other recurrent architectures
- Can be faster to train than more complex recurrent models such as LSTMs

---

## 🔄 Processing Pipeline

```text
Email Text
    ↓
Lowercase Conversion
    ↓
Hyperlink Removal
    ↓
Punctuation Removal
    ↓
Tokenization
    ↓
Stopword Removal
    ↓
Text-to-Sequence Conversion
    ↓
Sequence Padding
    ↓
GRU-based RNN
    ↓
Spam Probability
    ↓
Classification
    ↓
Spam / Ham
```

---

## 🧹 Text Preprocessing

Before sending the email to the neural network, the text goes through several preprocessing steps.

### 1. Lowercase Conversion

The email text is converted to lowercase so that words with different capitalization are treated consistently.

```text
"Congratulations YOU Won"
```

becomes:

```text
"congratulations you won"
```

### 2. Hyperlink Removal

URLs are removed from the email text before punctuation processing.

For example:

```text
Visit https://example.com/free-prize
```

is cleaned so that the URL does not interfere with tokenization.

### 3. Punctuation Removal

Punctuation marks are removed from the text.

```text
"Congratulations!!!"
```

becomes:

```text
"congratulations"
```

### 4. Tokenization

The cleaned text is divided into individual words using NLTK tokenization.

```text
"win a free prize"
```

becomes:

```text
["win", "a", "free", "prize"]
```

### 5. Stopword Removal

Common English stopwords are removed using the NLTK stopword corpus.

Examples include:

```text
the
is
a
an
and
to
```

This helps reduce unnecessary words before classification.

### 6. Sequence Conversion

The trained Keras tokenizer converts the processed text into numerical sequences.

```text
["win", "free", "prize"]
```

becomes something similar to:

```text
[42, 18, 97]
```

### 7. Padding

Sequences are padded to the maximum sequence length used during model training.

This ensures that every input has the same shape before being passed to the GRU model.

---

## 🤖 Model Prediction

The processed email is passed to the trained GRU model.

The model produces a probability value between `0` and `1`.

```text
Probability > 0.5
        ↓
      Spam

Probability ≤ 0.5
        ↓
       Ham
```

For example:

```text
Spam Probability: 92.45%
```

would result in:

```text
🚨 SPAM EMAIL
```

While a low spam probability would result in:

```text
✅ HAM / NOT SPAM
```

---

## 🖥️ Streamlit Application

The project includes a Streamlit web interface where users can paste an email message.

### Application Flow

```text
User enters email
        ↓
Click "Check Email"
        ↓
Preprocess email
        ↓
Convert text to sequence
        ↓
Pad sequence
        ↓
GRU model prediction
        ↓
Display classification
        ↓
Display probability
```

The application provides:

- Email input area
- Check Email button
- Spam/Ham classification
- Probability score
- Warning for empty input

---

## 📂 Project Structure

```text
email-spam-detection/
│
├── app.py
├── gru_model.keras
├── tokenizer.pkl
├── config.pkl
├── label_mapping.pkl
├── requirements.txt
├── README.md
└── Email_Spam_Detection.ipynb
```

### File Description

| File | Description |
|------|-------------|
| `app.py` | Streamlit application and prediction pipeline |
| `gru_model.keras` | Trained GRU neural network |
| `tokenizer.pkl` | Saved text tokenizer |
| `config.pkl` | Model configuration such as maximum sequence length |
| `label_mapping.pkl` | Mapping between numerical classes and labels |
| `requirements.txt` | Python dependencies |
| `Email_Spam_Detection.ipynb` | Model development and experimentation notebook |
| `README.md` | Project documentation |

---

## 🛠️ Tech Stack

### Programming Language

- **Python**

### Deep Learning

- **TensorFlow**
- **Keras**
- **GRU**
- **Recurrent Neural Networks**

### Natural Language Processing

- **NLTK**
- Tokenization
- Stopword removal
- Text preprocessing

### Data Processing

- **NumPy**
- **Pandas**

### Web Application

- **Streamlit**

### Model Serialization

- **Pickle**

### Deployment

- **Streamlit Community Cloud**

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MousumiBadyakar/email-spam-detection.git
```

### 2. Navigate to the Project

```bash
cd email-spam-detection
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS / Linux

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Example Inputs

### Example 1 — Spam Email

```text
CONGRATULATIONS! You have won $1,000,000!
Click https://example.com/free-prize to claim your reward now.
You are the lucky winner! Act now!
```

Expected result:

```text
🚨 SPAM EMAIL
```

---

### Example 2 — Ham Email

```text
Hi John,

Just wanted to confirm that our project meeting is scheduled for tomorrow at 10 AM.

Please let me know if you need anything before the meeting.

Best regards,
Mou
```

Expected result:

```text
✅ HAM / NOT SPAM
```

---

## 📊 Prediction Output

The application displays both the predicted class and the corresponding probability.

Example:

```text
🚨 SPAM EMAIL

Spam Probability: 94.32%
```

or:

```text
✅ HAM / NOT SPAM

Ham Probability: 96.18%
```

---

## 🔐 Model Artifacts

The Streamlit application loads the following trained artifacts.

### GRU Model

```text
gru_model.keras
```

Contains the trained neural network used for classification.

### Tokenizer

```text
tokenizer.pkl
```

Contains the fitted tokenizer used during model training.

Using the same tokenizer during inference ensures that input text is converted into sequences consistently.

### Configuration

```text
config.pkl
```

Stores configuration values required during prediction, such as the maximum sequence length.

### Label Mapping

```text
label_mapping.pkl
```

Maps the numerical model output to the corresponding class label.

For example:

```text
0 → Ham
1 → Spam
```

The exact mapping is loaded dynamically from the saved file.

---

## ⚙️ Prediction Logic

The prediction process follows these steps:

```python
cleaned_email = preprocess_text(email)

sequence = tokenizer.texts_to_sequences(
    [cleaned_email]
)

padded_sequence = pad_sequences(
    sequence,
    maxlen=config["max_length"],
    padding="post"
)

probability = model.predict(
    padded_sequence,
    verbose=0
)[0][0]
```

The probability is then converted into a class:

```python
predicted_class = int(probability > 0.5)
```

Finally, the numerical class is converted into the corresponding label using the saved label mapping.

---

## 🎯 Project Goal

The main goal of this project is to demonstrate the practical application of:

- Natural Language Processing
- Text preprocessing
- Sequence modeling
- Recurrent Neural Networks
- GRU architecture
- Deep learning-based classification
- Model deployment
- Streamlit application development

The project combines these concepts into an end-to-end machine learning application.

---

## 🔮 Future Improvements

Possible future improvements include:

- 📊 Add model performance metrics to the application
- 📈 Add confusion matrix and classification reports
- 🧠 Compare GRU with LSTM and traditional ML models
- ⚡ Improve inference performance
- 🎯 Experiment with different classification thresholds
- 📧 Support email file uploads
- 📎 Support `.txt` and `.eml` email files
- 🔍 Add explainability for predictions
- 📊 Display confidence levels in the UI
- 🛡️ Improve handling of suspicious URLs and email patterns

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes**.

The prediction should not be considered a definitive determination of whether an email is malicious or safe. Real-world email security systems typically use multiple signals, including sender reputation, URLs, attachments, metadata, and behavioral patterns.

---

## 👩‍💻 Author

**Mousumi Badyakar**

GitHub: [MousumiBadyakar](https://github.com/MousumiBadyakar)

---

⭐ If you find this project useful, consider giving the repository a star!

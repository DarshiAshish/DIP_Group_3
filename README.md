\documentclass[12pt,a4paper]{article}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{enumitem}

\geometry{margin=1in}

\title{\textbf{Comparative Analysis of Custom CNNs vs Pretrained Image Models Using Federated Learning on CIFAR-10 Dataset}}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Abstract}
This project explores the comparative analysis between custom Convolutional Neural Network (CNN) models and pretrained image models for image classification tasks using the CIFAR-10 dataset. The models were evaluated using federated learning, a decentralized machine learning technique where training is distributed across multiple devices without sharing raw data.

The custom CNN models were compared against well-known pretrained models such as VGG16, ResNet50, and InceptionV3, with the goal of determining the effectiveness and efficiency of each approach in the context of federated learning.

\section*{Project Overview}
This project focuses on performing a comparative analysis between custom CNN architectures and pretrained models (VGG16, ResNet50, InceptionV3) on the CIFAR-10 dataset. The goal is to evaluate the performance of each model in terms of accuracy and efficiency, when trained using federated learning.

Federated learning was employed to train the models in a decentralized way across multiple clients, helping to maintain data privacy as training happens locally on client devices, with only model updates being shared.

\section*{Key Features}
\begin{itemize}
    \item \textbf{Custom CNN Models:} Three custom CNN architectures designed for image classification.
    \item \textbf{Pre-trained Models:} Use of VGG16, ResNet50, and InceptionV3 pretrained on ImageNet, adapted for CIFAR-10.
    \item \textbf{Federated Learning:} Distributed training setup that simulates a client-server architecture for decentralized learning.
    \item \textbf{CIFAR-10 Dataset:} A standard dataset for image classification tasks containing 60,000 32x32 color images across 10 classes.
\end{itemize}

\section*{Technologies Used}
\begin{itemize}
    \item Python
    \item TensorFlow
    \item Keras
    \item NumPy
    \item Google Colab
    \item Jupyter Notebook
\end{itemize}

\section*{Demo}
To run a demo to understand the performance of the models:
\begin{enumerate}
    \item Clone the project into the \texttt{Colab Notebooks} folder in your Google Drive.
    \item Navigate to the \texttt{demo} folder and run the \texttt{final\_demo.ipynb} file.
\end{enumerate}

\textbf{Note:} The entire project is developed in Jupyter Notebook. Please do not try to run it locally.

\section*{Code Structure}
\begin{description}[style=nextline]
    \item[\texttt{read\_data.ipynb}] Script to load and explore the CIFAR-10 dataset for use in federated learning.
    \item[\texttt{preprocess\_data\_1.ipynb}] Script to preprocess the CIFAR-10 dataset for random distribution.
    \item[\texttt{preprocess\_data\_2.ipynb}] Script to preprocess the CIFAR-10 dataset for categorical distribution.
    \item[\texttt{custom\_model\_1.ipynb}] Defines the first custom CNN model.
    \item[\texttt{custom\_model\_2.ipynb}] Defines the second custom CNN model.
    \item[\texttt{custom\_model\_3.ipynb}] Defines the third custom CNN model.
    \item[\texttt{pretrained\_model\_1.ipynb}] Loads and retrieves embeddings using ResNet50.
    \item[\texttt{pretrained\_model\_2.ipynb}] Loads and retrieves embeddings using VGG16.
    \item[\texttt{pretrained\_model\_3.ipynb}] Loads and retrieves embeddings using InceptionV3.
    \item[\texttt{pretrained\_second\_layer.ipynb}] Defines the second layer of a pretrained model for further training.
    \item[\texttt{local\_model\_train\_custom.ipynb}] Trains custom models locally using federated learning and aggregates updates.
    \item[\texttt{local\_model\_train\_pretrained.ipynb}] Trains embeddings retrieved by pretrained models locally using federated learning and aggregates updates.
\end{description}

\section*{Installation Instructions}
To run this project:
\begin{enumerate}
    \item Clone the repository into your Google Drive under the \texttt{Colab Notebooks} folder:
    \begin{verbatim}
    git clone https://github.com/your-username/your-project-name.git
    \end{verbatim}
    \item Rename the \texttt{code} folder to \texttt{DIP\_proj}.
\end{enumerate}

\section*{How to Use}
\begin{enumerate}
    \item After cloning, find your folder under \texttt{Colab Notebooks}.
    \item Rename the \texttt{code} folder to \texttt{DIP\_proj}.
    \item Run \texttt{pretrained\_model\_1.ipynb}, \texttt{pretrained\_model\_2.ipynb}, and \texttt{pretrained\_model\_3.ipynb} to retrieve embeddings on the dataset. The embeddings will be stored in the same folder structure.
    \item Run \texttt{local\_model\_train\_custom.ipynb} and \texttt{local\_model\_train\_pretrained.ipynb} to perform the federated learning process.
    \item Observe the metrics for each model.
    \item For a demo, navigate to the \texttt{demo} folder and run the \texttt{final\_demo.ipynb} file.
\end{enumerate}

\end{document}

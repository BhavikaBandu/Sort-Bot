# Automatic Fruit Sorting & Segmentation (Image Processing + E-Nose)

## Overview

This project demonstrates the development of an automated fruit sorting system that utilizes **computer vision** and **e-nose (olfactory sensor)** technology to classify and assess the ripeness and spoilage level of fruits. Specifically, this repository focuses on the **image processing** component (using OpenCV) and the integration of **e-nose** sensors for spoilage detection. The system is capable of efficiently sorting fruits such as bananas into categories based on their type, ripeness, and spoilage.

## Key Features

- **Image Processing with OpenCV**:
  - Detects fruit type and evaluates ripeness using computer vision techniques.
  - Analyzes images of fruits and segments them into distinct categories based on visual characteristics.

- **E-Nose Integration for Spoilage Detection**:
  - Utilizes an olfactory sensor (e-nose) to detect spoilage by analyzing the volatile compounds emitted by the fruit.
  - Integrates real-time data from the e-nose to provide accurate spoilage classification.

- **Real-Time Sorting**:
  - Combines both image processing and e-nose data to automatically sort fruits into **raw**, **fresh**, and **spoiled** categories.

- **Data Analysis with Scikit-learn**:
  - Applies predictive models using **Scikit-learn** to analyze the collected data for better decision-making in sorting.

## Technologies Used

- **OpenCV**: For image processing and fruit classification.
- **Python 3.x**
- **Scikit-learn**: For predictive analysis on fruit ripeness and spoilage data.
- **E-Nose Technology**: Olfactory sensor technology used to detect spoilage and analyze volatile compounds emitted by fruits.
- **IoT Integration**: For real-time monitoring and data collection.

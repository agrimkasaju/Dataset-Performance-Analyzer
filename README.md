# Dataset Performance Analyzer

The **Dataset Performance Analyzer** is a machine learning–based tool designed to evaluate and optimize dataset performance metrics. Initially applied to an object detection dataset from a Smart Inventory System, the analyzer helps identify confidence score inconsistencies and provides adjusted predictions that can be used to improve model reliability and training quality.

---

## 🧠 Project Overview

This project demonstrates how machine learning can enhance dataset analysis for AI-driven systems. By training a regression model on key features such as class identifiers and prediction scores, the analyzer predicts adjusted confidence scores—offering insights into model consistency and data balance.

### Key Features

* Reads and processes object detection datasets (CSV format)
* Encodes categorical class labels
* Trains a Random Forest Regressor to predict adjusted confidence scores
* Outputs an updated CSV with optimized prediction values
* Easily adaptable for other dataset types or ML models

---

## 🧩 Technologies Used

* **Python 3.10+**
* **Pandas** for data manipulation
* **NumPy** for numerical operations
* **scikit-learn** for model training and evaluation
* **Azure ML** (optional) for remote job execution and monitoring

---

## ⚙️ How It Works

1. **Input Data**
   The script reads a CSV file containing object detection data with the following columns:

   * `agrim_Object_Detection_Column1_bounding_box`
   * `agrim_Object_Detection_Column1_class_id`
   * `agrim_Object_Detection_Column1_name`
   * `agrim_Object_Detection_Column1_score`

2. **Feature Engineering**

   * Encodes class names into numerical values.
   * Uses class IDs and encoded names as features.

3. **Model Training**

   * Splits the data into training and testing sets.
   * Trains a **RandomForestRegressor** to predict confidence scores.

4. **Output Generation**

   * Creates a new column: `adjusted_confidence`
   * Saves an updated dataset with improved predictions to the output path.

---

## 🚀 Usage

### Command Line

```bash
python predictionML.py --input_csv <path_to_input_csv> --output_csv <path_to_output_csv>
```

### Example

```bash
python predictionML.py --input_csv object_detection.csv --output_csv adjusted_predictions.csv
```

### Azure ML Execution

When run in Azure ML, inputs and outputs are automatically mounted as directories.
The script resolves the correct CSV file path before processing.

---

## 📈 Results

Below is a preview of the generated **adjusted_predictions.csv** file:

| agrim_Object_Detection_Column1_class_id | agrim_Object_Detection_Column1_name | agrim_Object_Detection_Column1_score | adjusted_confidence |
|----------------------------------------|------------------------------------|-------------------------------------|---------------------|
| 43 | bottle | 0.523 | 0.5688 |
| 46 | cup | 0.628 | 0.6615 |
| 81 | refrigerator | 0.312 | 0.4239 |
| ... | ... | ... | ... |

The full dataset is available here:  
👉 [adjusted_predictions.csv](./adjusted_predictions.csv)

---

## 💡 Future Improvements

* Integrate bounding box geometry for deeper performance insights
* Add visualization tools (Power BI / Matplotlib)
* Extend to detect data imbalance and recommend resampling strategies
* Incorporate support for time-series or multimodal datasets

---

## 👤 Author

**Agrim Kasaju**
Bachelor of Engineering in Computer Systems Engineering
Carleton University

🔗 [GitHub: agrimkasaju](https://github.com/agrimkasaju)

---

## 🏁 License

This project is released under the MIT License.
You are free to use, modify, and distribute it with proper attribution.

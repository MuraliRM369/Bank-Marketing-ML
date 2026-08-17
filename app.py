import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🏦 Bank Marketing Prediction App")

st.write(
    "Predict whether a customer will subscribe "
    "to a bank term deposit using machine learning."
)


# ============================================================
# REQUIRED FEATURES
# ============================================================

FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome"
]


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load(
            "model/logistic_regression.pkl"
        ),

        "Decision Tree": joblib.load(
            "model/decision_tree.pkl"
        ),

        "KNN": joblib.load(
            "model/knn.pkl"
        ),

        "Gaussian Naive Bayes": joblib.load(
            "model/naive_bayes.pkl"
        ),

        "Random Forest": joblib.load(
            "model/random_forest.pkl"
        )
    }

    preprocessor = joblib.load(
        "model/preprocessor.pkl"
    )

    return models, preprocessor


# Load saved models
models, preprocessor = load_models()


# ============================================================
# SIDEBAR - MODEL SELECTION
# ============================================================

st.sidebar.header("🤖 Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    list(models.keys())
)

selected_model = models[selected_model_name]


# ============================================================
# CSV UPLOAD
# ============================================================

st.header("📂 Upload CSV File")

uploaded_file = st.file_uploader(
    "Upload your customer CSV file",
    type=["csv"]
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file is not None:

    # ========================================================
    # READ CSV
    # ========================================================

    data = pd.read_csv(uploaded_file)

    st.subheader("📋 Uploaded Data")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )


    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in data.columns
    ]

    if missing_features:

        st.error(
            "The uploaded CSV is missing required columns."
        )

        st.write(missing_features)

        st.stop()


    # ========================================================
    # PREPARE INPUT FEATURES
    # ========================================================

    X = data[FEATURES].copy()


    # ========================================================
    # CHECK TARGET COLUMN
    # ========================================================

    has_target = "y" in data.columns


    # ========================================================
    # PREPARE TRUE TARGET VALUES
    # ========================================================

    if has_target:

        y_true = (
            data["y"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "no": 0,
                "yes": 1,
                "0": 0,
                "1": 1
            })
        )


        # Check invalid values

        if y_true.isna().any():

            st.error(
                "Invalid values found in the 'y' column."
            )

            st.write(
                data["y"].value_counts(
                    dropna=False
                )
            )

            st.stop()


        # Convert to integer

        y_true = y_true.astype(int)


    # ========================================================
    # PREPROCESS DATA
    # ========================================================

    try:

        X_processed = preprocessor.transform(X)

    except Exception as e:

        st.error(
            "Error while preprocessing the uploaded data."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # MAKE PREDICTIONS
    # ========================================================

    try:

        predictions = selected_model.predict(
            X_processed
        )

    except Exception as e:

        st.error(
            "Error while making predictions."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # CONVERT PREDICTIONS TO 0/1
    # ========================================================

    prediction_series = pd.Series(
        predictions
    )


    prediction_series = (
        prediction_series
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "no": 0,
            "yes": 1,
            "0": 0,
            "1": 1
        })
    )


    # Check prediction values

    if prediction_series.isna().any():

        st.error(
            "The model produced unexpected prediction values."
        )

        st.write(
            pd.Series(predictions).unique()
        )

        st.stop()


    # Convert to integer

    y_pred = prediction_series.astype(int)


    # ========================================================
    # PREDICTION RESULTS
    # ========================================================

    st.header("🔮 Prediction Results")

    result_data = data.copy()


    result_data["Prediction"] = (
        y_pred.values
    )


    result_data["Prediction_Label"] = (
        result_data["Prediction"]
        .map({
            0: "No",
            1: "Yes"
        })
    )


    st.dataframe(
        result_data.head(20),
        use_container_width=True
    )


    # ========================================================
    # DOWNLOAD PREDICTIONS
    # ========================================================

    csv_output = result_data.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Predictions CSV",
        data=csv_output,
        file_name="predictions.csv",
        mime="text/csv"
    )


    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    if has_target:

        st.header("📊 Model Evaluation")


        # ====================================================
        # ACCURACY
        # ====================================================

        accuracy = accuracy_score(
            y_true,
            y_pred
        )


        # ====================================================
        # PRECISION
        # ====================================================

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )


        # ====================================================
        # RECALL
        # ====================================================

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )


        # ====================================================
        # F1 SCORE
        # ====================================================

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )


        # ====================================================
        # MCC
        # ====================================================

        mcc = matthews_corrcoef(
            y_true,
            y_pred
        )


        # ====================================================
        # AUC
        # ====================================================

        auc = None


        try:

            if hasattr(
                selected_model,
                "predict_proba"
            ):

                probabilities = (
                    selected_model
                    .predict_proba(
                        X_processed
                    )
                )


                # Positive class probability

                if probabilities.shape[1] >= 2:

                    y_score = (
                        probabilities[:, 1]
                    )


                    auc = roc_auc_score(
                        y_true,
                        y_score
                    )

        except Exception:

            auc = None


        # ====================================================
        # DISPLAY SIX METRICS
        # ====================================================

        st.subheader(
            "📈 Evaluation Metrics"
        )


        # First row

        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )


        if auc is not None:

            col2.metric(
                "AUC",
                f"{auc:.4f}"
            )

        else:

            col2.metric(
                "AUC",
                "N/A"
            )


        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )


        # Second row

        col4, col5, col6 = st.columns(3)


        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )


        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )


        col6.metric(
            "MCC",
            f"{mcc:.4f}"
        )


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        st.subheader(
            "📋 Classification Report"
        )


        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[
                "No",
                "Yes"
            ],
            output_dict=True,
            zero_division=0
        )


        report_df = pd.DataFrame(
            report
        ).transpose()


        st.dataframe(
            report_df,
            use_container_width=True
        )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        st.subheader(
            "🔲 Confusion Matrix"
        )


        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1]
        )


        fig, ax = plt.subplots()


        ax.imshow(cm)


        ax.set_xlabel(
            "Predicted"
        )


        ax.set_ylabel(
            "Actual"
        )


        ax.set_title(
            f"Confusion Matrix - {selected_model_name}"
        )


        ax.set_xticks(
            [0, 1]
        )


        ax.set_yticks(
            [0, 1]
        )


        ax.set_xticklabels(
            ["No", "Yes"]
        )


        ax.set_yticklabels(
            ["No", "Yes"]
        )


        # Add values inside matrix

        for i in range(
            cm.shape[0]
        ):

            for j in range(
                cm.shape[1]
            ):

                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center"
                )


        st.pyplot(fig)


else:

    st.info(
        "👆 Please upload a CSV file to begin."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Bank Marketing Prediction | Machine Learning Project"
)
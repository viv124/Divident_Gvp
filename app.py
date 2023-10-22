from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = 'Uploads'
app.config['OUTPUT_FOLDER'] = 'FilteredOutput'


loaded_model = joblib.load('model.joblib')
loaded_vectorizer = joblib.load('vectorizer.joblib')

description_col = ["descri", "Description", "description", "Description", "Desc", "Descr", "Trans Description", "Transaction Desc"]
credit = ["credit", "Credit", "cr", "CR", "Deposit", "Amount Credited"]
ref_no = ["Ref_No", "ref_no", "Ref_No", "Reference Number", "Ref Num", "Transaction ID"]


filtered_data_list = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        files = request.files.getlist('file')  # Get a list of uploaded files

        if not files:
            flash('No files uploaded.')
            return redirect(url_for('index'))

        for file in files:
            df = process_uploaded_file(file, description_col, ref_no)

            if df is not None:
                input_data = df['FileData']
                input_data_vectorized = loaded_vectorizer.transform(input_data)
                predictions = loaded_model.predict(input_data_vectorized)
                df['Predictions'] = predictions

                # Convert 'Credit' column to numeric, handling non-numeric values gracefully
                df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce')

                # Filter rows where predictions are 1
                predicted_1 = df[df['Predictions'] == 1]

                # Append the filtered data to the list
                filtered_data_list.append(predicted_1)

                # Save the updated DataFrame back to the original file
                df.to_excel(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), index=False)

        # Concatenate the filtered data from all files
        if filtered_data_list:
            concatenated_data = pd.concat(filtered_data_list, ignore_index=True)

            # Save the concatenated data to a single Excel file
            merged_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'merged_data.xlsx')
            concatenated_data.to_excel(merged_filename, index=False)

            total_credit_payment = concatenated_data['Credit'].sum()

            return render_template('result.html', predicted_1=concatenated_data, total_credit_payment=total_credit_payment)

        else:
            flash('No valid data found in uploaded files.')
            return redirect(url_for('index'))

@app.route('/download')
def download():
    merged_filename = os.path.join(app.config['OUTPUT_FOLDER'], 'merged_data.xlsx')
    return send_file(
        merged_filename,
        as_attachment=True,
        download_name='merged_data.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

def process_uploaded_file(file, description_col, ref_no_col):
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        df = pd.read_excel(filename)

        # Filter columns based on keywords in description_col and ref_no_col
        filtered_description_cols = [col for col in df.columns if any(keyword.lower() in col.lower() for keyword in description_col)]
        filtered_ref_no_cols = [col for col in df.columns if any(keyword.lower() in col.lower() for keyword in ref_no_col)]

        if len(filtered_description_cols) > 0 and len(filtered_ref_no_cols) > 0:
            df['FileData'] = df[filtered_description_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1) + ' ' + df[filtered_ref_no_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
            return df

    return None

if __name__ == '__main__':
    app.run(debug=True)

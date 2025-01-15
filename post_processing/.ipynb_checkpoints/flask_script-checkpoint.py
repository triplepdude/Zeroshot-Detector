from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os

app = Flask(__name__)
AUTO_CORRECTED_FILE = "auto_corrected_mismatches.csv"
OUTPUT_FILE = "reviewed_mismatches.csv"
mismatches = pd.read_csv(AUTO_CORRECTED_FILE)
if os.path.exists(OUTPUT_FILE):
    reviewed = pd.read_csv(OUTPUT_FILE) #'resume' functionality
    reviewed_texts = set(reviewed['text'])
else:
    reviewed = pd.DataFrame(columns=['text', 'label', 'predicted_category', 'auto_corrected', 'user_category'])
    reviewed_texts = set()
to_review = mismatches[mismatches['auto_corrected'].isna()]
to_review = to_review[~to_review['text'].isin(reviewed_texts)] #only blank auto-correct columns

@app.route("/", methods=["GET", "POST"])
def review_mismatches():
    global reviewed, to_review
    if request.method == "POST":
        text = request.form["text"]
        user_category = request.form["category"]
        mismatches.loc[mismatches["text"] == text, "user_category"] = user_category
        new_rows = mismatches[mismatches["text"] == text]
        reviewed = pd.concat([reviewed, new_rows], ignore_index=True)
        reviewed.to_csv(OUTPUT_FILE, index=False)
        to_review = to_review[to_review["text"] != text]
        return redirect(url_for("review_mismatches"))
    if to_review.empty:
        return "All mismatches have been reviewed. Thank you!"
    current_row = to_review.iloc[0]
    return render_template("review_updated.html", text=current_row["text"], 
                           label=current_row["label"], 
                           predicted_category=current_row["predicted_category"])

if __name__ == "__main__":
    app.run(debug=True)

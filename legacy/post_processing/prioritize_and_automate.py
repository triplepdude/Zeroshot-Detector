
import pandas as pd
from collections import Counter

def auto_correct_mismatches(mismatches, rules):
    corrections = []
    for _, row in mismatches.iterrows():
        text = row['text']
        corrected_category = None

        # Apply rules for auto-correction
        for category, keywords in rules.items():
            if any(keyword in text.lower() for keyword in keywords):
                corrected_category = category
                break

        # Add to corrections or mark as unresolved
        corrections.append(corrected_category if corrected_category else None)
    
    mismatches['auto_corrected'] = corrections
    return mismatches

def resolve_frequent_mismatches(mismatches, output_file, threshold=10):
    # Group by text and count occurrences
    text_counts = Counter(mismatches['text'])
    frequent_texts = {text for text, count in text_counts.items() if count >= threshold}
    
    # Filter frequent mismatches
    frequent_mismatches = mismatches[mismatches['text'].isin(frequent_texts)]

    # Save frequent mismatches to a separate file for review
    frequent_mismatches.to_csv(output_file, index=False)
    print(f"Frequent mismatches saved to {output_file}.")

if __name__ == "__main__":
    mismatches_file = "mismatches.csv"
    auto_corrected_file = "auto_corrected_mismatches.csv"
    frequent_file = "frequent_mismatches.csv"

    # Load mismatches
    mismatches = pd.read_csv(mismatches_file)

    # Define rules for auto-correction
    rules = {
        "ground_vehicles": ["tank", "jeep", "armored vehicle", "truck", "humvee"],
        "maritime": ["ship", "submarine", "boat", "naval", "vessel"],
        "weapons": ["gun", "rifle", "missile", "grenade", "artillery"],
        "aircraft": ["plane", "helicopter", "jet", "drone", "aircraft"]
    }

    # Apply automated corrections
    mismatches = auto_correct_mismatches(mismatches, rules)
    mismatches.to_csv(auto_corrected_file, index=False)
    print(f"Auto-corrected mismatches saved to {auto_corrected_file}.")

    # Handle frequent mismatches
    resolve_frequent_mismatches(mismatches, frequent_file, threshold=10)

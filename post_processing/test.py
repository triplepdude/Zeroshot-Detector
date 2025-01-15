import pandas as pd
import os

def prompt_for_category(mismatches_file, output_file):
    # Load the mismatches file
    mismatches = pd.read_csv(mismatches_file)
    
    # If the output file exists, load progress and continue from where it was left off
    if os.path.exists(output_file):
        resolved = pd.read_csv(output_file)
        resolved_texts = set(resolved['text'])
        print(f"Resuming progress. {len(resolved_texts)} items already resolved.")
    else:
        resolved = pd.DataFrame(columns=['text', 'label', 'predicted_category', 'user_category'])
        resolved_texts = set()

    # Iterate over mismatches and prompt user for category
    for _, row in mismatches.iterrows():
        if row['text'] in resolved_texts:
            continue  # Skip already resolved items

        print(f"Text: {row['text']}")
        print(f"Label: {row['label']}, Predicted Category: {row['predicted_category']}")
        user_input = input("Enter the correct category: ")
        
        # Add the user's response to the resolved DataFrame
        resolved = pd.concat([
            resolved,
            pd.DataFrame([{
                'text': row['text'],
                'label': row['label'],
                'predicted_category': row['predicted_category'],
                'user_category': user_input
            }])
        ], ignore_index=True)

        # Save progress to the output file after every entry
        resolved.to_csv(output_file, index=False)
        print("Progress saved.")

if __name__ == "__main__":
    mismatches_file = "text_with_openAI_Validation.csv"
    output_file = "resolved_mismatches.csv"
    
    # Find mismatches between label and predicted_category
    data = pd.read_csv(mismatches_file)
    mismatches = data[data['label'] != data['predicted_category']]
    
    # Save mismatches to a file
    mismatches_file = "mismatches.csv"
    mismatches.to_csv(mismatches_file, index=False)
    print(f"Mismatches saved to {mismatches_file}.")
    
    # Prompt user for corrections and save results
    prompt_for_category(mismatches_file, output_file)

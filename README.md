# Current State -- This repository is more a collection of files rather than a functional tool offering.
# Zeroshot-Detector
Zero-Shot Classification models to process videos (.mp4 format) in MINO Buckets and extract every 30th frame, then processing via Florence2 and Bart-MNLI to determine text label and frame category. Used for an Independent study, resulting in this paper: https://hdl.handle.net/20.500.14216/1718 
## Current State
## Usage
Edit the Recursive Florence 2 + Bart + Confidence.py file to include your bucket names and MINO endpoint. ensure the endpoint does not require authentication, or modify to allow for tls. run via python3. Optionally, you can change the category strings and/or the prompts for better performance.
### Future work
I plan on coming back to this when I undergo the coursework to better put together my own trained model with adjusted weights and performance specific to this application rather than relying on massively clunky models like these.

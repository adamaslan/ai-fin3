# SSH

# Install everything in one go:
sudo apt update && sudo apt install -y python3-pip && pip3 install yfinance pandas numpy torch scikit-learn google-cloud-storage --user

# Set your bucket (replace YOUR_BUCKET_NAME):
echo 'export GCS_BUCKET="YOUR_BUCKET_NAME"' >> ~/.bashrc && source ~/.bashrc

# Create the Python file (then paste in the code):
nano stock_predictor.py

# Set up cron job:
(crontab -l 2>/dev/null; echo "0 15 * * 1-5 cd ~ && python3 stock_predictor.py >> predictions.log 2>&1") | crontab -

# Test it:
python3 stock_predictor.py
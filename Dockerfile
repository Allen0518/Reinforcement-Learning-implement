FROM sonoisa/deep-learning-coding:pytorch1.6.0_tensorflow2.3.0

# Copy the requirements file
COPY requirements.txt /app/train/

# Copy the training scripts
COPY train/agent.py train/train.py /app/train/

WORKDIR /app/train

# Install the required Python packages
RUN pip install -r requirements.txt

# Run the training script
CMD ["python", "train.py"]



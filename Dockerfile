# 1. Use an official Python runtime as a parent image
# Using python:3.10-slim for a good balance of features and size.
# Adjust the Python version if your bot specifically needs another one.
FROM python:3.10-slim

# 2. Set environment variables
#    - PYTHONUNBUFFERED: Prevents Python output from being buffered, good for logs.
#    - PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk.
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

#    - PORT: Render will set this environment variable for Web Services.
#      Your application should listen on this port if it's a web service.
#      For background workers, this isn't strictly necessary unless your app uses it.
#      We'll default it here, but Render's value will override.
ENV PORT=10000

# 3. Set the working directory in the container
WORKDIR /app

# 4. Install system dependencies (if any)
# For this bot, we don't have explicit system dependencies beyond what python:slim provides.
# If you were using libraries that needed, e.g., build-essentials or specific C libraries,
# you'd add them here:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# 5. Copy the requirements file and install Python dependencies
#    This step is done separately to leverage Docker's layer caching.
#    If requirements.txt doesn't change, this layer won't be rebuilt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code into the container
COPY . .

# 7. Expose the port the app runs on (for Web Services)
#    This is documentary; Render uses the PORT env var to route traffic.
EXPOSE ${PORT}

# 8. Define the command to run your application
#    This command will be executed when the container starts.
#    Render will also use this as the "Start Command" unless you override it
#    in the Render dashboard.
#    Ensure your bot.py's `if __name__ == "__main__":` block correctly
#    chooses between webhook or polling mode based on environment variables
#    (like USE_WEBHOOKS and WEBHOOK_URL).
CMD ["python", "telegram_bot.py"]
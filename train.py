import os
from datetime import datetime

from absl import app, flags, logging
from ml_collections.config_flags import config_flags


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("experiment_configs")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def main(_):
    pass

if __name__ == "__main__":
    app.run(main)
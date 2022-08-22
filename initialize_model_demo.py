import os
from datetime import datetime

from absl import app, flags, logging
from ml_collections.config_flags import config_flags

from mobilevit.models.mobilevit import get_training_model

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("experiment_configs")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main(_):
    model = get_training_model(
        model_name=FLAGS.experiment_configs.model_name,
        image_shape=(
            FLAGS.experiment_configs.image_height,
            FLAGS.experiment_configs.image_width,
            FLAGS.experiment_configs.image_channels,
        ),
        num_classes=FLAGS.experiment_configs.num_classes,
    )

    # model.summary()


if __name__ == "__main__":
    app.run(main)

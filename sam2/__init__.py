# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from hydra import initialize_config_dir

initialize_config_dir(
    config_dir=str(Path(__file__).resolve().parent.parent), version_base="1.2"
)

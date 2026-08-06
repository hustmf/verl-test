# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from verl.utils import device as device_module
from verl.utils.device import get_superpod_id

NPU_SMI_SPOD_OUTPUT = """
+-------------------------------------------------------------------------------------------+
| npu-smi 25.3.rc1                            Version: 25.3.rc1                           |
+-------------------------------+-----------------------------------------------------------+
| Super Pod ID                 : 3          | Pod ID : 1                                    |
+-------------------------------+-----------------------------------------------------------+
"""


@pytest.fixture(autouse=True)
def clear_superpod_id_cache():
    get_superpod_id.cache_clear()
    yield
    get_superpod_id.cache_clear()


def _fake_platform(device_name: str):
    platform = MagicMock()
    platform.device_name = device_name
    return platform


def _completed_process(stdout: str):
    result = MagicMock()
    result.stdout = stdout
    return result


def test_returns_none_on_non_npu_platform():
    with (
        patch.object(device_module, "get_platform", return_value=_fake_platform("cuda")),
        patch.object(device_module.subprocess, "run") as mock_run,
    ):
        assert get_superpod_id() is None
        mock_run.assert_not_called()


def test_parses_super_pod_id():
    with (
        patch.object(device_module, "get_platform", return_value=_fake_platform("npu")),
        patch.object(device_module.subprocess, "run", return_value=_completed_process(NPU_SMI_SPOD_OUTPUT)) as mock_run,
    ):
        assert get_superpod_id() == 3
        mock_run.assert_called_once_with(
            ["npu-smi", "info", "-t", "spod-info", "-i", "0", "-c", "0"],
            capture_output=True,
            text=True,
            check=True,
        )


def test_result_is_cached():
    with (
        patch.object(device_module, "get_platform", return_value=_fake_platform("npu")),
        patch.object(device_module.subprocess, "run", return_value=_completed_process(NPU_SMI_SPOD_OUTPUT)) as mock_run,
    ):
        assert get_superpod_id() == 3
        assert get_superpod_id() == 3
        assert mock_run.call_count == 1


def test_returns_none_on_command_failure():
    with (
        patch.object(device_module, "get_platform", return_value=_fake_platform("npu")),
        patch.object(
            device_module.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(returncode=1, cmd="npu-smi"),
        ),
    ):
        assert get_superpod_id() is None


def test_returns_none_when_npu_smi_missing():
    with (
        patch.object(device_module, "get_platform", return_value=_fake_platform("npu")),
        patch.object(device_module.subprocess, "run", side_effect=FileNotFoundError("npu-smi")),
    ):
        assert get_superpod_id() is None


def test_returns_none_on_unparseable_output():
    with (
        patch.object(device_module, "get_platform", return_value=_fake_platform("npu")),
        patch.object(device_module.subprocess, "run", return_value=_completed_process("some unrelated npu-smi output")),
    ):
        assert get_superpod_id() is None

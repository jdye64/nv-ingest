# Copyright (c) 2024, NVIDIA CORPORATION.
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


def bytesfromhex(hex_input):
    """
    Function to convert hex to bytes.

    Parameters
    ----------
    hex_input : hex
        Hex string to store bytes in cuDF.

    Returns
    -------
    bytes
        Hex encoded object converted to bytes.
    """

    return bytes.fromhex(hex_input)


def hexfrombytes(bytes_input):
    """
    Function to bytes to hex string.

    Parameters
    ----------
    bytes_input : bytes
        Raw bytes of object.

    Returns
    -------
    hex
        Hex string to store bytes in cuDF.
    """

    return bytes_input.hex()
